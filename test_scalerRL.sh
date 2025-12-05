#!/bin/bash
# 최소 단위 테스트 스크립트: sampler, trainer, orchestrator가 함께 작동하며 scalerRL 공식과 벤치마크 수행
# 적은 GPU와 시간으로도 실행 가능

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
cd "$PROJECT_DIR"

CONFIG_FILE="${1:-configs/test_scalerRL.yaml}"
ORCH_PORT="${ORCH_PORT:-59888}"
ORCH_URL="http://127.0.0.1:${ORCH_PORT}"

# Run ID 생성
RUN_ID=$(date +"%Y%m%d_%H%M%S")_test_scalerRL
LOG_DIR="logs/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "ScalerRL 단위 테스트 시작"
echo "Run ID: $RUN_ID"
echo "Config: $CONFIG_FILE"
echo "Log Dir: $LOG_DIR"
echo "========================================="

# 환경 변수 설정
export RUN_ID
export ORCH_SERVER="$ORCH_URL"
export WANDB_DISABLED="${WANDB_DISABLED:-false}"

# PID 파일 경로
ORCH_PID_FILE="$LOG_DIR/orchestrator.pid"
TRAINER_PID_FILE="$LOG_DIR/trainer.pid"
SAMPLER_PID_FILE="$LOG_DIR/sampler.pid"

# 정리 함수 - 종료 스크립트 호출
cleanup() {
    echo ""
    echo "========================================="
    echo "종료 스크립트 실행 중..."
    
    # 종료 스크립트 호출 (절대 경로 사용)
    STOP_SCRIPT="$PROJECT_DIR/scripts/stop_all.sh"
    if [ -f "$STOP_SCRIPT" ]; then
        echo "종료 스크립트 실행: $STOP_SCRIPT --log-dir $LOG_DIR"
        bash "$STOP_SCRIPT" --log-dir "$LOG_DIR" || {
            echo "경고: 종료 스크립트 실행 실패. 수동으로 종료합니다..."
            # 폴백: 직접 종료
            if [ -f "$SAMPLER_PID_FILE" ]; then
                SAMPLER_PID=$(cat "$SAMPLER_PID_FILE" 2>/dev/null || echo "")
                if [ -n "$SAMPLER_PID" ] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
                    kill "$SAMPLER_PID" 2>/dev/null || true
                    sleep 1
                    kill -9 "$SAMPLER_PID" 2>/dev/null || true
                fi
            fi
            if [ -f "$TRAINER_PID_FILE" ]; then
                TRAINER_PID=$(cat "$TRAINER_PID_FILE" 2>/dev/null || echo "")
                if [ -n "$TRAINER_PID" ] && kill -0 "$TRAINER_PID" 2>/dev/null; then
                    kill "$TRAINER_PID" 2>/dev/null || true
                    sleep 1
                    kill -9 "$TRAINER_PID" 2>/dev/null || true
                fi
            fi
            if [ -f "$ORCH_PID_FILE" ]; then
                ORCH_PID=$(cat "$ORCH_PID_FILE" 2>/dev/null || echo "")
                if [ -n "$ORCH_PID" ] && kill -0 "$ORCH_PID" 2>/dev/null; then
                    curl -s -X POST "$ORCH_URL/stop" > /dev/null 2>&1 || true
                    sleep 2
                    kill "$ORCH_PID" 2>/dev/null || true
                    sleep 1
                    kill -9 "$ORCH_PID" 2>/dev/null || true
                fi
            fi
        }
    else
        echo "경고: 종료 스크립트를 찾을 수 없습니다: $STOP_SCRIPT"
        echo "      (예상 경로: $PROJECT_DIR/scripts/stop_all.sh)"
        echo "직접 종료합니다..."
        # 폴백: 직접 종료
        if [ -f "$SAMPLER_PID_FILE" ]; then
            SAMPLER_PID=$(cat "$SAMPLER_PID_FILE" 2>/dev/null || echo "")
            if [ -n "$SAMPLER_PID" ] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
                kill "$SAMPLER_PID" 2>/dev/null || true
                sleep 1
                kill -9 "$SAMPLER_PID" 2>/dev/null || true
            fi
        fi
        if [ -f "$TRAINER_PID_FILE" ]; then
            TRAINER_PID=$(cat "$TRAINER_PID_FILE" 2>/dev/null || echo "")
            if [ -n "$TRAINER_PID" ] && kill -0 "$TRAINER_PID" 2>/dev/null; then
                kill "$TRAINER_PID" 2>/dev/null || true
                sleep 1
                kill -9 "$TRAINER_PID" 2>/dev/null || true
            fi
        fi
        if [ -f "$ORCH_PID_FILE" ]; then
            ORCH_PID=$(cat "$ORCH_PID_FILE" 2>/dev/null || echo "")
            if [ -n "$ORCH_PID" ] && kill -0 "$ORCH_PID" 2>/dev/null; then
                curl -s -X POST "$ORCH_URL/stop" > /dev/null 2>&1 || true
                sleep 2
                kill "$ORCH_PID" 2>/dev/null || true
                sleep 1
                kill -9 "$ORCH_PID" 2>/dev/null || true
            fi
        fi
    fi
    
    echo "종료 완료"
    echo "========================================="
}

# 시그널 핸들러 등록
trap cleanup EXIT INT TERM

# 1. Orchestrator 시작 (CPU만 사용, GPU 사용 안 함)
echo ""
echo "1. Orchestrator 시작 중... (CPU만 사용)"
# Orchestrator는 CPU에서만 실행되도록 CUDA_VISIBLE_DEVICES를 비워둠
CUDA_VISIBLE_DEVICES="" python ralo_cli.py orch \
    --config "$CONFIG_FILE" \
    --orch-port "$ORCH_PORT" \
    --run-id "$RUN_ID" \
    --log-dir "$LOG_DIR" \
    > "$LOG_DIR/orchestrator.log" 2>&1 &
ORCH_PID=$!
echo $ORCH_PID > "$ORCH_PID_FILE"
echo "Orchestrator PID: $ORCH_PID"

# Orchestrator가 시작될 때까지 대기 (체크포인트 로딩 시간 고려하여 충분히 대기)
echo "Orchestrator 시작 대기 중... (체크포인트 로딩 시간 포함, 최대 5분 대기)"
for i in {1..300}; do
    if curl -s "$ORCH_URL/stats" > /dev/null 2>&1; then
        echo "Orchestrator 준비 완료!"
        break
    fi
    if [ $i -eq 300 ]; then
        echo "오류: Orchestrator가 시작되지 않았습니다 (5분 초과)"
        exit 1
    fi
    # 진행 상황 표시 (30초마다)
    if [ $((i % 30)) -eq 0 ]; then
        echo "  대기 중... ($((i / 60))분 $((i % 60))초 경과)"
    fi
    sleep 1
done

# 2. Trainer 시작 (GPU 0 사용)
echo ""
echo "2. Trainer 시작 중... (GPU 0 사용)"
CUDA_VISIBLE_DEVICES=0 python ralo_cli.py train \
    --config "$CONFIG_FILE" \
    --orchestrator "$ORCH_URL" \
    --run-id "$RUN_ID" \
    --log-dir "$LOG_DIR" \
    > "$LOG_DIR/trainer.log" 2>&1 &
TRAINER_PID=$!
echo $TRAINER_PID > "$TRAINER_PID_FILE"
echo "Trainer PID: $TRAINER_PID (GPU 0)"

# Trainer가 등록될 때까지 잠시 대기
sleep 5

# 3. Sampler 시작 (GPU 1 사용, trainer와 분리)
echo ""
echo "3. Sampler 시작 중... (GPU 1 사용)"
GEN_DEVICES="${GEN_DEVICES:-1}"  # 기본값을 1로 변경 (trainer는 0 사용)
export GEN_DEVICES
CUDA_VISIBLE_DEVICES=1 python ralo_cli.py gen \
    --config "$CONFIG_FILE" \
    --orchestrator "$ORCH_URL" \
    --run-id "$RUN_ID" \
    --log-dir "$LOG_DIR" \
    > "$LOG_DIR/sampler.log" 2>&1 &
SAMPLER_PID=$!
echo $SAMPLER_PID > "$SAMPLER_PID_FILE"
echo "Sampler PID: $SAMPLER_PID (GPU: $GEN_DEVICES)"

echo ""
echo "========================================="
echo "모든 프로세스 시작 완료!"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo ""
echo "PIDs:"
echo "  Orchestrator: $ORCH_PID"
echo "  Trainer: $TRAINER_PID"
echo "  Sampler: $SAMPLER_PID"
echo ""
echo "모니터링:"
echo "  tail -f $LOG_DIR/orchestrator.log"
echo "  tail -f $LOG_DIR/trainer.log"
echo "  tail -f $LOG_DIR/sampler.log"
echo "  curl $ORCH_URL/stats"
echo "  curl $ORCH_URL/eval/stats"
echo "  curl $ORCH_URL/eval/fit"
echo ""
        echo "종료하려면 Ctrl+C를 누르세요 (종료 스크립트가 실행됩니다)"
echo "========================================="

# 프로세스 모니터링
ORCH_ALIVE=true
TRAINER_ALIVE=true
SAMPLER_ALIVE=true

while true; do
    sleep 5
    
    # Orchestrator 체크
    if ! kill -0 "$ORCH_PID" 2>/dev/null; then
        if [ "$ORCH_ALIVE" = true ]; then
            echo "[WARNING] Orchestrator가 종료되었습니다"
            ORCH_ALIVE=false
        fi
    fi
    
    # Trainer 체크
    if ! kill -0 "$TRAINER_PID" 2>/dev/null; then
        if [ "$TRAINER_ALIVE" = true ]; then
            echo "[WARNING] Trainer가 종료되었습니다"
            TRAINER_ALIVE=false
        fi
    fi
    
    # Sampler 체크
    if ! kill -0 "$SAMPLER_PID" 2>/dev/null; then
        if [ "$SAMPLER_ALIVE" = true ]; then
            echo "[WARNING] Sampler가 종료되었습니다"
            SAMPLER_ALIVE=false
        fi
    fi
    
    # 모든 프로세스가 종료되면 루프 종료
    if [ "$ORCH_ALIVE" = false ] && [ "$TRAINER_ALIVE" = false ] && [ "$SAMPLER_ALIVE" = false ]; then
        echo ""
        echo "모든 프로세스가 종료되었습니다"
        break
    fi
    
    # 상태 확인 (선택사항)
    if curl -s "$ORCH_URL/stats" > /dev/null 2>&1; then
        STATS=$(curl -s "$ORCH_URL/stats" 2>/dev/null || echo "{}")
        SHOULD_STOP=$(echo "$STATS" | grep -o '"should_stop":[^,}]*' | grep -o 'true' || echo "")
        if [ "$SHOULD_STOP" = "true" ]; then
            echo ""
            echo "Orchestrator가 종료 신호를 받았습니다"
            sleep 5
            break
        fi
    fi
done

echo ""
echo "========================================="
echo "테스트 완료!"
echo "결과 확인:"
echo "  - 로그: $LOG_DIR/"
echo "  - 평가 결과: curl $ORCH_URL/eval/results?version=0"
echo "  - ScaleRL 피팅: curl $ORCH_URL/eval/fit"
echo "  - 컴퓨팅 통계: curl $ORCH_URL/compute/stats"
echo "========================================="

