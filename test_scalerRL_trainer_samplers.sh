#!/bin/bash
# 로컬 Orchestrator에 연결하는 Trainer 1개, Sampler 2개 실행 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
cd "$PROJECT_DIR"

CONFIG_FILE="${1:-configs/test_scalerRL.yaml}"
ORCH_HOST="${ORCH_HOST:-172.16.162.40}"  # 로컬 Orchestrator
ORCH_PORT="${ORCH_PORT:-59888}"
ORCH_URL="http://${ORCH_HOST}:${ORCH_PORT}"

# GPU 할당
TRAINER_GPU="${TRAINER_GPU:-0}"
SAMPLER1_GPU="${SAMPLER1_GPU:-1}"
SAMPLER2_GPU="${SAMPLER2_GPU:-2}"

# Run ID 생성 (Orchestrator와 동일한 형식 사용 가능)
RUN_ID="${RUN_ID:-$(date +"%Y%m%d_%H%M%S")_test_scalerRL_dist}"
LOG_DIR="logs/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Trainer & Sampler 시작"
echo "Run ID: $RUN_ID"
echo "Config: $CONFIG_FILE"
echo "Orchestrator: $ORCH_URL"
echo "Log Dir: $LOG_DIR"
echo ""
echo "GPU 할당:"
echo "  Trainer: GPU $TRAINER_GPU"
echo "  Sampler 1: GPU $SAMPLER1_GPU"
echo "  Sampler 2: GPU $SAMPLER2_GPU"
echo "========================================="

# 환경 변수 설정
export RUN_ID
export ORCH_SERVER="$ORCH_URL"
export WANDB_DISABLED="${WANDB_DISABLED:-false}"

# PID 파일 경로
TRAINER_PID_FILE="$LOG_DIR/trainer.pid"
SAMPLER1_PID_FILE="$LOG_DIR/sampler1.pid"
SAMPLER2_PID_FILE="$LOG_DIR/sampler2.pid"

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
            if [ -f "$SAMPLER2_PID_FILE" ]; then
                SAMPLER2_PID=$(cat "$SAMPLER2_PID_FILE" 2>/dev/null || echo "")
                if [ -n "$SAMPLER2_PID" ] && kill -0 "$SAMPLER2_PID" 2>/dev/null; then
                    kill "$SAMPLER2_PID" 2>/dev/null || true
                    sleep 1
                    kill -9 "$SAMPLER2_PID" 2>/dev/null || true
                fi
            fi
            if [ -f "$SAMPLER1_PID_FILE" ]; then
                SAMPLER1_PID=$(cat "$SAMPLER1_PID_FILE" 2>/dev/null || echo "")
                if [ -n "$SAMPLER1_PID" ] && kill -0 "$SAMPLER1_PID" 2>/dev/null; then
                    kill "$SAMPLER1_PID" 2>/dev/null || true
                    sleep 1
                    kill -9 "$SAMPLER1_PID" 2>/dev/null || true
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
        }
    else
        echo "경고: 종료 스크립트를 찾을 수 없습니다: $STOP_SCRIPT"
        echo "      (예상 경로: $PROJECT_DIR/scripts/stop_all.sh)"
        echo "직접 종료합니다..."
        # 폴백: 직접 종료
        if [ -f "$SAMPLER2_PID_FILE" ]; then
            SAMPLER2_PID=$(cat "$SAMPLER2_PID_FILE" 2>/dev/null || echo "")
            if [ -n "$SAMPLER2_PID" ] && kill -0 "$SAMPLER2_PID" 2>/dev/null; then
                kill "$SAMPLER2_PID" 2>/dev/null || true
                sleep 1
                kill -9 "$SAMPLER2_PID" 2>/dev/null || true
            fi
        fi
        if [ -f "$SAMPLER1_PID_FILE" ]; then
            SAMPLER1_PID=$(cat "$SAMPLER1_PID_FILE" 2>/dev/null || echo "")
            if [ -n "$SAMPLER1_PID" ] && kill -0 "$SAMPLER1_PID" 2>/dev/null; then
                kill "$SAMPLER1_PID" 2>/dev/null || true
                sleep 1
                kill -9 "$SAMPLER1_PID" 2>/dev/null || true
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
    fi
    
    echo "종료 완료"
    echo "========================================="
}

# 시그널 핸들러 등록
trap cleanup EXIT INT TERM

# Orchestrator 연결 확인
echo ""
echo "Orchestrator 연결 확인 중 ($ORCH_URL)..."
ORCH_READY=false
for i in {1..10}; do
    if curl -s --connect-timeout 2 "$ORCH_URL/stats" > /dev/null 2>&1; then
        echo "Orchestrator 연결 성공!"
        ORCH_READY=true
        break
    fi
    if [ $i -eq 10 ]; then
        echo "오류: Orchestrator에 연결할 수 없습니다 ($ORCH_URL)"
        echo "Orchestrator가 실행 중인지 확인하세요: curl $ORCH_URL/stats"
        exit 1
    fi
    sleep 1
done

# 1. Trainer 시작 (GPU 0)
echo ""
echo "1. Trainer 시작 중 (GPU $TRAINER_GPU)..."
CUDA_VISIBLE_DEVICES=$TRAINER_GPU python ralo_cli.py train \
    --config "$CONFIG_FILE" \
    --orchestrator "$ORCH_URL" \
    --run-id "$RUN_ID" \
    --log-dir "$LOG_DIR" \
    > "$LOG_DIR/trainer.log" 2>&1 &
TRAINER_PID=$!
echo $TRAINER_PID > "$TRAINER_PID_FILE"
echo "Trainer PID: $TRAINER_PID (GPU: $TRAINER_GPU)"

# Trainer가 등록될 때까지 잠시 대기
sleep 5

# 2. Sampler 1 시작 (GPU 1)
echo ""
echo "2. Sampler 1 시작 중 (GPU $SAMPLER1_GPU)..."
CUDA_VISIBLE_DEVICES=$SAMPLER1_GPU GEN_DEVICES=$SAMPLER1_GPU python ralo_cli.py gen \
    --config "$CONFIG_FILE" \
    --orchestrator "$ORCH_URL" \
    --run-id "$RUN_ID" \
    --log-dir "$LOG_DIR" \
    --log-file "sampler1.log" \
    > "$LOG_DIR/sampler1.log" 2>&1 &
SAMPLER1_PID=$!
echo $SAMPLER1_PID > "$SAMPLER1_PID_FILE"
echo "Sampler 1 PID: $SAMPLER1_PID (GPU: $SAMPLER1_GPU)"

# 3. Sampler 2 시작 (GPU 2)
echo ""
echo "3. Sampler 2 시작 중 (GPU $SAMPLER2_GPU)..."
CUDA_VISIBLE_DEVICES=$SAMPLER2_GPU GEN_DEVICES=$SAMPLER2_GPU python ralo_cli.py gen \
    --config "$CONFIG_FILE" \
    --orchestrator "$ORCH_URL" \
    --run-id "$RUN_ID" \
    --log-dir "$LOG_DIR" \
    --log-file "sampler2.log" \
    > "$LOG_DIR/sampler2.log" 2>&1 &
SAMPLER2_PID=$!
echo $SAMPLER2_PID > "$SAMPLER2_PID_FILE"
echo "Sampler 2 PID: $SAMPLER2_PID (GPU: $SAMPLER2_GPU)"

echo ""
echo "========================================="
echo "모든 프로세스 시작 완료!"
echo "Run ID: $RUN_ID"
echo "Log directory: $LOG_DIR"
echo ""
echo "PIDs:"
echo "  Trainer (GPU $TRAINER_GPU): $TRAINER_PID"
echo "  Sampler 1 (GPU $SAMPLER1_GPU): $SAMPLER1_PID"
echo "  Sampler 2 (GPU $SAMPLER2_GPU): $SAMPLER2_PID"
echo ""
echo "모니터링:"
echo "  tail -f $LOG_DIR/trainer.log"
echo "  tail -f $LOG_DIR/sampler1.log"
echo "  tail -f $LOG_DIR/sampler2.log"
echo "  curl $ORCH_URL/stats"
echo "  curl $ORCH_URL/eval/stats"
echo "  curl $ORCH_URL/eval/fit"
echo ""
        echo "종료하려면 Ctrl+C를 누르세요 (종료 스크립트가 실행됩니다)"
echo "========================================="

# 프로세스 모니터링
TRAINER_ALIVE=true
SAMPLER1_ALIVE=true
SAMPLER2_ALIVE=true

while true; do
    sleep 5
    
    # Trainer 체크
    if ! kill -0 "$TRAINER_PID" 2>/dev/null; then
        if [ "$TRAINER_ALIVE" = true ]; then
            echo "[WARNING] Trainer가 종료되었습니다"
            TRAINER_ALIVE=false
        fi
    fi
    
    # Sampler 1 체크
    if ! kill -0 "$SAMPLER1_PID" 2>/dev/null; then
        if [ "$SAMPLER1_ALIVE" = true ]; then
            echo "[WARNING] Sampler 1이 종료되었습니다"
            SAMPLER1_ALIVE=false
        fi
    fi
    
    # Sampler 2 체크
    if ! kill -0 "$SAMPLER2_PID" 2>/dev/null; then
        if [ "$SAMPLER2_ALIVE" = true ]; then
            echo "[WARNING] Sampler 2가 종료되었습니다"
            SAMPLER2_ALIVE=false
        fi
    fi
    
    # 모든 프로세스가 종료되면 루프 종료
    if [ "$TRAINER_ALIVE" = false ] && [ "$SAMPLER1_ALIVE" = false ] && [ "$SAMPLER2_ALIVE" = false ]; then
        echo ""
        echo "모든 프로세스가 종료되었습니다"
        break
    fi
    
    # 상태 확인
    if curl -s --connect-timeout 2 "$ORCH_URL/stats" > /dev/null 2>&1; then
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

