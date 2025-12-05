#!/bin/bash
# Orchestrator 실행 스크립트 (원격 서버 172.16.162.40에서 실행)
# 이 스크립트는 원격 서버에서 Orchestrator를 시작하는 데 사용됩니다

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR" && pwd)"
cd "$PROJECT_DIR"

CONFIG_FILE="${1:-configs/test_scalerRL.yaml}"
ORCH_HOST="${ORCH_HOST:-0.0.0.0}"  # 모든 인터페이스에서 접근 가능
ORCH_PORT="${ORCH_PORT:-59888}"
RUN_ID="${RUN_ID:-$(date +"%Y%m%d_%H%M%S")_orch}"

LOG_DIR="logs/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Orchestrator 시작"
echo "Run ID: $RUN_ID"
echo "Config: $CONFIG_FILE"
echo "Host: $ORCH_HOST"
echo "Port: $ORCH_PORT"
echo "Log Dir: $LOG_DIR"
echo "========================================="

# 환경 변수 설정
export RUN_ID
export ORCH_HOST
export ORCH_PORT

# PID 파일
ORCH_PID_FILE="$LOG_DIR/orchestrator.pid"

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
            if [ -f "$ORCH_PID_FILE" ]; then
                ORCH_PID=$(cat "$ORCH_PID_FILE" 2>/dev/null || echo "")
                if [ -n "$ORCH_PID" ] && kill -0 "$ORCH_PID" 2>/dev/null; then
                    curl -s -X POST "http://127.0.0.1:${ORCH_PORT}/stop" > /dev/null 2>&1 || true
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
        if [ -f "$ORCH_PID_FILE" ]; then
            ORCH_PID=$(cat "$ORCH_PID_FILE" 2>/dev/null || echo "")
            if [ -n "$ORCH_PID" ] && kill -0 "$ORCH_PID" 2>/dev/null; then
                curl -s -X POST "http://127.0.0.1:${ORCH_PORT}/stop" > /dev/null 2>&1 || true
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

# Orchestrator 시작
echo ""
echo "Orchestrator 시작 중..."
python ralo_cli.py orch \
    --config "$CONFIG_FILE" \
    --orch-port "$ORCH_PORT" \
    --run-id "$RUN_ID" \
    --log-dir "$LOG_DIR" \
    > "$LOG_DIR/orchestrator.log" 2>&1 &
ORCH_PID=$!
echo $ORCH_PID > "$ORCH_PID_FILE"
echo "Orchestrator PID: $ORCH_PID"

# Orchestrator가 시작될 때까지 대기 (초기 가중치 저장에 시간이 걸릴 수 있음)
echo "Orchestrator 시작 대기 중... (최대 600초)"
echo "  초기 가중치 저장 중일 수 있습니다 (약 14GB, 시간이 걸릴 수 있습니다)"
for i in {1..600}; do
    # 프로세스가 살아있는지 확인
    if ! kill -0 "$ORCH_PID" 2>/dev/null; then
        echo ""
        echo "오류: Orchestrator 프로세스가 종료되었습니다"
        echo "로그 확인: tail -50 $LOG_DIR/orchestrator.log"
        exit 1
    fi
    
    # 서버가 응답하는지 확인
    if curl -s --connect-timeout 2 "http://127.0.0.1:${ORCH_PORT}/stats" > /dev/null 2>&1; then
        echo ""
        echo "Orchestrator 준비 완료!"
        echo ""
        echo "========================================="
        echo "Orchestrator 실행 중"
        echo "  URL: http://${ORCH_HOST}:${ORCH_PORT}"
        echo "  PID: $ORCH_PID"
        echo "  로그: $LOG_DIR/orchestrator.log"
        echo ""
        echo "상태 확인:"
        echo "  curl http://${ORCH_HOST}:${ORCH_PORT}/stats"
        echo "  curl http://${ORCH_HOST}:${ORCH_PORT}/eval/stats"
        echo ""
        echo "종료하려면 Ctrl+C를 누르세요 (종료 스크립트가 실행됩니다)"
        echo "========================================="
        break
    fi
    
    # 진행 상황 표시 (10초마다)
    if [ $((i % 10)) -eq 0 ]; then
        echo "  대기 중... (${i}/600초) - 로그 확인: tail -f $LOG_DIR/orchestrator.log"
    fi
    
    if [ $i -eq 600 ]; then
        echo ""
        echo "경고: Orchestrator가 600초 내에 시작되지 않았습니다"
        echo "프로세스가 실행 중인지 확인: ps aux | grep $ORCH_PID"
        echo "로그 확인: tail -50 $LOG_DIR/orchestrator.log"
        echo ""
        echo "프로세스가 실행 중이면 계속 대기하거나, 로그를 확인하세요"
        # 프로세스가 살아있으면 계속 실행하도록 함
        if kill -0 "$ORCH_PID" 2>/dev/null; then
            echo "프로세스는 실행 중입니다. 계속 대기합니다..."
        else
            exit 1
        fi
    fi
    sleep 1
done

# Orchestrator가 종료될 때까지 대기
wait $ORCH_PID

