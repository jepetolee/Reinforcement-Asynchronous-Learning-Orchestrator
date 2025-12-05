RALO — Reinforcement Asynchronous Learning Orchestrator

RALO는 중앙 집중식 Orchestrator를 중심으로 구축된 강화학습을 위한 비동기 Trainer–Sampler 시스템입니다. Orchestrator는 문제 큐, 샘플 큐, gradient 집계, optimizer, 가중치/버전 저장을 담당합니다. 여러 샘플러 노드가 생성한 데이터를 업로드하고 트레이너가 이를 소비해 학습하며, 샘플러는 최신 가중치를 서버에서 받아 즉시 반영합니다. 이 중앙 집중식 아키텍처는 일관된 모델 업데이트, 효율적인 리소스 관리, 확장 가능한 분산 학습을 보장합니다.

## 아키텍처

RALO는 4가지 핵심 구성요소로 이루어져 있습니다:

1. **Orchestrator**: 여러 Trainer 노드로부터 gradient를 수집하고, optimizer step을 수행하여 전역 모델을 업데이트합니다. A3C 스타일의 중앙 집중식 업데이트를 담당합니다.
2. **Orchestrator**: 샘플 데이터 수집, 간단한 메트릭 집계, 최신 가중치 제공 (Orchestrator가 업데이트한 가중치를 저장)
3. **Trainer(DDP)**: /get으로 배치 수신 → forward/backward 수행 → /gradient/upload_chunk + /gradient/upload_finalize로 gradient 전송 → 최신 가중치 pull
   - **DDP (Distributed Data Parallel)**: PyTorch의 분산 학습 방식. 여러 GPU/노드에서 동일한 모델을 병렬로 실행하며, 각 GPU가 서로 다른 데이터 배치를 처리합니다. `torchrun --nproc_per_node=N`으로 실행하면 N개의 프로세스가 각각 하나의 GPU를 담당합니다.
4. **Sampler(vLLM 기반)**: 프롬프트로 샘플 생성 → /upload로 서버에 전송 → 최신 가중치 pull
5. **멀티스레드 HTTP 서버**: Orchestrator는 멀티스레드 Bottle 서버(ThreadingMixIn을 사용한 WSGIRefServer)를 사용하여 여러 Trainer와 Sampler의 동시 요청을 처리하여 처리량을 향상시키고 타임아웃 문제를 줄입니다.

### 아키텍처 흐름

```
Problem dataset ─▶ Orchestrator (ProblemProvider)
Samplers ──(GET /problem/get)───────────┐
Samplers ──(POST /upload)───────────────┤
                                        ├─ SampleQueueManager ──(GET /get)── Trainers
Trainers ──(POST /gradient/upload_chunk + /gradient/upload_finalize)───▶ GradientAggregator ── optimizer/weights
Samplers & Trainers ──(GET /weights/*) / (POST /weights/*) to stay in sync
```

### Orchestrator를 사용하는 이유

RALO는 중앙 집중식 Orchestrator 아키텍처를 채택합니다. 이는 다음과 같은 이점을 제공합니다:

1. **중앙 집중식 Gradient Aggregation**: 여러 Trainer의 gradient를 Orchestrator에서 수집하고 평균화하여 전역 모델을 업데이트합니다. 이는 A3C 스타일의 중앙 집중식 업데이트로, 분산 학습의 일관성을 보장합니다.

2. **통합된 Sample Queue 관리**: Sampler가 생성한 샘플을 Orchestrator가 중앙에서 관리하고, Trainer에게 배분합니다. 이를 통해 샘플 생성과 소비의 흐름을 효율적으로 제어할 수 있습니다.

3. **가중치 버전 관리**: Orchestrator가 모델 가중치의 버전을 관리하여, 모든 Trainer와 Sampler가 동일한 버전의 가중치를 사용하도록 보장합니다.

4. **Problem 데이터 중앙화**: 학습 문제 데이터를 Orchestrator에서 관리하여, Sampler가 직접 데이터셋을 읽을 필요 없이 API를 통해 문제를 가져올 수 있습니다.

5. **확장성과 유연성**: Orchestrator를 통해 Trainer와 Sampler를 독립적으로 확장할 수 있으며, 각 컴포넌트의 역할이 명확히 분리됩니다.

### Orchestrator 아키텍처의 핵심 특징

RALO의 Orchestrator 아키텍처는 다음과 같은 특징을 가집니다:

- **중앙 집중식 Optimizer Step**: Orchestrator만 optimizer step을 수행합니다. Trainer는 gradient 계산 및 전송만 담당하며, optimizer step은 수행하지 않습니다.

- **Gradient 기반 업데이트**: Trainer는 `accum_steps`마다 gradient를 수집하여 Orchestrator로 전송합니다. Orchestrator는 `update_steps`개의 gradient가 모이면 평균화하고 optimizer step을 수행하여 전역 모델을 업데이트합니다.

- **Trainer 역할 분리**: Trainer의 역할은 Forward → Backward → **Gradient 전송** → 가중치 다운로드입니다. Optimizer step은 Orchestrator에서만 수행됩니다.

- **다중 Trainer 완전 지원**: A3C(Asynchronous Advantage Actor-Critic) 스타일의 구조로, 여러 비동기 Trainer 노드가 독립적으로 경험을 수집하고 중앙 Orchestrator가 전역 네트워크를 업데이트합니다.

- **전역 모델 일관성**: Orchestrator가 단일 전역 모델을 유지하므로, 모든 Trainer와 Sampler가 동일한 버전의 모델을 사용하여 일관성을 보장합니다.

### 디스크 기반 Gradient 저장 (Disk-Based Gradient Storage)

RALO는 대용량 모델(예: 7B+ 파라미터)을 처리할 때 Orchestrator의 RAM 부족 문제를 방지하기 위해 디스크 기반 gradient 저장 전략을 구현합니다.

**주요 이점:**
1.  **메모리 효율성**: 7B 모델의 gradient 100개 이상을 RAM에 저장하려면 수백 GB가 필요합니다. 디스크 기반 저장은 대기 중인 gradient 수와 관계없이 RAM 사용량을 일정하게 유지합니다.
2.  **안정성**: 장시간 학습 실행 중 Orchestrator의 OOM(Out Of Memory) 충돌을 방지합니다.
3.  **확장성**: 디스크 공간에 의해서만 제한되므로 더 큰 모델과 더 큰 배치 크기(더 많은 `update_steps`)로 확장할 수 있습니다.

**작동 방식:**

RALO는 대용량 모델을 효율적으로 처리하기 위해 **디스크 기반 gradient 저장 및 처리 파이프라인**을 사용합니다:

1. **청크 업로드**: 
   - Trainer는 큰 gradient를 청크로 분할(기본값: 50MB/청크)하고 `/gradient/upload_chunk` 엔드포인트를 통해 순차적으로 업로드합니다.
   - 각 청크는 RAM 누적을 방지하기 위해 즉시 `gradient_chunks_dir`에 디스크로 저장됩니다.
   - Orchestrator는 청크 메타데이터(upload_id, 청크 인덱스, 타임스탬프)를 메모리에 추적하지만, 실제 gradient 데이터는 디스크에 유지됩니다.

2. **재조립**:
   - gradient의 모든 청크가 수신되면, trainer는 `/gradient/upload_finalize` 엔드포인트를 호출합니다.
   - Orchestrator는 디스크에서 모든 청크 파일을 읽고, 순서대로 연결하여 완전한 gradient를 재조립합니다.
   - 재조립된 gradient는 메타데이터(worker_id, step_id, batch_id 등)와 함께 `gradient_storage_dir`에 단일 파일로 저장됩니다.
   - **청크 파일은 재조립 후 즉시 삭제**되어 디스크 공간을 확보합니다.

3. **점진적 누적**:
   - `update_steps`개의 gradient가 수집되면 optimizer step이 트리거됩니다.
   - Orchestrator는 `gradient_storage_dir`에서 gradient 파일을 **하나씩** 로드합니다.
   - 각 파일에 대해:
     - 디스크에서 gradient 로드
     - 모델 파라미터 gradient에 직접 누적(CPU에서)
     - 처리 후 **즉시 파일 삭제**
   - 이를 통해 메모리에는 한 번에 최대 하나의 gradient만 존재하도록 보장하여 RAM 사용량을 일정하게 유지합니다.

4. **비동기 처리**:
   - Optimizer step은 HTTP 요청을 블로킹하지 않도록 백그라운드 스레드에서 수행됩니다.
   - Orchestrator는 optimizer step을 처리하는 동안에도 새로운 gradient 업로드를 계속 수신할 수 있습니다.

5. **자동 정리**:
   - 오래된 청크 파일(`chunk_timeout`보다 오래된)은 디스크 공간 누수를 방지하기 위해 자동으로 정리됩니다.
   - 재조립된 gradient 파일은 optimizer step 중 처리 후 즉시 삭제됩니다.
   - 디스크 사용량은 `max_gradient_disk_mb`(기본값: 1TB)로 제한되며, 초과 시 가장 오래된 파일이 제거됩니다.

**설정 파라미터:**
- `gradient_chunks_dir` (기본값: 자동 생성): 업로드 중 gradient 청크 파일을 저장할 디렉토리. 기본값: `./orchestrator_gradient_chunks_{port}`
- `gradient_storage_dir` (기본값: 자동 생성): 재조립된 gradient 파일을 저장할 디렉토리. 기본값: `./orchestrator_gradients_{port}`
- `max_gradient_disk_mb` (기본값: 1024000.0MB = 1TB): 재조립된 gradient 파일 최대 디스크 사용량. 초과 시 가장 오래된 파일을 자동으로 제거합니다.
- `max_chunk_disk_mb` (기본값: 1024000.0MB = 1TB): Gradient 청크 파일 최대 디스크 사용량. 초과 시 가장 오래된 청크를 자동으로 제거합니다.

**왜 디스크 기반만 사용하는가:**
- **RAM 기반 gradient 저장은 제거되었습니다** 아키텍처를 단순화하고 일관된 동작을 보장하기 위해.
- 모든 gradient는 이제 디스크 기반 경로를 통해 처리되며, 이는 다음을 제공합니다:
  - **예측 가능한 메모리 사용**: `update_steps`나 모델 크기와 관계없이 RAM 사용량이 일정하게 유지됩니다.
  - **확장성**: OOM 오류 없이 100개 이상의 대기 중인 gradient를 처리할 수 있습니다.
  - **안정성**: 대용량 모델에서 RAM 기반 저장으로 발생할 수 있는 OOM 충돌을 방지합니다.

**성능 고려사항:**
- **I/O 오버헤드**: 디스크 기반 저장은 I/O 오버헤드를 도입하지만(~5-10% 느림) OOM 충돌을 방지합니다.
- **메모리 절약**: RAM 사용량은 `update_steps`와 관계없이 일정하게 유지되어 매우 큰 배치 크기로 확장할 수 있습니다.
- **디스크 공간**: 충분한 디스크 공간을 확보하세요 (권장: `update_steps × 평균_gradient_크기_mb × 2`).
- **디스크 속도**: 더 나은 성능을 위해 `gradient_chunks_dir`와 `gradient_storage_dir`에 빠른 SSD를 사용하세요.


## 구성 파일 기반 실행
- `configs/example.yaml`을 복사해 실험 설정(데이터셋, Sampler/Trainer 알고리즘, wandb 등)을 조정하세요.
- `python ralo_cli.py <orch|gen|train> --config my_exp.yaml` 형태로 실행하며, CLI 인자/환경변수가 YAML 값을 덮어씁니다.(우선순위: CLI > ENV > CONFIG > 기본값)
- Sampler/Trainer 알고리즘은 `ralo.algorithms.register_*` API로 등록할 수 있어 TreePO 이외의 알고리즘도 주입 가능합니다.
- 로깅은 `ExperimentLogger` 추상화(wandb/NoOp 제공)를 사용하므로 원하는 로거를 쉽게 교체할 수 있습니다.

## 빠른 시작

1. **구성 파일 준비**
   ```bash
   cp configs/example.yaml my_exp.yaml
   ```
   - `dataset` 섹션에서 Orchestrator가 큐에 넣을 HF 데이터셋과 필터를 결정합니다.
   - `sampler` / `trainer` `params` 값은 각 알고리즘(TreePO 등)에 그대로 전달됩니다.
   - `wandb`로 로깅 사용 여부와 프로젝트/런 이름을 설정합니다.

## 쉘 스크립트로 실행 (백그라운드 프로세스 권장)

쉘 스크립트를 사용하면 백그라운드에서 프로세스를 실행하고 자동으로 로그 파일에 저장할 수 있습니다. 모든 로그는 고유한 run_id를 가진 `logs/{run_id}/` 디렉토리에 저장됩니다.

### Orchestrator 시작

```bash
# 쉘 스크립트 사용 (권장)
./scripts/start_orch.sh \
  --config my_exp.yaml \
  --orch-port 59888 \
  --orch-host 0.0.0.0

# 또는 환경변수 사용
export ORCH_HOST=0.0.0.0
export ORCH_PORT=59888
./scripts/start_orch.sh --config my_exp.yaml
```

### Sampler 시작

```bash
# 쉘 스크립트 사용
./scripts/start_sampler.sh \
  --config my_exp.yaml \
  --orchestrator http://<server-ip>:59888 \
  --gen-devices 0,1,2,3 \
  --run-id <run_id>

# 또는 환경변수 사용
export ORCH_SERVER=http://<server-ip>:59888
export GEN_DEVICES=0,1,2,3
./scripts/start_sampler.sh --config my_exp.yaml --run-id <run_id>
```

### Trainer 시작

```bash
# 쉘 스크립트 사용 (분산 학습용 torchrun 포함)
./scripts/start_trainer.sh \
  --config my_exp.yaml \
  --orchestrator http://<server-ip>:59888 \
  --cuda-visible-devices 0,1,2,3 \
  --nproc-per-node 4 \
  --master-port 29501 \
  --skip-trainer-registration 1 \
  --torch-nccl-async-error-handling 1 \
  --run-id <run_id>

# 또는 환경변수 사용
export ORCH_SERVER=http://<server-ip>:59888
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SKIP_TRAINER_REGISTRATION=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
./scripts/start_trainer.sh \
  --config my_exp.yaml \
  --nproc-per-node 4 \
  --master-port 29501 \
  --run-id <run_id>
```

### 모든 프로세스 한번에 시작

```bash
# Orchestrator, Trainer, Sampler를 함께 시작
./scripts/start_all.sh \
  --config my_exp.yaml \
  --orch-port 59888 \
  --orch-host 0.0.0.0 \
  --orchestrator http://<server-ip>:59888 \
  --gen-devices 0,1,2,3 \
  --cuda-visible-devices 0,1,2,3 \
  --nproc-per-node 4 \
  --master-port 29501 \
  --skip-trainer-registration 1 \
  --torch-nccl-async-error-handling 1
```

### 프로세스 관리

```bash
# 모든 프로세스 상태 확인
./scripts/status.sh --log-dir logs/<run_id>

# 특정 run의 모든 프로세스 종료
./scripts/stop_all.sh --log-dir logs/<run_id>

# 모든 로그 디렉토리의 프로세스 종료 (와일드카드 확장)
./scripts/stop_all.sh --log-dir logs/*

# 패턴에 맞는 모든 프로세스 종료 (따옴표로 감싼 와일드카드)
./scripts/stop_all.sh --log-dir 'logs/20251120_*'
```

**로그 파일:**
- 로그는 각 프로세스별로 `logs/{run_id}/` 디렉토리에 저장됩니다:
  - `orchestrator.log` - Orchestrator stdout/stderr
  - `trainer.log` - Trainer stdout/stderr
  - `sampler.log` - Sampler stdout/stderr
- PID 파일: `orchestrator.pid`, `trainer.pid`, `sampler.pid`
- Run ID는 `run_id.txt`에 저장됩니다

## SLURM으로 실행 (클러스터 환경)

RALO는 HPC 클러스터에서 실행하기 위한 SLURM 스크립트를 제공합니다. 이 스크립트들은 작업 제출, 리소스 할당, 자동 로깅을 처리합니다.

### 사전 요구사항

- SLURM 워크로드 관리자
- CUDA 모듈 로드됨
- Python 가상 환경 활성화됨
- 모든 스크립트가 `scripts/` 디렉토리에 있음

### 모든 작업 제출 (권장)

모든 컴포넌트를 실행하는 가장 쉬운 방법은 `slurm_submit_all.sh` 스크립트를 사용하는 것입니다:

```bash
# 환경 변수 설정
export CONFIG_FILE=my_exp.yaml
export ORCHESTRATOR_URL=http://<orchestrator-host>:59888

# 모든 작업 제출 (orchestrator, trainer, sampler)
bash scripts/slurm_submit_all.sh
```

**Partition 및 GPU 설정 커스터마이징:**

환경 변수를 통해 partition과 GPU 설정을 오버라이드할 수 있습니다:

```bash
# Partition 및 GPU 타입 오버라이드
export SLURM_PARTITION=gpu3
export SLURM_GPU_TYPE=a6000ada
export SLURM_GPU_COUNT_TRAINER=4
export SLURM_GPU_COUNT_SAMPLER=4
# 참고: Orchestrator는 CPU만 사용, GPU 불필요

# 커스텀 설정으로 제출
export CONFIG_FILE=my_exp.yaml
export ORCHESTRATOR_URL=http://<orchestrator-host>:59888
bash scripts/slurm_submit_all.sh
```

이 스크립트는 다음을 수행합니다:
1. 모든 컴포넌트에 대한 공유 `run_id` 생성
2. orchestrator 작업을 먼저 제출
3. orchestrator 시작을 위해 10초 대기
4. trainer 및 sampler 작업 제출
5. 모든 작업 ID를 로그 디렉토리에 저장

### 개별 작업 제출

개별 작업을 제출할 수도 있습니다:

#### Orchestrator

```bash
export CONFIG_FILE=my_exp.yaml
export RUN_ID=my_run_123
export ORCH_PORT=59888
export ORCH_HOST=0.0.0.0

# 기본값: partition=gpu4, CPU만 사용 (GPU 불필요)
sbatch scripts/slurm_orch.sh

# 또는 partition 오버라이드 (orchestrator는 CPU만 사용):
SLURM_PARTITION=cpu \
sbatch --partition=$SLURM_PARTITION scripts/slurm_orch.sh
```

#### Trainer

```bash
export CONFIG_FILE=my_exp.yaml
export RUN_ID=my_run_123
export ORCHESTRATOR_URL=http://<orchestrator-host>:59888
export NPROC_PER_NODE=4
export MASTER_PORT=29501

# 기본값: partition=gpu4, GPU=a6000:4
sbatch scripts/slurm_trainer.sh

# 또는 sbatch 옵션으로 partition과 GPU 오버라이드:
SLURM_PARTITION=gpu3 SLURM_GPU_TYPE=a6000ada SLURM_GPU_COUNT_TRAINER=4 \
sbatch --partition=$SLURM_PARTITION --gres=gpu:$SLURM_GPU_TYPE:$SLURM_GPU_COUNT_TRAINER scripts/slurm_trainer.sh
```

#### Sampler

```bash
export CONFIG_FILE=my_exp.yaml
export RUN_ID=my_run_123
export ORCHESTRATOR_URL=http://<orchestrator-host>:59888
export GEN_DEVICES=0,1,2,3

# 기본값: partition=gpu4, GPU=a6000:4
sbatch scripts/slurm_sampler.sh

# 또는 sbatch 옵션으로 partition과 GPU 오버라이드:
SLURM_PARTITION=gpu3 SLURM_GPU_TYPE=a6000ada SLURM_GPU_COUNT_SAMPLER=4 \
sbatch --partition=$SLURM_PARTITION --gres=gpu:$SLURM_GPU_TYPE:$SLURM_GPU_COUNT_SAMPLER scripts/slurm_sampler.sh
```

### SLURM 작업 종료

```bash
# 특정 run의 모든 작업 종료
bash scripts/slurm_stop.sh --log-dir logs/<run_id>

# 모든 로그 디렉토리의 모든 작업 종료 (와일드카드 확장)
bash scripts/slurm_stop.sh --log-dir logs/*

# 패턴에 맞는 모든 작업 종료 (따옴표로 감싼 와일드카드)
bash scripts/slurm_stop.sh --log-dir 'logs/20251120_*'
```

**참고:** 스크립트는 쉘에서 확장된 와일드카드(`logs/*`)와 따옴표로 감싼 패턴(`'logs/*'`) 모두를 지원합니다. 두 방법 모두 정상적으로 작동합니다.

### SLURM 스크립트 설정

모든 SLURM 스크립트는 다음 기본 설정을 사용합니다:

- **파티션**: `gpu4` (환경 변수 `SLURM_PARTITION`으로 오버라이드 가능)
- **노드**: 1
- **태스크당 CPU**: 4
- **GPU**: 
  - Orchestrator: **CPU만 사용** (GPU 불필요 - optimizer step을 CPU에서 수행)
  - Trainer: `a6000` 타입 GPU 4개, 노드당 4 프로세스 (환경 변수 `SLURM_GPU_TYPE` 및 `SLURM_GPU_COUNT_TRAINER`로 오버라이드 가능)
  - Sampler: `a6000` 타입 GPU 4개 (환경 변수 `SLURM_GPU_TYPE` 및 `SLURM_GPU_COUNT_SAMPLER`로 오버라이드 가능)

**커스터마이징 옵션:**

1. **`slurm_submit_all.sh`에서 환경 변수 사용 (권장)**: `slurm_submit_all.sh` 실행 전 환경 변수 설정:
   ```bash
   export SLURM_PARTITION=gpu3
   export SLURM_GPU_TYPE=a6000ada
   export SLURM_GPU_COUNT_TRAINER=4
   export SLURM_GPU_COUNT_SAMPLER=4
   # 참고: Orchestrator는 CPU만 사용, GPU 불필요
   bash scripts/slurm_submit_all.sh
   ```

2. **개별 스크립트에서 환경 변수 사용**: 개별 스크립트 제출 시 `sbatch` 옵션과 함께 환경 변수 사용:
   ```bash
   # Orchestrator용 (CPU만 사용)
   SLURM_PARTITION=cpu \
   sbatch --partition=$SLURM_PARTITION scripts/slurm_orch.sh
   
   # Trainer용
   SLURM_PARTITION=gpu3 SLURM_GPU_TYPE=a6000ada SLURM_GPU_COUNT_TRAINER=4 \
   sbatch --partition=$SLURM_PARTITION --gres=gpu:$SLURM_GPU_TYPE:$SLURM_GPU_COUNT_TRAINER scripts/slurm_trainer.sh
   
   # Sampler용
   SLURM_PARTITION=gpu3 SLURM_GPU_TYPE=a6000ada SLURM_GPU_COUNT_SAMPLER=4 \
   sbatch --partition=$SLURM_PARTITION --gres=gpu:$SLURM_GPU_TYPE:$SLURM_GPU_COUNT_SAMPLER scripts/slurm_sampler.sh
   ```

3. **스크립트 파일 편집**: 각 스크립트 파일(`slurm_orch.sh`, `slurm_trainer.sh`, `slurm_sampler.sh`)의 `#SBATCH` 지시문을 직접 수정.

### SLURM 로그 파일

- SLURM 출력: `slurm_logs/{process}.{node}.{jobid}.out`
- SLURM 오류: `slurm_logs/{process}.{node}.{jobid}.err`
- 애플리케이션 로그: `logs/{run_id}/{process}_node{nodeid}.log`
- 작업 ID: `logs/{run_id}/{process}_slurm_jobid.txt`
- SLURM 정보: `logs/{run_id}/{process}_slurm_info.txt`

### 환경 변수

SLURM 스크립트는 다음 환경 변수를 지원합니다:

**애플리케이션 설정:**
- `CONFIG_FILE`: YAML 설정 파일 경로 (기본값: `my_exp.yaml`)
- `LOG_DIR`: 로그 디렉토리 (기본값: `logs/{run_id}`)
- `RUN_ID`: 실행 ID (제공되지 않으면 자동 생성)
- `ORCHESTRATOR_URL`: Orchestrator URL (trainer/sampler에 필요)
- `ORCH_PORT`: Orchestrator 포트 (기본값: 59888)
- `ORCH_HOST`: Orchestrator 호스트 (기본값: 0.0.0.0)
- `NPROC_PER_NODE`: Trainer의 노드당 프로세스 수 (기본값: 4)
- `MASTER_PORT`: 분산 학습용 마스터 포트 (기본값: 29501)
- `GEN_DEVICES`: Sampler용 GPU 장치 (설정되지 않으면 SLURM에서 자동 감지)

**SLURM 리소스 설정 (`slurm_submit_all.sh`용):**
- `SLURM_PARTITION`: SLURM 파티션 이름 (기본값: `gpu4`)
- `SLURM_GPU_TYPE`: GPU 타입 (기본값: `a6000`)
- `SLURM_GPU_COUNT_TRAINER`: Trainer용 GPU 개수 (기본값: `4`)
- `SLURM_GPU_COUNT_SAMPLER`: Sampler용 GPU 개수 (기본값: `4`)
- **참고**: Orchestrator는 CPU만 사용, GPU 불필요

**예시:**
```bash
# SLURM 리소스 커스터마이징
export SLURM_PARTITION=gpu3
export SLURM_GPU_TYPE=a6000ada
export SLURM_GPU_COUNT_TRAINER=8
# Orchestrator는 CPU만 사용, GPU 불필요

# 애플리케이션 설정
export CONFIG_FILE=my_exp.yaml
export ORCHESTRATOR_URL=http://<orchestrator-host>:59888

# 작업 제출
bash scripts/slurm_submit_all.sh
```

### SLURM 작업 모니터링

```bash
# 작업 상태 확인
squeue -u $USER

# 특정 작업 ID 확인
squeue -j <job_id1>,<job_id2>,<job_id3>

# 작업 세부 정보 보기
scontrol show job <job_id>

# 작업 출력 보기
tail -f slurm_logs/orch.*.out
tail -f slurm_logs/trainer.*.out
tail -f slurm_logs/sampler.*.out
```

### 참고사항

- **Orchestrator가 먼저 시작되어야 함**: Orchestrator 작업은 trainer/sampler 작업보다 먼저 제출되고 실행되어야 합니다
- **네트워크 구성**: 다중 노드 설정의 경우 `ORCHESTRATOR_URL`이 실제 노드 호스트명/IP를 사용하는지 확인하세요
- **GPU 할당**: 스크립트는 `SLURM_STEP_GPUS`를 통해 SLURM이 할당한 GPU를 자동으로 감지합니다
- **메모리 최적화**: Trainer 스크립트는 메모리 파편화를 줄이기 위해 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 포함합니다
- **노드 식별**: 로그 파일에는 쉬운 식별을 위한 노드 정보가 포함됩니다 (예: `trainer_node0.log`)

## Python으로 직접 실행

포그라운드 실행을 위해 Python으로 직접 실행할 수도 있습니다:

### Orchestrator 시작

```bash
export ORCH_HOST=0.0.0.0
export ORCH_PORT=59888
python ralo_cli.py orch --config my_exp.yaml
```

### Trainer 시작

```bash
export ORCH_SERVER=http://<server-ip>:59888
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
SKIP_TRAINER_REGISTRATION=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 --master_port=29501 \
  ralo_cli.py train --config my_exp.yaml
```

### Sampler 시작

```bash
export ORCH_SERVER=http://<server-ip>:59888
export GEN_DEVICES=0,1,2,3
python ralo_cli.py gen --config my_exp.yaml
```

**팁:**
- `ralo_cli.py` 기본 명령은 `train`이므로 `torchrun ... ralo_cli.py --config ...` 형태로도 동작합니다.
- `--orchestrator=http://host:port` CLI 인자는 YAML/환경변수보다 우선합니다. (우선순위: CLI > ENV > CONFIG > 기본값)
- `GEN_DEVICES`를 설정하지 않으면 기본으로 `[0]` GPU를 사용합니다.
- Python으로 직접 실행할 때는 `--log-dir`과 `--run-id`를 사용하여 로그를 정리할 수 있습니다.
- wandb를 끄려면 config에서 `wandb.enabled=false` 또는 `WANDB_DISABLED=true` 환경변수를 사용하세요.

기본 데이터셋은 HuggingFace `qwedsacf/competition_math`(Level 3–5)이며, 다른 커리큘럼을 원하면 config 또는 로딩 함수를 수정하면 됩니다.

## 하이퍼파라미터 설정 & 관계

모든 하이퍼파라미터는 YAML 설정 파일을 통해 제어할 수 있습니다. 이 섹션에서는 주요 파라미터와 그들 간의 유기적 관계를 설명합니다.

### 완전한 YAML 설정 구조

```yaml
model_path: Qwen/Qwen2.5-7B
orchestrator_port: 59888
update_steps: 128  # Orchestrator: optimizer step 전에 필요한 gradient 업로드 수
lr: 1.0e-6  # 학습률 (orchestrator optimizer에서 사용)
epochs: 10

sampler:
  algorithm: treepo
  params:
    rollout_num: 16  # 문제당 rollout 수
    train_batch_size: 1  # 학습 배치 크기 (rollout_num과 나누어떨어져야 함)
    gen_max_tokens: 1024  # 샘플당 최대 생성 토큰 수
    gen_temperature: 0.8  # 샘플링 온도 (0.0-2.0, 높을수록 더 랜덤)
    max_pending_samples: 12800  # 큐에 이만큼 샘플이 있으면 생성 일시 중지
    gen_pending_time: 10.0  # 큐가 가득 찰 때 대기 시간 (초)
    version_poll_interval: 5.0  # 새 가중치 버전 확인 주기 (초, 기본값: 5초)
    max_batch_retry: 3  # 배치 처리 실패 시 최대 재시도 횟수 (기본값: 3)
    treepo_kwargs:
      generation_length: 1024
      depth: 7
      budget_coefficient: 2
      sampling_batch_size: 16

trainer:
  algorithm: treepo
  params:
    rollout_num: 16
    train_batch_size: 1
    accum_steps: 64  # orchestrator로 전송하기 전 로컬 gradient accumulation 수
    lr: 1.0e-6  # 학습률 (orchestrator lr과 일치해야 함)
    grad_offload: true  # Backward 중 OOM 방지를 위한 gradient CPU 오프로딩 활성화 (14K+ 긴 시퀀스 권장)
    gradient_checkpointing_ratio: 1.0  # Gradient checkpointing을 활성화할 레이어 비율 (1.0 = 모든 레이어, 0.5 = 마지막 50% 레이어)
    max_batch_retry: 3  # 배치 처리 실패 시 최대 재시도 횟수 (기본값: 3)
    clip_param: 0.2  # PPO 클리핑 파라미터 (기본값: 0.2)
    pending_retry_timeout: 360.0  # 대기 중인 배치 재시도 타임아웃 (초, 기본값: 360초 = 6분)

orchestrator:
  batch_timeout: 3600.0  # 배치 처리 타임아웃 (초)
  problem_timeout: 600.0  # 문제 처리 타임아웃 (초, 기본값: 10분)
  queue_size: 1600  # 최대 training queue 크기
  status_report_interval: 30.0  # 상태 리포트 출력 주기 (초, 기본값: 30초)
  lock_ttl: 30.0  # Lock Time-To-Live (초, 기본값: 30초)
  server_threads: 10  # 동시 HTTP 요청 처리 스레드 수 (기본값: 10)
  timeout_check_interval: 60.0  # 타임아웃 체크 주기 (초)
  keep_last_versions: 2  # 디스크에 유지할 weight 버전 수
  chunk_size_mb: 50  # Gradient 업로드 청크 크기 (MB, 기본값: 50MB)
  download_chunk_size_mb: 32  # 가중치 다운로드 청크 크기 (MB, 기본값: 32MB)
  # Gradient 청크 관리 (메모리 누수 방지)
  chunk_timeout: 600.0  # 오래된 gradient 청크 타임아웃 (초, 기본값: 10분)
  max_concurrent_uploads: 50  # 최대 동시 gradient 청크 업로드 수 (기본값: 50)
  chunk_cleanup_interval: 60.0  # 청크 정리 간격 (초, 기본값: 60초)
  # 디스크 기반 Gradient 저장 (대용량 모델용)
  gradient_chunks_dir: null  # Gradient 청크 파일 디렉토리 (기본값: 자동 생성, ./orchestrator_gradient_chunks_{port})
  gradient_storage_dir: null  # 복원된 gradient 파일 디렉토리 (기본값: 자동 생성, ./orchestrator_gradients_{port})
  max_gradient_disk_mb: 1024000.0  # 재조립된 gradient 파일 최대 디스크 사용량 (MB, 기본값: 1TB)
  max_chunk_disk_mb: 1024000.0  # Gradient 청크 파일 최대 디스크 사용량 (MB, 기본값: 1TB)
  # HTTP 요청 타임아웃 (초)
  get_batch_timeout: 60.0  # Trainer get_batch 요청 타임아웃 (기본값: 60초)
  send_gradients_timeout: 300.0  # Gradient 업로드 요청 타임아웃 (기본값: 5분)
  download_weights_timeout: 600.0  # 가중치 다운로드 요청 타임아웃 (기본값: 10분)
  upload_samples_timeout: 300.0  # 샘플 업로드 요청 타임아웃 (기본값: 5분)
  fetch_problem_timeout: 10.0  # 문제 가져오기 요청 타임아웃 (기본값: 10초)
  register_timeout: 10.0  # Trainer 등록 타임아웃 (기본값: 10초)
  stats_timeout: 5.0  # Stats 요청 타임아웃 (기본값: 5초)
  heartbeat_timeout: 2.0  # 하트비트 요청 타임아웃 (기본값: 2초)
  version_check_timeout: 5.0  # 버전 확인 요청 타임아웃 (기본값: 5초)
  next_step_timeout: 5.0  # next_step 요청 타임아웃 (기본값: 5초)
  lock_timeout: 5.0  # Lock 획득/해제 요청 타임아웃 (기본값: 5초)
  # 로그 제어 설정 (특정 로그 카테고리를 비활성화하여 출력 감소)
  log_control:
    log_sample_upload: true  # 샘플 업로드 로그 활성화 (기본값: true)
    log_batch_dispatch: true  # 배치 디스패치 로그 활성화 (기본값: true)
    log_gradient_received: true  # Gradient 수신 로그 활성화 (기본값: true)
    log_gradient_reassembled: true  # 재조립된 gradient 로그 활성화 (기본값: true)
    log_gradient_chunks: true  # Gradient 청크 진행 로그 활성화 (기본값: true)
    log_optimizer_step: true  # Optimizer step 완료 로그 활성화 (기본값: true)
    log_processing_gradient: true  # 재조립된 gradient 처리 로그 활성화 (기본값: true)
    log_status_report: true  # 주기적 상태 리포트 로그 활성화 (기본값: true)
    log_http_access: true  # HTTP 액세스 로그 활성화 (WSGI 서버 로그) (기본값: true)

wandb:
  enabled: true  # Wandb 로깅 활성화/비활성화
  project: entropy seesaw  # Wandb 프로젝트 이름
  run_name: TreePO_experiment  # Wandb 실행 이름
  tags: ["demo"]  # Wandb 실행 태그 (선택사항)
  # entity: your-entity  # 선택사항: wandb 엔티티/팀 이름

dataset:
  name: qwedsacf/competition_math  # HuggingFace 데이터셋 이름
  split: train  # 사용할 데이터셋 split
  filter_levels: ["Level 3", "Level 4", "Level 5"]  # 난이도별 문제 필터링
  shuffle_seed: 42  # 데이터셋 셔플링을 위한 랜덤 시드
```

### 파라미터 관계 & 흐름 제어

#### 1. **학습 흐름: Sampler → Orchestrator → Trainer**

**Sampler 파라미터:**
- `max_pending_samples` (기본: 12800): 흐름 제어 임계값. Orchestrator의 큐가 이 크기에 도달하면, sampler는 `gen_pending_time` 초 동안 생성을 일시 중지합니다.
- `gen_pending_time` (기본: 10.0): 큐가 가득 찰 때 대기 시간.
- **관계**: `max_pending_samples`는 `orchestrator.queue_size`보다 크거나 같아야 조기 일시 중지를 방지할 수 있습니다. `max_pending_samples < queue_size`이면 큐에 공간이 있어도 sampler가 일시 중지될 수 있습니다.
- `compute_report_interval`/`compute_report_token_threshold`: Sampler worker가 `/compute/report`로 GPU-seconds와 토큰을 얼마나 자주 업로드할지 제어합니다(기본 60초/32k 토큰).

**Orchestrator 파라미터:**
- `queue_size` (기본: 1600): Training queue의 하드 리밋. 가득 차면 새 샘플이 거부됩니다.
- **관계**: `queue_size`는 병목 지점 역할을 합니다. Sampler가 trainer가 소비하는 것보다 빠르게 생성하면 큐가 가득 차고 sampler가 일시 중지됩니다.

#### 2. **Gradient Accumulation & 업데이트 흐름**

**Trainer 파라미터:**
- `accum_steps` (기본: 다양): Orchestrator로 전송하기 전에 로컬에서 accumulate할 micro-batch 수.
- **효과**: 각 trainer는 `accum_steps`개의 micro-batch를 accumulate한 후, orchestrator로 **1개의 gradient 업로드**를 전송합니다.
- `compute_report_interval`/`compute_report_token_threshold`: Learner(Trainer) 프로세스의 GPU 텔레메트리 전송 주기/토큰 임계값.

**Orchestrator 파라미터:**
- `update_steps` (기본: 128): Optimizer step 전에 필요한 **gradient 업로드** 수 (micro-batch가 아님).
- **관계**: 
  - 4개의 trainer가 있고 각각 `accum_steps=64`인 경우:
    - 각 trainer는 64개의 micro-batch당 1개의 gradient 업로드를 전송
    - Orchestrator는 총 `update_steps=128`개의 gradient 업로드가 필요
    - 이는 128개의 trainer 사이클 = 128 × 64 = 8192개의 micro-batch를 전역적으로 의미
  - **전역 유효 배치 크기**: `update_steps × accum_steps × num_trainers` (모든 trainer가 동일한 `accum_steps`를 가진 경우)

**학습률:**
- Trainer와 orchestrator 설정 모두의 `lr`은 일치해야 합니다 (orchestrator의 optimizer가 이 값을 사용).

#### 3. **배치 크기 관계**

**전역 배치 크기 공식:**
```
global_batch = train_batch_size × world_size × accum_steps × (update_steps / num_trainers)
```

**스케일링 규칙:**
- **GPU 스케일링 시 전역 배치 유지**: 
  - 1 GPU → 2 GPU: `accum_steps`를 절반으로 하거나 `update_steps`를 절반으로
- **GPU 증가와 함께 전역 배치 증가**: 
  - `accum_steps`와 `update_steps`를 유지하거나 비례적으로 증가
- **메모리 제약**: 
  - `train_batch_size` 증가는 OOM을 유발할 수 있음 (per-GPU 메모리 증가)
  - `accum_steps` 증가는 per-GPU 메모리를 증가시키지 않음 (gradient가 in-place로 accumulate됨)
  - 분산 학습은 per-GPU 메모리를 증가시키지 않음

#### 4. **메모리 최적화 파라미터**

RALO는 [LSRL](https://github.com/lsdefine/lsrl)에서 영감을 받은 고급 메모리 최적화 기법을 포함하여 제한된 GPU 메모리에서도 긴 시퀀스(14K+) 학습을 가능하게 합니다.

**Gradient 오프로딩:**
- `grad_offload` (기본: false): Backward pass 후 즉시 gradient를 CPU로 오프로딩합니다.
- **효과**: 
  - `backward()` 직후 gradient가 CPU로 이동하여 gradient accumulation 중 GPU 메모리를 해제합니다.
  - 긴 시퀀스나 큰 `accum_steps` 사용 시 OOM 오류를 방지합니다.
  - Optimizer 상태가 CPU에 저장되어 GPU 메모리 사용량을 줄입니다.
- **사용 시기**: 
  - **시퀀스 ≥ 14K 토큰 권장**
  - **`accum_steps` ≥ 64 권장**
  - **Backward pass 중 OOM 발생 시 필수**
- **성능**: 비동기 CPU 전송으로 인한 오버헤드 최소 (~5-10%).
- **기술적 세부사항**: 
  - CPU 기반 optimizer 상태를 사용하는 CPUAdamW optimizer 사용
  - Gradient는 accumulation 완료 전에 backward 직후 즉시 오프로딩됨
  - Accumulation 중 주기적 activation 정리로 메모리 누수 방지

**Gradient Checkpointing:**
- `gradient_checkpointing_ratio` (기본: 1.0): Gradient checkpointing을 활성화할 transformer 레이어 비율.
- **효과**: 
  - Backward pass 중 activation을 재계산하여 메모리와 계산을 교환합니다.
  - `1.0` = 모든 레이어에서 checkpointing 사용 (최대 메모리 절약, ~30% 느림)
  - `0.5` = 마지막 50% 레이어에서 checkpointing 사용 (균형)
  - `0.0` = checkpointing 없음 (가장 빠름, 최대 메모리 사용)
- **사용 시기**: 
  - GPU 메모리가 제한된 경우 활성화
  - 매우 긴 시퀀스나 큰 모델의 경우 `1.0` 사용
  - 메모리/계산 균형을 위해 `0.5` 사용
- **메모리 절약**: 시퀀스 길이에 따라 activation 메모리를 50-70% 감소시킬 수 있습니다.

**메모리 최적화 흐름:**
1. **Forward pass**: 모델이 activation 계산 (활성화된 경우 checkpointing 사용)
2. **Backward pass**: Gradient 계산, `grad_offload=true`인 경우 즉시 CPU로 오프로딩
3. **Gradient accumulation**: Gradient가 CPU에서 accumulate되고 GPU 메모리 해제
4. **주기적 정리**: Accumulation 단계의 25%마다 activation 정리
5. **Optimizer step**: CPU에서 수행 (CPUAdamW), 가중치를 GPU로 다시 동기화

**시퀀스 길이별 권장 설정:**
- **< 4K 토큰**: `grad_offload: false`, `gradient_checkpointing_ratio: 0.0` (가장 빠름)
- **4K-8K 토큰**: `grad_offload: false`, `gradient_checkpointing_ratio: 0.5` (균형)
- **8K-14K 토큰**: `grad_offload: true`, `gradient_checkpointing_ratio: 0.5` (메모리 효율적)
- **≥ 14K 토큰**: `grad_offload: true`, `gradient_checkpointing_ratio: 1.0` (최대 메모리 절약)

#### 5. **타임아웃 & 안정성 파라미터**

**Orchestrator 타임아웃:**
- `batch_timeout` (기본: 3600초 = 1시간): 배치가 requeue되기 전에 처리될 수 있는 최대 시간.
- `problem_timeout` (기본: 600초 = 10분): Sampler가 문제를 처리할 수 있는 최대 시간 (초과 시 재큐).
- `timeout_check_interval` (기본: 60초): Orchestrator가 타임아웃된 배치/문제를 체크하는 주기.
- **관계**: `timeout_check_interval`은 적시에 실패를 감지하기 위해 `batch_timeout`과 `problem_timeout`보다 작아야 합니다.
- **사용 사례**: 
  - 배치 처리에 더 오래 걸리는 경우 (예: 매우 큰 모델이나 느린 GPU) `batch_timeout`을 증가시키세요.
  - 문제가 복잡하고 더 긴 생성 시간이 필요한 경우 `problem_timeout`을 증가시키세요.

**Weight 버전 관리:**
- `keep_last_versions` (기본: 2): 디스크에 유지할 weight 버전 수.
- **효과**: 오래된 버전은 자동으로 정리되어 디스크 공간을 절약합니다.

**Gradient 업로드 청킹:**
- `chunk_size_mb` (기본: 50MB): Gradient가 이 크기를 초과할 때 청크로 분할하는 크기.
- **효과**: 큰 gradient는 HTTP 타임아웃을 방지하기 위해 자동으로 청크로 분할됩니다.
- **사용 사례**: 빠른 네트워크에서는 증가, 느린 네트워크나 메모리 사용량을 줄이려면 감소.

**Gradient 청크 메모리 관리:**
- `chunk_timeout` (기본: 1200.0초 = 20분): 완료되지 않은 gradient 청크의 타임아웃 시간.
- **효과**: 이 시간을 초과한 미완료 청크 업로드를 자동으로 정리하여 메모리 누수를 방지합니다. 이 시간보다 오래된 청크는 자동으로 제거됩니다.
- **사용 사례**: 
  - 느린 네트워크나 큰 모델로 인해 gradient 업로드가 오래 걸리는 경우 증가.
  - 메모리 압박을 받는 경우 감소하여 메모리를 빠르게 해제.
  - 예상 gradient 업로드 시간보다 길게 설정 (일반적으로 큰 모델의 경우 5-10분).
- `max_concurrent_uploads` (기본: 200): 동시에 허용되는 gradient 청크 업로드의 최대 개수.
- **효과**: 메모리 고갈을 방지하기 위해 동시 청크 업로드 수를 제한합니다. 이 한계에 도달하면 새로운 업로드는 HTTP 503 상태로 거부됩니다.
- **사용 사례**: 
  - 많은 trainer가 있는 대규모 배포 (예: 16+ GPU)에서는 증가.
  - 메모리 압박이나 OOM 오류가 발생하는 경우 감소.
  - 권장: Trainer 수 × 2-4 (재시도 및 동시 배치를 고려).
- `max_chunk_disk_mb` (기본: 1024000.0MB = 1TB): Gradient 청크 파일 최대 디스크 사용량 (MB).
- **효과**: 디스크 사용량이 이 제한을 초과하면 가장 오래된 청크 파일을 자동으로 제거합니다. 이는 완료되지 않은 업로드로 인한 디스크 공간 고갈을 방지합니다.
- **사용 사례**: 
  - 대용량 디스크 공간(예: 2TB+)이 있는 시스템의 경우 증가.
  - 디스크 공간이 제한적인 경우 감소.
  - `(max_concurrent_uploads × chunk_size_mb × 2)`를 기준으로 설정하여 버퍼를 허용.
- `chunk_cleanup_interval` (기본: 60.0초 = 1분): 오래된 gradient 청크를 주기적으로 정리하는 간격 (초).
- **효과**: orchestrator가 오래된 청크를 확인하고 제거하는 빈도를 제어합니다. 더 자주 정리하면 메모리 사용량이 줄어들지만 CPU 오버헤드가 증가합니다.
- **사용 사례**: 
  - 공격적인 메모리 관리나 메모리 압박이 있는 경우 감소 (예: 30초).
  - 안정적인 환경에서 CPU 오버헤드를 줄이려면 증가 (예: 120초).
- **메모리 누수 방지**: 이러한 파라미터는 완료되지 않은 gradient 업로드로 인한 디스크 공간 누수를 방지하기 위해 함께 작동합니다:
  - 청크는 `chunk_timeout` 초 후 자동으로 정리됩니다.
  - 청크의 디스크 사용량은 `max_chunk_disk_mb`(기본값: 1TB)로 제한됩니다.
  - 재조립된 gradient의 디스크 사용량은 `max_gradient_disk_mb`(기본값: 1TB)로 제한됩니다.
  - 정리는 `chunk_cleanup_interval` 초마다 실행됩니다.
  - 상태 보고서에 모니터링을 위한 청크/디스크 사용량이 포함됩니다.

**가중치 다운로드 청킹:**
- `download_chunk_size_mb` (기본: 32MB): 가중치 다운로드 청크 크기.
- **효과**: 가중치는 메모리 사용량을 관리하기 위해 청크로 스트리밍됩니다.
- **사용 사례**: 네트워크 대역폭과 사용 가능한 메모리에 따라 조정.

**디스크 기반 Gradient 저장:**
이 파라미터들은 디스크 기반 gradient 저장 시스템을 제어합니다 (대용량 모델 권장):
- `gradient_chunks_dir` (기본: 자동 생성): 업로드 중 gradient 청크 파일을 저장할 디렉토리.
  - 기본값: `./orchestrator_gradient_chunks_{port}`
  - **효과**: 청크를 RAM 대신 디스크에 저장하여 메모리 고갈을 방지합니다.
  - **사용 사례**: 디스크 위치를 제어하려면 사용자 정의 디렉토리를 지정하세요 (예: 빠른 SSD).
- `gradient_storage_dir` (기본: 자동 생성): 재조립된 gradient 파일을 저장할 디렉토리.
  - 기본값: `./orchestrator_gradients_{port}`
  - **효과**: 재조립된 gradient를 처리 전에 디스크에 저장합니다. Optimizer step 중 파일을 하나씩 로드합니다.
  - **사용 사례**: 더 나은 I/O 성능을 위해 사용자 정의 디렉토리를 지정하세요 (예: 빠른 SSD).
- `max_gradient_disk_mb` (기본: 1024000.0MB = 1TB): 재조립된 gradient 파일 최대 디스크 사용량.
  - **효과**: 디스크 사용량이 이 한계를 초과하면 가장 오래된 gradient 파일을 자동으로 제거합니다. 이는 디스크 공간 고갈을 방지합니다.
  - **사용 사례**: 
    - 대용량 디스크 공간(예: 2TB+)이 있는 시스템의 경우 증가.
    - 디스크 공간이 제한적인 경우 감소.
    - `(update_steps × 평균_gradient_크기_mb × 2)`를 기준으로 설정하여 버퍼를 허용.
    - 7B 모델에서 `update_steps=128`인 경우, 일반적인 gradient 크기는 ~30GB이므로 최소 `128 × 30 × 2 = 7680MB` (7.5GB)로 설정하되, 안전을 위해 1TB 권장.
  - **참고**: Gradient 파일은 optimizer step 중 처리 후 즉시 자동으로 삭제되므로, 이 제한은 주로 엣지 케이스에 대한 보호입니다.
- **이점**: 
  - 대용량 모델(7B+ 파라미터)에서 Orchestrator의 OOM 오류를 방지합니다.
  - 디스크 공간에 의해서만 제한되므로 더 큰 배치 크기(`update_steps`)로 확장할 수 있습니다.
  - 대기 중인 gradient 수와 관계없이 RAM 사용량을 일정하게 유지합니다.
  - 100개 이상의 대기 중인 gradient로도 메모리 문제 없이 학습할 수 있습니다.
  - **자동 파일 정리**: Gradient 파일은 처리 후 즉시 삭제되어 디스크 사용량을 최소화합니다.
- **성능**: 
  - I/O 오버헤드: RAM 기반 저장보다 ~5-10% 느리지만 OOM 충돌을 방지합니다.
  - 디스크 공간 요구사항: `update_steps`와 모델 크기에 따라 충분한 공간을 확보하세요.
  - 권장: 더 나은 성능을 위해 `gradient_chunks_dir`와 `gradient_storage_dir`에 빠른 SSD를 사용하세요.
  - **처리 흐름**: Gradient는 디스크에서 하나씩 로드되어 모델 파라미터에 직접 누적되고, 파일은 즉시 삭제되어 디스크 사용량을 최소화합니다.

**재시도 & 안정성:**
- `max_batch_retry` (기본: 3): 배치 처리 실패 시 최대 재시도 횟수.
- **효과**: 실패한 배치는 이 횟수만큼 재시도된 후 삭제됩니다.
- **사용 사례**: 불안정한 네트워크 환경에서는 증가, 안정적인 환경에서는 감소하여 빠르게 실패.
- `pending_retry_timeout` (기본: 360.0초): 대기 중인 배치 재시도 타임아웃 (초).
- **효과**: 대기 중인 배치를 재시도하기 전 대기 시간을 제어합니다.
- **사용 사례**: 예상 배치 처리 시간과 네트워크 지연에 따라 조정.

**로그 제어:**
- `orchestrator.log_control`: Orchestrator 로그 출력 상세도를 제어하는 설정 섹션.
- **사용 가능한 플래그** (모두 하위 호환성을 위해 기본값은 `true`):
  - `log_sample_upload`: 샘플 업로드 로그 활성화 (예: `[ORCH] Sample uploaded`)
  - `log_batch_dispatch`: 배치 디스패치 로그 활성화 (예: `[ORCH] Batch dispatched to trainer`)
  - `log_gradient_received`: Gradient 수신 로그 활성화 (예: `[ORCH] Gradient received from ...`)
  - `log_gradient_reassembled`: 재조립된 gradient 로그 활성화 (예: `[ORCH] Reassembled gradient`)
  - `log_gradient_chunks`: Gradient 청크 진행 로그 활성화 (예: `[ORCH] Gradient chunks: X/Y`)
  - `log_optimizer_step`: Optimizer step 완료 로그 활성화 (예: `[ORCH] ✓ Optimizer step completed`)
  - `log_processing_gradient`: 재조립된 gradient 처리 로그 활성화 (예: `[ORCH] Processing reassembled gradient`)
  - `log_status_report`: 주기적 상태 보고 로그 활성화 (예: `[ORCH STATUS] Step: ...`)
  - `log_http_access`: HTTP 액세스 로그 활성화 (WSGI 서버 로그) (기본값: true)
- **효과**: 특정 로그 카테고리를 비활성화하면 모니터링이 쉬워지도록 콘솔 출력이 줄어듭니다.
- **사용 사례**: 
  - 특정 플래그를 `false`로 설정하여 학습 중 로그 노이즈 감소
  - 주기적 모니터링을 위해 `log_status_report: true` 유지
  - 모델 업데이트 추적을 위해 `log_optimizer_step: true` 유지
  - 더 깔끔한 출력을 위해 `log_sample_upload` 또는 `log_gradient_chunks` 같은 상세 로그 비활성화
- **설정 예시**:
  ```yaml
  orchestrator:
    log_control:
      log_sample_upload: false  # 빈번한 샘플 업로드 로그 비활성화
      log_batch_dispatch: false  # 배치 디스패치 로그 비활성화
      log_gradient_received: false  # Gradient 수신 로그 비활성화
      log_gradient_chunks: false  # Gradient 청크 진행 로그 비활성화
      log_status_report: true  # 상태 보고 로그 유지
      log_optimizer_step: true  # Optimizer step 로그 유지
  ```
- **참고**: 오류 로그 및 중요한 시스템 메시지는 이러한 설정과 관계없이 항상 활성화됩니다.

**알고리즘 파라미터:**
- `clip_param` (기본: 0.2): 정책 업데이트를 위한 PPO 클리핑 파라미터.
- **효과**: 큰 정책 변경을 방지하기 위해 정책 업데이트의 크기를 제한합니다.
- **사용 사례**: 표준 PPO 하이퍼파라미터; 일반적으로 0.1-0.3 범위.

**Orchestrator 로깅 & 잠금:**
- `status_report_interval` (기본: 30.0초): Orchestrator 상태 리포트 출력 주기 (초).
- **효과**: Orchestrator가 상태 업데이트를 출력하는 빈도를 제어합니다.
- **사용 사례**: 더 자주 업데이트하려면 감소 (더 많은 로그 출력), 로그 양을 줄이려면 증가.
- `lock_ttl` (기본: 30.0초): Lock Time-To-Live (초).
- **효과**: 갱신되지 않으면 이 시간 후에 Lock이 자동으로 만료됩니다.
- **사용 사례**: 예상 작업 시간에 따라 조정; 긴 작업은 더 긴 TTL이 필요합니다.
- `server_threads` (기본: 10): 동시 HTTP 요청 처리 최대 스레드 수.
- **효과**: 스레드 풀을 사용하여 동시 요청 처리 스레드 수를 제한합니다. 이 제한을 초과하는 요청은 스레드가 사용 가능해질 때까지 대기열에 대기합니다.
- **사용 사례**: 
  - 더 많은 동시 클라이언트를 위해 증가 (예: 대규모 배포의 경우 20-50).
  - 리소스 제약(CPU/메모리)이 있는 경우 감소.
  - 권장: Trainer 수 + Sampler 수 + 2-5 여유분.
- **중요**: 이 파라미터는 실제로 스레드 풀 크기를 제한하여 너무 많은 동시 요청으로 인한 리소스 고갈을 방지합니다.

**가중치 버전 폴링:**
- `version_poll_interval` (기본: 5.0초): Orchestrator에서 새 가중치 버전을 확인하는 주기 (초).
- **효과**: Samplers/trainers가 이 주기로 orchestrator를 폴링하여 새 가중치를 확인합니다.
- **사용 사례**: 
  - 더 빠른 가중치 동기화를 위해 감소 (더 많은 네트워크 트래픽).
  - 네트워크 부하를 줄이기 위해 증가 (더 느린 동기화).

#### 6. **생성 파라미터**

**Sampler 생성:**
- `gen_max_tokens` (기본: 1024): 생성된 샘플당 최대 토큰 수.
- `gen_temperature` (기본: 0.8): 샘플링 온도 (0.0 = 결정적, 2.0 = 매우 랜덤).
- `rollout_num` (기본: 16): 문제당 rollout 수.
- **관계**: `train_batch_size`는 `rollout_num`과 나누어떨어져야 합니다.

#### 7. **Wandb 로깅 설정**

**Wandb 파라미터:**
- `enabled` (기본: true): Wandb 로깅 활성화/비활성화.
- `project` (기본: "entropy seesaw"): Wandb 프로젝트 이름 (실행이 기록될 프로젝트).
- `run_name` (기본: "TreePO_experiment"): 이 Wandb 실행의 이름.
- `tags` (기본: []): Wandb에서 실행을 정리하기 위한 선택적 태그 목록.
- `entity` (선택사항): Wandb 엔티티/팀 이름. 지정하지 않으면 기본 Wandb 계정을 사용합니다.
- **효과**: 활성화되면 학습 메트릭, 손실, 시스템 통계가 Wandb에 기록되어 실험을 추적할 수 있습니다.
- **사용 사례**: Wandb 로깅을 비활성화하려면 `enabled: false`로 설정하거나 `WANDB_DISABLED=true` 환경변수를 사용하세요.

#### 8. **데이터셋 설정**

**데이터셋 파라미터:**
- `name` (기본: "qwedsacf/competition_math"): 로드할 HuggingFace 데이터셋 식별자.
- `split` (기본: "train"): 사용할 데이터셋 split (예: "train", "test", "validation").
- `filter_levels` (기본: ["Level 3", "Level 4", "Level 5"]): 난이도별로 문제를 필터링할 레벨 목록.
  - 이 레벨과 일치하는 문제만 Orchestrator의 문제 큐에 로드됩니다.
- `shuffle_seed` (기본: 42): 학습 전 데이터셋 셔플링을 위한 랜덤 시드.
  - 재현 가능한 실험을 위해 동일한 시드를 사용하세요.
- **효과**: Orchestrator는 이러한 설정에 따라 `dataset × epochs` 문제를 큐에 미리 로드합니다.
- **사용 사례**: 
  - 다른 데이터셋을 사용하려면 `name`을 변경하세요.
  - 특정 난이도 범위에 집중하려면 `filter_levels`를 조정하세요.
  - 다른 문제 순서를 원하면 `shuffle_seed`를 변경하세요.

#### 9. **HTTP 요청 타임아웃 설정**

**Orchestrator HTTP 타임아웃:**
모든 HTTP 요청 타임아웃은 `orchestrator` 섹션을 통해 설정할 수 있습니다. 이는 다양한 작업에 대해 시스템이 대기하는 시간을 제어합니다:

- `get_batch_timeout` (기본: 60.0초): Trainer의 배치 가져오기 `GET /get` 요청 타임아웃.
- `send_gradients_timeout` (기본: 300.0초 = 5분): Trainer의 gradient 업로드 `POST /gradient/upload` 요청 타임아웃.
- `download_weights_timeout` (기본: 600.0초 = 10분): 가중치 다운로드 요청 타임아웃 (대용량 파일).
- `upload_samples_timeout` (기본: 300.0초 = 5분): Sampler의 `POST /upload` 요청 타임아웃.
- `fetch_problem_timeout` (기본: 10.0초): Sampler의 `GET /problem/get` 요청 타임아웃.
- `register_timeout` (기본: 10.0초): Trainer 등록 요청 타임아웃.
- `stats_timeout` (기본: 5.0초): `GET /stats` 요청 타임아웃.
- `heartbeat_timeout` (기본: 2.0초): Trainer 하트비트 요청 타임아웃.
- `version_check_timeout` (기본: 5.0초): 가중치 버전 확인 요청 타임아웃.
- `next_step_timeout` (기본: 5.0초): 전역 step 증가 요청 타임아웃.
- `lock_timeout` (기본: 5.0초): Lock 획득/해제 요청 타임아웃.

**사용 사례**: 
- 느린 네트워크나 대형 모델 가중치의 경우 타임아웃을 증가시키세요.
- 더 빠르게 실패하여 문제를 빨리 감지하려면 타임아웃을 감소시키세요.
- `download_weights_timeout`은 모델 크기에 충분히 커야 합니다 (예: 7B 모델 ~15GB).

### 권장 설정

**소규모 (1-2 GPU):**
```yaml
update_steps: 64
trainer:
  params:
    accum_steps: 32
orchestrator:
  queue_size: 800
  batch_timeout: 1800.0
sampler:
  params:
    max_pending_samples: 1600
```

**중규모 (4-8 GPU):**
```yaml
update_steps: 128
trainer:
  params:
    accum_steps: 64
orchestrator:
  queue_size: 1600
  batch_timeout: 3600.0
sampler:
  params:
    max_pending_samples: 12800
```

**대규모 (16+ GPU):**
```yaml
update_steps: 256
trainer:
  params:
    accum_steps: 128
orchestrator:
  queue_size: 3200
  batch_timeout: 7200.0
sampler:
  params:
    max_pending_samples: 25600
```

## 스케일링/배치 규칙
- 전역 배치: `global_batch = per_gpu_batch × world_size × accum_steps × (update_steps / num_trainers)`
- GPU를 1→2로 늘리고 전역 배치를 유지하려면: `accum_steps` 또는 `update_steps`를 절반으로
- GPU를 늘리면서 전역 배치를 키우고 싶다면: `accum_steps`와 `update_steps`를 유지하거나 증가
- 주의: per-GPU 배치를 무리하게 키우면 OOM 발생. 분산은 per-GPU 메모리를 늘려주지 않습니다.

### Orchestrator 모드에서의 스케일링
- 여러 Trainer 노드를 추가하면 gradient 수집 속도가 빨라져 `update_steps`에 더 빨리 도달
- 각 Trainer 노드는 독립적으로 `accum_steps`마다 gradient를 전송하므로, Trainer 노드 수가 많을수록 전역 업데이트 빈도가 증가
- `update_steps`는 Orchestrator의 `ralo_cli.py orch` 실행 시 설정되며, 모든 Trainer 노드의 `accum_steps`와는 별개입니다

## 주요 파일
- `ralo/orchestrator.py`: Orchestrator 서버 (gradient 수집 및 optimizer step)
- `ralo/ralo.py`: RALO 핵심 트레이너/샘플러 로직
- `ralo/orchestrator.py`: Orchestrator(큐, 메트릭, 가중치 I/O)
- `ralo/__init__.py`: 공개 API (예: `from ralo import RALO, ReferenceServer, OrchestratorServer`)
- `ralo_cli.py`: 실행 진입점(서브커맨드: `ref`, `gen`, `orch`, 기본은 `train`)

## 환경변수

### 공통
- `ORCH_SERVER`: Orchestrator 주소(기본 `http://127.0.0.1:59888`)
- `OMP_NUM_THREADS`: CPU 스레드 수(기본 32)
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`: NCCL 비동기 에러 핸들링 권장
- `SKIP_TRAINER_REGISTRATION=1`: Trainer 등록을 건너뜀 (Slurm/폐쇄형 환경에서 핸들셰이크 실패 시 사용)

### Orchestrator 관련
- `ORCH_SERVER`: Orchestrator 서버 주소 (Trainer 노드에서 사용, 예: `http://<orch-ip>:59888`)
- `ORCH_HOST`: Orchestrator 바인딩 호스트 (기본 `0.0.0.0`)
- `ORCH_PORT`: Orchestrator 포트 (기본 `59888`)

### Sampler 관련
- `GEN_DEVICES`: 샘플러가 사용할 GPU 목록(예: `0,1`)

## 가중치 동기화

### Orchestrator 모드
- **Trainer**: 로컬에서 `accum_steps`개의 micro-batch에 대해 gradient를 accumulate한 후, orchestrator로 **1개의 gradient 업로드**를 전송 (`/gradient/upload`)
- **Orchestrator**: `update_steps`개의 **gradient 업로드** (micro-batch가 아님)가 모이면 optimizer step 수행 후 가중치를 저장하고 버전을 증가시킴
- **Trainer**: Orchestrator가 버전을 업데이트하면 자동으로 Orchestrator에서 최신 가중치 다운로드
- **Sampler**: `/weights/version` 조회 후 `/weights/download`로 최신 가중치 pull → vLLM 워커에 반영

### 레거시 모드
- **Trainer**: `step % accum_steps == 0`마다 Orchestrator로 state_dict 업로드
- **Sampler**: `/weights/version` 조회 후 `/weights/download`로 최신 가중치 pull → vLLM 워커에 반영

### Orchestrator 타임아웃 설정
Orchestrator는 trainer 실패를 처리하고 리소스 누수를 방지하기 위한 여러 타임아웃 파라미터를 가지고 있습니다:

**YAML 설정:**
```yaml
orchestrator:
  batch_timeout: 3600.0  # 배치 처리 타임아웃 (초) (기본: 3600 = 1시간)
  queue_size: 1600       # 최대 training queue 크기 (기본: 1600)
  timeout_check_interval: 60.0  # 타임아웃 배치 체크 주기 (초) (기본: 60 = 1분)
  keep_last_versions: 2   # 유지할 weight 버전 수 (기본: 2)
```

**타임아웃 규칙:**
1. **`batch_timeout`**: 배치가 타임아웃으로 간주되기 전에 처리될 수 있는 최대 시간. Trainer가 죽거나 멈추면 이 타임아웃 후 배치가 requeue됩니다. 기본: 3600초 (1시간). 학습 배치 처리에 더 오래 걸리면 이 값을 증가시키세요.
2. **`timeout_check_interval`**: Orchestrator가 타임아웃된 배치를 체크하는 주기. 기본: 60초. 적시에 실패를 감지하려면 `batch_timeout`보다 작아야 합니다.
3. **`queue_size`**: 학습을 위해 큐에 넣을 수 있는 최대 샘플 수. 가득 차면 sampler의 새 샘플이 공간이 생길 때까지 거부됩니다.
4. **`keep_last_versions`**: 디스크에 유지할 weight 버전 수. 오래된 버전은 자동으로 정리되어 디스크 공간을 절약합니다.

**설정 예시:**
매우 큰 배치로 장기 실행 학습의 경우 타임아웃을 증가시킬 수 있습니다:
```yaml
orchestrator:
  batch_timeout: 7200.0  # 매우 큰 배치의 경우 2시간
  timeout_check_interval: 120.0  # 2분마다 체크
  queue_size: 3200  # 더 많은 샘플을 위한 더 큰 큐
```

**타임아웃 동작:**
- 배치가 타임아웃되면 자동으로 requeue되어 다른 trainer가 가져갈 수 있습니다.
- Orchestrator는 다음을 로그로 출력합니다: `[ORCH] Requeued timeout batch batch_XXX (elapsed: XXXs)`
- 많은 배치가 동시에 타임아웃되면 trainer 실패나 네트워크 문제를 나타낼 수 있습니다.

## 로깅/메트릭

### Trainer 출력
Trainer는 다음과 같은 출력을 생성합니다:

**초기화 단계:**
- `[TRAINER] Registered with orchestrator at http://...` - Orchestrator 등록 성공
  - **등록이 필요한 이유:**
    1. **초기 가중치 동기화**: Trainer가 시작할 때 Orchestrator의 최신 가중치 버전을 받아서 로컬 모델을 동기화합니다.
    2. **Orchestrator 연결 확인**: Orchestrator가 실행 중인지 확인하고, 통신 경로가 정상인지 검증합니다.
    3. **설정 정보 교환**: `update_steps` 등 Orchestrator의 설정 정보를 받아서 학습 루프에 활용합니다.
    4. **Worker 식별**: `worker_id`를 등록하여 여러 Trainer 노드를 구분할 수 있게 합니다.
- `[TRAINER] Updated local weights to version X` - 초기 가중치 다운로드 완료

**학습 루프 중 (rank 0만):**
- **tqdm progress bar**: `Gradient Step: X: 100%|████████| 50/50 [01:23<00:00, 1.66s/it]`
  - 각 gradient 전송마다 업데이트됨
- `[TRAINER] Updated local weights to version X` - Orchestrator가 새 버전을 업데이트하면 자동으로 다운로드

**에러/경고:**
- `[TRAINER] Failed to reach orchestrator (attempt X/10), retrying in Xs...` - Orchestrator 연결 실패 시 재시도
- `[TRAINING PROC] failed to fetch global step: ...` - global step 조회 실패
- `[TRAINER] Failed to download weights version X from orchestrator.` - 가중치 다운로드 실패

**종료 시:**
- `[TRAINER] Orchestrator signaled stop (global_step: X)` - Orchestrator 종료 신호 수신
- `[TRAINER] Sent stop signal to sampler workers` - Sampler에 종료 신호 전송

**참고:**
- DDP 환경에서는 각 rank가 독립적으로 출력하므로, rank 0만 progress bar를 표시합니다.
- Gradient는 Orchestrator로 전송되며, Trainer는 optimizer step을 수행하지 않습니다 (Orchestrator가 중앙 집중식으로 처리).

### Sampler 출력
Sampler (vLLM generation workers)는 다음과 같은 출력을 생성합니다:

**초기화 단계:**
- `START vLLM generation...` - Generation workers 시작
- `[SAMPLER] Starting problem fetcher from orchestrator at http://...` - Problem fetcher 스레드 시작
- `[GEN X] Generation worker process uses GPU Y` - Worker 프로세스가 GPU에서 초기화됨
- `[GEN X] CUDA_VISIBLE_DEVICES: ...` - GPU 가시성 설정
- `[GEN X] PID: ...` - 프로세스 ID
- `[VLLM PROC X] Initialized with weights version Y from server` - 초기 가중치 버전 로드됨
- `[VLLM PROC X] Could not fetch initial version from server: ..., starting with version -1` - 초기 버전 조회 실패 (치명적이지 않음)

**생성 중:**
- `[SAMPLER] Fetched X problems (queue size: Y)` - Problem 가져오기 진행 상황 (100개마다)
- `[VLLM PROC X] newer weights available: Y > Z, updating workers directly from server...` - 새 모델 버전 감지, 업데이트 중
- `[VLLM PROC X] model updated to version Y. Results: ...` - 모델 가중치 업데이트 성공
- `[VLLM PROC X] weight apply failed (attempt X/Y). Results: ...` - 가중치 업데이트 실패, 재시도 중
- `[VLLM PROC X] ERROR applying model (attempt X/Y): ...` - 가중치 업데이트 중 에러
- `[VLLM PROC X] giving up updating to Y for now; will retry later. Last results: ...` - 재시도 후 가중치 업데이트 실패
- `[VLLM PROC X] server reported 404 for weights vY; retry later` - 서버에서 가중치 버전을 찾을 수 없음

**완료 시:**
- `[GEN X] Generation worker finished, sending end signal to orchestrator ...` - Worker 완료, 종료 신호 전송 중
- `[GEN X] Failed to send end signal: ...` - 종료 신호 전송 실패 (치명적이지 않음)
- `[SAMPLER] Received end signal from orchestrator after X problems` - 모든 문제 처리 완료, 종료 중

**에러/경고:**
- `[SAMPLER] Error requesting problem from orchestrator: ...` - Problem 가져오기 에러, 재시도 중
- `[SAMPLER] Unexpected response: ...` - Orchestrator로부터 예상치 못한 응답
- `[GEN X] pending samples too many, wait for training process ...` - 대기 중인 샘플이 너무 많음, 대기 중
- `[GEN X] Error in generation worker: ...` - Generation worker 에러
- `[GEN X] Dropping batch after X retries.` - 재시도 후 배치 삭제
- `[GEN X] Failed to requeue batch: ...` - 배치 재큐 실패
- `[GEN MONITOR] Dropping batch X after Y retries (timeout).` - 배치 타임아웃, 모니터에 의해 삭제됨
- `[GEN MONITOR] Requeued timed-out batch X (retry Y).` - 타임아웃된 배치 재큐됨
- `[GEN MONITOR] Failed to requeue batch X: ...` - 타임아웃된 배치 재큐 실패

**참고:**
- 여러 generation workers가 병렬로 실행됩니다 (gen_device에 지정된 GPU당 하나씩).
- 각 worker는 독립적으로 문제를 가져오고, 샘플을 생성하며, Orchestrator에 업로드합니다.
- Workers는 Orchestrator에서 새 버전이 사용 가능할 때 자동으로 모델 가중치를 업데이트합니다.
- Problem fetcher는 작은 버퍼(기본값: 20개 문제)를 유지하여 전체 데이터셋을 미리 로드하지 않습니다.

### Orchestrator 출력
Orchestrator는 학습 진행 상황을 다음과 같이 표시합니다:

**초기화:**
- `[ORCH] Problem queue initialized with X items` - 문제 큐 초기화 완료
- `[ORCH] Saved initial weights as version 0` 또는 `[ORCH] Loaded weights version X from disk` - 가중치 초기화/로드
- `[ORCH] Orchestrator server started on http://...` - 서버 시작
- `[ORCH] Status reports every 30 seconds. Use /stats endpoint for detailed info.` - 주기적 상태 리포트 안내

**학습 중 (주기적 리포트, 30초마다):**
- `[ORCH STATUS] Step: X | Gradients: Y (pending: Z/W, P%) | Queue: Q samples | Processed: A/B batches | Problems: C/D | Version: V`
  - **Step**: 현재 global step
  - **Gradients**: 총 수집된 gradient 수, 대기 중인 배치 수, 진행률
  - **Queue**: 샘플 큐 크기
  - **Processed**: 처리된/전체 배치 수
  - **Problems**: 분배된/전체 문제 수
  - **Version**: 현재 모델 버전

**Gradient 업로드 시:**
- `[ORCH] Gradient received from worker_id (step X) | Pending: Y/Z (P%) | Total: N gradients | Queue: Q samples` - Gradient 수신
- `[ORCH] ✓ Optimizer step completed -> version X | Total gradients: Y | Global step: Z` - Optimizer step 완료

**샘플/배치 처리:**
- `[ORCH] Sample uploaded | Queue: X samples | Total enqueued: Y` - 샘플 업로드 (10개마다 출력)
  - **Queue**: 현재 학습 큐에 있는 샘플 수 (`orchestrator.queue_size`로 제어)
  - **Total enqueued**: 시작 이후 업로드된 총 샘플 수 (누적)
  - **관계**: `Queue` ≤ `queue_size` (큐가 가득 차면 sampler가 `max_pending_samples`로 일시 중지)
- `[ORCH] Batch dispatched to trainer | Queue: X | Processing: Y | Total dequeued: Z` - 배치 전송 (5개마다 출력)
  - **Queue**: 전달 후 남은 큐의 샘플 수
  - **Processing**: 현재 trainer가 처리 중인 배치 수
  - **Total dequeued**: trainer에 전달된 총 배치 수 (누적)

**Gradient 업로드 (청크 단위):**
- `[ORCH] Gradient chunks: X/Y (Z%)` - 청크 단위 gradient 업로드 진행률
  - Gradient가 청크로 분할되어 업로드될 때 진행률 표시 (대형 모델)
  - `chunk_size_mb` 파라미터로 제어 (기본값: 50MB/청크)
- `[ORCH] /gradient/upload_finalize called!` - 마지막 청크 수신, 재조립 시작
- `[ORCH] Reassembled gradient: X chunks → YMB` - 청크에서 gradient 재조립 완료
- `[ORCH] Processing reassembled gradient from worker_id (step X)` - 특정 trainer의 gradient 처리 중
- `[ORCH] Gradient received from worker_id (step X) | Pending: Y/Z (P%) | Total: N gradients | Queue: Q samples`
  - **Pending**: 현재 대기 중인 gradient 업로드 수 / optimizer step에 필요한 `update_steps`
  - **Total**: 시작 이후 수신된 총 gradient 업로드 수
  - **Queue**: 현재 샘플 큐 크기

**종료 시:**
- `[ORCH] Received end signal from sampler` - Sampler 종료 신호 수신
- `[ORCH] Queue empty with X pending batches; finalizing optimizer step` - 최종 optimizer step 수행

**상세 정보:**
- `/stats` 엔드포인트로 실시간 상태 조회 가능:
  - `global_step`, `pending_batches`, `total_gradients`, `queue_size`, `current_version` 등

## HTTP 요청 타임아웃 설정

모든 HTTP 요청 타임아웃은 YAML의 `orchestrator` 섹션에서 설정할 수 있습니다. 이 타임아웃은 클라이언트가 오류를 발생시키기 전에 Orchestrator 응답을 기다리는 시간을 제어합니다.

### 타임아웃 파라미터

| 파라미터 | 기본값 | 설명 | 증가가 필요한 경우 |
|----------|--------|------|-------------------|
| `get_batch_timeout` | 60.0초 | Trainer 배치 가져오기 타임아웃 | Orchestrator가 optimizer step으로 바쁠 때 |
| `send_gradients_timeout` | 300.0초 (5분) | Gradient 업로드 타임아웃 | 매우 큰 모델이나 느린 네트워크 |
| `download_weights_timeout` | 600.0초 (10분) | 가중치 다운로드 타임아웃 | 큰 모델이나 느린 네트워크 |
| `upload_samples_timeout` | 300.0초 (5분) | 샘플 업로드 타임아웃 | 큰 배치나 느린 네트워크 |
| `fetch_problem_timeout` | 10.0초 | 문제 가져오기 타임아웃 | Orchestrator가 과부하일 때 |
| `register_timeout` | 10.0초 | Trainer 등록 타임아웃 | Orchestrator 시작이 느릴 때 |
| `stats_timeout` | 5.0초 | Stats 엔드포인트 타임아웃 | 일반적으로 충분함 |
| `heartbeat_timeout` | 2.0초 | Heartbeat 타임아웃 | 일반적으로 충분함 |
| `version_check_timeout` | 5.0초 | 버전 확인 타임아웃 | 일반적으로 충분함 |
| `next_step_timeout` | 5.0초 | Next step 요청 타임아웃 | 일반적으로 충분함 |
| `lock_timeout` | 5.0초 | Lock 획득/해제 타임아웃 | 일반적으로 충분함 |

### 설정 예시

```yaml
orchestrator:
  # ... 기타 설정 ...
  get_batch_timeout: 120.0  # Optimizer step 중 타임아웃 오류 발생 시 증가
  send_gradients_timeout: 600.0  # 매우 큰 모델의 경우 증가
  download_weights_timeout: 1200.0  # 느린 네트워크에서 큰 모델의 경우 증가
```

### 타임아웃 오류 처리

- **Trainer `get_batch()` 타임아웃**: Trainer가 지수 백오프로 재시도합니다. Orchestrator가 CPU optimizer step으로 바쁠 때 `get_batch_timeout`을 증가시키세요.
- **Gradient 업로드 타임아웃**: Trainer가 재시도합니다. 큰 모델이나 느린 네트워크의 경우 `send_gradients_timeout`을 증가시키세요.
- **가중치 다운로드 타임아웃**: Trainer가 재시도합니다. 큰 모델의 경우 `download_weights_timeout`을 증가시키세요.

## 트러블슈팅

### 일반적인 문제
- **OOM**: per-GPU 배치 축소, `accum_steps` 증가, 시퀀스 길이 축소, bf16/FP16 유지, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **통신 문제**: 포트/방화벽 개방, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 설정
- **평균 Loss 출력 필요**: 전역 all-reduce 후 rank0 출력 로직을 추가하세요.
- **타임아웃 오류**: `ReadTimeoutError` 또는 `HTTPConnectionPool timeout` 발생 시:
  - YAML config의 `orchestrator` 섹션에서 관련 타임아웃 값을 증가시키세요
  - Orchestrator 로그에서 CPU optimizer step 지속 시간 확인 (HTTP 요청을 블로킹할 수 있음)
  - Orchestrator가 자주 바쁠 때 `get_batch_timeout` 증가를 고려하세요

### Orchestrator 관련
- **Gradient 업로드 실패**: `ORCH_SERVER` 환경변수가 올바르게 설정되었는지 확인, Orchestrator가 실행 중인지 확인 (`curl http://<orch-ip>:59888/stats`)
- **가중치 동기화 지연**: Orchestrator의 `pending_batches`가 `update_steps`에 도달하지 않으면 업데이트가 발생하지 않습니다. 여러 Trainer 노드가 정상적으로 gradient를 전송하는지 확인
- **버전 불일치**: Trainer가 최신 가중치를 받지 못하는 경우, Orchestrator의 `/weights/version`과 Orchestrator의 `/weights/version`을 비교하여 확인

### Slurm/폐쇄형 환경
- **Trainer 등록 실패**: Slurm이나 폐쇄형 서버 환경에서 Orchestrator와의 핸들셰이크가 실패할 수 있습니다. 이 경우:
  - 등록 실패 시 경고만 출력하고 학습은 계속 진행됩니다 (기본 동작)
  - 등록을 완전히 건너뛰려면 `SKIP_TRAINER_REGISTRATION=1` 환경변수를 설정하세요
  - 등록이 실패해도 gradient 전송과 가중치 다운로드는 정상적으로 작동합니다

## 다국어 지원 (한국어/영어)

RALO는 한국어와 영어 텍스트를 지원하는 다국어 학습을 지원합니다. 토크나이저는 모델의 학습 데이터에 따라 두 언어를 자동으로 처리합니다.

### 한국어/영어 모델 사용하기

1. **모델 선택**: 다국어 모델(한국어와 영어 데이터로 학습된 모델) 또는 한국어 전용 모델을 사용하세요:
   ```python
   model_path = "your-korean-english-model-path"
   ```

2. **프롬프트 템플릿**: `ralo_cli.py`의 기본 프롬프트 템플릿은 `apply_chat_template()`을 사용하여 다국어 입력을 자동으로 처리합니다:
   ```python
   def make_prompt_fn(self, item):
       return self.tokenizer.apply_chat_template(
           [
               {"role": "system", "content": system_prompt},
               {"role": "user", "content": item["Q"]},  # 한국어 또는 영어 가능
           ],
           tokenize=False,
           add_generation_prompt=True,
       )
   ```

3. **커스텀 프롬프트 함수**: 한국어/영어를 위한 프롬프트 포맷팅을 커스터마이즈하려면:
   ```python
   ralo = RALO(model_path="...", ...)
   
   def custom_prompt_fn(self, item):
       # 한국어/영어 프롬프트 처리
       question = item.get("Q", "")
       # 여기에 커스텀 포맷팅 로직 작성
       return formatted_prompt
   
   ralo.set_rollout_prompt_fn(custom_prompt_fn)
   ```

4. **토크나이저 설정**: 토크나이저는 모델 경로에서 자동으로 로드됩니다. 한국어 전용 토크나이징을 위해서는:
   - 모델이 한국어 토크나이저(예: SentencePiece, 한국어 어휘가 포함된 BPE)로 학습되었는지 확인
   - 모델이 한국어를 지원하면 토크나이저가 자동으로 한국어 문자를 처리합니다

5. **데이터셋 형식**: 데이터셋의 `Q` (질문) 및 `A` (답변) 필드에 한국어 또는 영어 텍스트가 포함되어야 합니다:
   ```python
   {
       "Q": "한국어 질문 또는 English question",
       "A": "한국어 답변 또는 English answer"
   }
   ```

### 참고사항
- 모델의 토크나이저가 언어 지원을 결정합니다. 지원되는 언어는 모델 문서를 확인하세요.
- vLLM은 모델의 토크나이저를 자동으로 사용하므로 생성 시 추가 설정이 필요하지 않습니다.
- 최상의 결과를 위해서는 한국어-영어 데이터로 특별히 학습된 모델이나 대상 언어로 파인튜닝된 모델을 사용하세요.

## 평가 서브시스템

RALO는 각 가중치 버전 업데이트마다 자동으로 벤치마크를 실행하고 결과를 wandb에 기록할 수 있습니다. Orchestrator가 버전별 평가 작업을 스케줄링하고 Sampler가 백그라운드에서 이를 수행합니다.

흐름(버전당):
- 벤치마크 작업 등록(AIME 2024/2025, GPQA, HLE/커스텀 등)
- Sampler가 작업을 선점 → 최신 가중치 적용 → vLLM으로 후보 생성
- 플러그인 메트릭으로 점수 계산
- 결과를 Orchestrator에 보고하고 `eval/<benchmark>/<metric>` 경로로 wandb에 기록

### 자동 평가 활성화 (설정)
YAML 예시:

```yaml
evaluation:
  enabled: true
  schedule: on_version_change
  max_parallel_jobs: 1
  devices: [0]
  wandb_namespace: "eval"
  benchmarks:
    - name: aime_2024
      loader: builtin:aime_2024
      split: test
      max_items: null
      num_candidates: 1
      prompt_template: builtin:aime_cot
      answer_extractor: builtin:boxed
      metrics: [builtin:accuracy@1, builtin:accuracy@5]
    - name: aime_2025
      loader: builtin:aime_2025
      split: test
      metrics: [builtin:accuracy@1]
    - name: gpqa_diamond
      loader: hf:Idavidrein/gpqa
      config: diamond
      split: validation
      metrics: [builtin:mc_accuracy@1]
    - name: hle
      loader: hf:<your_org/your_hle_dataset>
      split: test
      metrics: [your_pkg.metrics:custom_score]
```

### 기본 제공 벤치마크
- AIME 2024/2025 (`HuggingFaceH4/aime_20xx`) – boxed 정답 추출
- GPQA (`Idavidrein/gpqa`, 예: `config: diamond`) – 객관식 추출
- HLE – 범용 HF 어댑터(데이터셋 id/필드명 구성)

### 메트릭 & 플러그인
내장 메트릭:
- `builtin:accuracy@K` (예: `builtin:accuracy@1`, `builtin:accuracy@5`)
-, `builtin:mc_accuracy@1` (객관식)
-, `builtin:exact_match`

사용자 정의 메트릭 예:
```python
def custom_score(preds, refs, **kw):
    correct = sum(int(str(p).strip() == str(r).strip()) for p, r in zip(preds, refs))
    return {"my/custom_accuracy": correct / max(1, len(refs))}
```

### 실행 & 결과
- 활성화 시 새 버전마다 자동 실행됩니다.
- Orchestrator REST: `/eval/job/get`, `/eval/job/claim`, `/eval/job/report`, `/eval/results?version=X`, `/eval/stats`, `/compute/stats`.
- wandb에는 `eval/<benchmark>/<metric>`로 기록(`wandb_namespace` 접두).

### 리소스 제어
- `max_parallel_jobs`로 동시 평가 작업 수 제한.
- `devices`로 평가 GPU 고정(비어 있으면 Sampler GPU 공유).

### 스케일링 법칙 적합(ScaleRL)
- Sampler(Actor)·Trainer(Learner)·Evaluator가 `/compute/report`로 보낸 GPU-seconds를 합산한 GPU-hour를 `Compute`로 사용하여 각 벤치마크/메트릭 쌍에 대해 ScaleRL 식 [[arXiv:2510.13786](https://arxiv.org/html/2510.13786v1)]을 적합합니다:

  \[
  R(C) = R_0 + \frac{A - R_0}{1 + \left(\frac{C_{\text{mid}}}{C}\right)^B}
  \]

  여기서 \(R_0\)는 기본 성능, \(A\)는 최종 한계 성능, \(C_{\text{mid}}\)는 절반 성능을 얻는 데 필요한 연산량, \(B\)는 곡선의 기울기를 나타냅니다.
- 적합된 파라미터는 wandb에 `eval/<benchmark>/<metric>/fit_r0`, `fit_a`, `fit_c_mid`, `fit_b`, `fit_loss` 형태로 기록되며, `/eval/fit` 엔드포인트와 GPU-hour 요약이 포함된 `eval_fit_summary.json`에서 조회할 수 있습니다.
- 학습 종료 시 마지막 평가/피팅이 완료될 때까지 기다리려면 `evaluation.shutdown_timeout_sec` 값을 조정하세요.

### GPU-시간 텔레메트리
- Sampler/Trainer/Evaluator는 `compute_report_interval`·`compute_report_token_threshold` 설정에 따라 `/compute/report`로 GPU-seconds와 토큰 수를 주기적으로 업로드합니다(기본 60초/32k 토큰).
- `/compute/stats` 혹은 `eval_fit_summary.json`에서 누적 GPU-hour/토큰을 확인할 수 있으며, wandb에는 `eval/<benchmark>/compute_gpu_hours`, `compute_actor_hours` 등으로 노출됩니다.

## 라이선스
프로젝트 루트의 라이선스 정책에 따릅니다(명시되지 않았다면 내부 용도로 가정).


