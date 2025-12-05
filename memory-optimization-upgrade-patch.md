# RALO 메모리 최적화 업그레이드 패치안

## 목표

분석 문서의 ZeRO-2/3와 NeMo Megatron 기법을 통합하여 단일/분산 환경 모두에서 메모리 효율성 향상

## 주요 업그레이드 항목

### 1. ZeRO-2 스타일 Gradient Sharding (분산 학습)

**파일**: `ralo/cpuadamw.py`, `ralo/ralo.py`

- **현재**: 분산 학습 시 all-reduce 후 rank 0만 CPU로 오프로딩
- **개선**: 각 GPU가 자신의 gradient shard만 CPU로 오프로딩 (ZeRO-2 방식)
- **효과**: 분산 학습 시 gradient 메모리를 GPU 수에 비례하여 절약

**구현**:

- `DistributedCPUAdamW`에 gradient sharding 로직 추가
- 각 rank가 자신의 파라미터 부분에 대한 gradient만 유지
- Reduce-scatter 패턴으로 gradient 분산

### 2. Selective Activation Recomputation 개선

**파일**: `ralo/ralo.py`, `ralo/utils.py`

- **현재**: 전체 레이어에 gradient checkpointing 적용
- **개선**: 선택적 activation 재계산 (중요한 activation만 저장)
- **효과**: 메모리와 계산 비용의 최적 균형

**구현**:

- `enable_gradient_checkpointing` 함수 개선
- 레이어별 checkpointing 전략 선택 (attention output만 저장, MLP intermediate는 재계산)
- 설정 파일에 `selective_checkpointing` 옵션 추가

### 3. Sequence Parallelism 기본 지원 (긴 시퀀스)

**파일**: `ralo/ralo.py`, `ralo/services/training_service.py`

- **새 기능**: 시퀀스 길이를 GPU 간 분산 처리
- **효과**: 긴 시퀀스(14K+)에서 activation 메모리 절약

**구현**:

- `CPUOffloadTrainer`에 sequence parallelism 옵션 추가
- 분산 학습 시 시퀀스를 chunk로 분할하여 각 GPU에 할당
- Attention 계산 시 all-gather로 전체 attention matrix 수집
- 설정 파일에 `sequence_parallelism` 옵션 추가

### 4. 하이브리드 메모리 최적화 전략

**파일**: `ralo/ralo.py`, `ralo/cpuadamw.py`

- **새 기능**: 환경에 따라 자동으로 최적 전략 선택
- **전략**:
  - 단일 GPU: CPU offloading (현재 방식)
  - 분산 + 짧은 시퀀스: ZeRO-2 스타일 gradient sharding
  - 분산 + 긴 시퀀스: Sequence Parallelism + CPU Offloading 하이브리드

**구현**:

- `_select_memory_strategy()` 메서드 추가
- GPU 수, 시퀀스 길이, 메모리 사용량에 따라 자동 선택
- 설정 파일에서 수동 오버라이드 가능

### 5. 설정 파일 확장

**파일**: `ralo/config.py`, `configs/example.yaml`, `my_exp.yaml`

**새로운 설정 옵션**:

```yaml
trainer:
  params:
    # 기존 옵션들...
    grad_offload: true
    gradient_checkpointing_ratio: 1.0
    
    # 새로운 옵션들
    zero2_gradient_sharding: true  # 분산 학습 시 ZeRO-2 스타일 gradient sharding
    selective_checkpointing: true  # 선택적 activation 재계산
    sequence_parallelism: false  # 시퀀스 병렬화 (긴 시퀀스용)
    auto_memory_strategy: true  # 자동 메모리 전략 선택
```

### 6. 문서 업데이트

**파일**: `README.md`, `README_KO.md`

- 새로운 메모리 최적화 옵션 설명 추가
- ZeRO-2, Sequence Parallelism 사용 가이드 추가
- 하이브리드 전략 선택 가이드 추가

## 구현 세부사항

### 파일별 변경사항

1. **ralo/cpuadamw.py**

   - `DistributedCPUAdamW`에 gradient sharding 로직 추가
   - `_shard_gradients()` 메서드 추가
   - 각 rank가 자신의 gradient shard만 처리하도록 수정

2. **ralo/ralo.py**

   - `CPUOffloadTrainer.__init__()`에 sequence parallelism 옵션 추가
   - `_select_memory_strategy()` 메서드 추가
   - `_offload_gradients_to_cpu()` 개선 (ZeRO-2 지원)
   - Sequence parallelism을 위한 forward/backward 래퍼 추가

3. **ralo/utils.py**

   - `enable_gradient_checkpointing()` 개선
   - `enable_selective_checkpointing()` 함수 추가
   - Attention output만 저장하는 로직 추가

4. **ralo/config.py**

   - `TrainerConfig`에 새로운 옵션 필드 추가
   - 기본값 설정

5. **configs/example.yaml, my_exp.yaml**

   - 새로운 메모리 최적화 옵션 추가 (주석 포함)

6. **README.md, README_KO.md**

   - 새로운 옵션 설명 추가
   - 사용 예시 추가

## 호환성

- 기존 설정 파일과 하위 호환 (새 옵션은 기본값 사용)
- 단일 GPU 환경: 기존 동작 유지
- 분산 학습 환경: 새로운 최적화 기법 자동 적용

## 테스트 시나리오

1. 단일 GPU + 긴 시퀀스: CPU offloading만 사용
2. 분산 학습 + 짧은 시퀀스: ZeRO-2 gradient sharding
3. 분산 학습 + 긴 시퀀스: Sequence Parallelism + CPU Offloading