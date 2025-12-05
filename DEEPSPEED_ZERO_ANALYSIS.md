# DeepSpeed ZeRO-2 & ZeRO-3 메모리 최적화 로직 분석

## 개요

DeepSpeed의 ZeRO (Zero Redundancy Optimizer)는 대규모 모델 학습 시 GPU 메모리 사용을 최소화하기 위한 분산 최적화 기법입니다. 각 단계별로 메모리 절약 수준과 통신 오버헤드가 다릅니다.

## ZeRO-1: Optimizer State Partitioning

### 핵심 아이디어
- **Optimizer State를 GPU 간 분산 저장**
- AdamW의 경우: momentum (exp_avg), variance (exp_avg_sq)를 각 GPU에 분산

### 메모리 절약
- **절약량**: Optimizer state 메모리의 `1/N` (N = GPU 수)
- AdamW의 경우: 파라미터당 8 bytes (momentum 4 bytes + variance 4 bytes)
- 예: 7B 모델, 4 GPU → Optimizer state: 56GB → 14GB per GPU

### 통신 패턴
- Optimizer step 시: **All-Gather**로 모든 GPU에서 optimizer state 수집
- 각 GPU는 자신이 담당하는 파라미터 부분의 optimizer state만 유지

### 구현 로직
```python
# 의사코드
def zero1_step():
    # 1. 각 GPU는 자신의 파라미터 부분에 대한 optimizer state만 유지
    # 2. Optimizer step 수행 전에 All-Gather로 전체 파라미터 수집
    all_gather(parameters)  # 통신: O(P) where P = parameter size
    # 3. Optimizer step 수행
    optimizer.step()
    # 4. 각 GPU는 자신의 부분만 유지, 나머지 해제
```

---

## ZeRO-2: Optimizer State + Gradient Partitioning

### 핵심 아이디어
- **ZeRO-1 + Gradient Sharding**
- Backward pass 후 계산된 gradient를 GPU 간 분산 저장

### 메모리 절약
- **절약량**: 
  - Optimizer state: `1/N` (ZeRO-1과 동일)
  - Gradient: `1/N` (추가 절약)
- Gradient 메모리: 파라미터당 4 bytes (FP32) 또는 2 bytes (FP16/BF16)
- 예: 7B 모델, 4 GPU → Gradient: 28GB → 7GB per GPU

### 통신 패턴
1. **Backward Pass 중**:
   - 각 GPU는 자신이 담당하는 파라미터 부분의 gradient만 계산/저장
   - Gradient는 자동으로 분산됨

2. **Optimizer Step 전**:
   - **Reduce-Scatter**: Gradient를 평균화하고 분산 저장
   - 각 GPU는 자신의 부분만 유지

3. **Optimizer Step**:
   - **All-Gather**: Optimizer state 수집 (ZeRO-1과 동일)
   - Optimizer step 수행

### 구현 로직
```python
# 의사코드
def zero2_backward():
    # 1. Forward pass (모든 GPU에서 전체 모델 사용)
    output = model(input)
    loss = criterion(output, target)
    
    # 2. Backward pass
    loss.backward()
    # 각 GPU는 자신이 담당하는 파라미터 부분의 gradient만 유지
    # 나머지 gradient는 즉시 해제
    
def zero2_step():
    # 1. Reduce-Scatter: Gradient 평균화 및 분산
    reduce_scatter(gradients)  # 통신: O(P)
    # 각 GPU는 자신의 부분만 유지
    
    # 2. All-Gather: Optimizer step을 위해 파라미터 수집
    all_gather(parameters)  # 통신: O(P)
    
    # 3. Optimizer step
    optimizer.step()
    
    # 4. 각 GPU는 자신의 부분만 유지
```

### 통신 비용
- **Backward**: Gradient reduce-scatter → O(P)
- **Forward**: 없음 (전체 모델 사용)
- **Step**: Parameter all-gather → O(P)
- **총 통신**: 2 × O(P) per step

---

## ZeRO-3: Optimizer State + Gradient + Parameter Partitioning

### 핵심 아이디어
- **ZeRO-2 + Parameter Sharding**
- 모델 파라미터 자체를 GPU 간 분산 저장

### 메모리 절약
- **절약량**: 
  - Optimizer state: `1/N`
  - Gradient: `1/N`
  - Parameters: `1/N` (추가 절약)
- Parameter 메모리: 파라미터당 2 bytes (BF16) 또는 4 bytes (FP32)
- 예: 7B 모델, 4 GPU → Parameters: 14GB → 3.5GB per GPU

### 통신 패턴
1. **Forward Pass**:
   - **All-Gather**: 필요한 파라미터를 모든 GPU에서 수집
   - Forward 계산 수행
   - **Release**: 사용한 파라미터 해제 (다른 GPU로 반환)

2. **Backward Pass**:
   - **All-Gather**: 필요한 파라미터 수집
   - Backward 계산 수행
   - **Reduce-Scatter**: Gradient 평균화 및 분산
   - 파라미터 해제

3. **Optimizer Step**:
   - **All-Gather**: Optimizer step을 위해 파라미터 수집
   - Optimizer step 수행
   - 파라미터 해제

### 구현 로직
```python
# 의사코드
def zero3_forward():
    # 1. All-Gather: Forward에 필요한 파라미터 수집
    all_gather(parameters)  # 통신: O(P)
    
    # 2. Forward pass
    output = model(input)
    
    # 3. 파라미터 해제 (메모리 절약)
    release_parameters()
    
    return output

def zero3_backward():
    # 1. All-Gather: Backward에 필요한 파라미터 수집
    all_gather(parameters)  # 통신: O(P)
    
    # 2. Backward pass
    loss.backward()
    
    # 3. Reduce-Scatter: Gradient 평균화 및 분산
    reduce_scatter(gradients)  # 통신: O(P)
    
    # 4. 파라미터 해제
    release_parameters()

def zero3_step():
    # 1. All-Gather: Optimizer step을 위해 파라미터 수집
    all_gather(parameters)  # 통신: O(P)
    
    # 2. Optimizer step
    optimizer.step()
    
    # 3. 파라미터 해제
    release_parameters()
```

### 통신 비용
- **Forward**: Parameter all-gather → O(P)
- **Backward**: Parameter all-gather + Gradient reduce-scatter → 2 × O(P)
- **Step**: Parameter all-gather → O(P)
- **총 통신**: 4 × O(P) per step (ZeRO-2의 2배)

### 최적화 기법
1. **Gradient Bucketing**: 여러 파라미터의 gradient를 묶어서 한 번에 통신
2. **Overlap Communication**: 통신과 계산을 오버랩하여 지연 시간 숨김
3. **CPU Offloading**: ZeRO-Infinity에서 파라미터를 CPU/NVMe로 오프로드

---

## 메모리 사용량 비교 (7B 모델, 4 GPU 예시)

| 구성 요소 | 기본 DDP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---------|---------|--------|--------|--------|
| Parameters (BF16) | 14GB × 4 | 14GB × 4 | 14GB × 4 | 3.5GB × 4 |
| Gradients (FP32) | 28GB × 4 | 28GB × 4 | 7GB × 4 | 7GB × 4 |
| Optimizer States | 56GB × 4 | 14GB × 4 | 14GB × 4 | 14GB × 4 |
| **총 메모리** | **392GB** | **224GB** | **140GB** | **98GB** |
| **Per GPU** | **98GB** | **56GB** | **35GB** | **24.5GB** |

---

## 현재 RALO 구현과의 비교

### RALO의 접근 방식 (CPUAdamW + Gradient Offloading)

#### 유사점
1. **Optimizer State를 CPU에 저장** (ZeRO-1과 유사하지만 CPU 사용)
   - ZeRO-1: GPU 간 분산
   - RALO: CPU에 완전 오프로드

2. **Gradient를 즉시 CPU로 이동** (ZeRO-2와 유사하지만 CPU 사용)
   - ZeRO-2: GPU 간 분산
   - RALO: CPU로 완전 오프로드

#### 차이점

| 특징 | ZeRO-2/3 | RALO (현재) |
|------|----------|-------------|
| **분산 방식** | GPU 간 분산 (Sharding) | CPU 오프로딩 |
| **통신** | All-Gather, Reduce-Scatter | CPU-GPU 전송 (비동기) |
| **파라미터 저장** | ZeRO-3에서 GPU 간 분산 | 모든 GPU에 전체 복사 |
| **메모리 절약** | GPU 메모리 내에서 분산 | CPU 메모리 활용 |
| **통신 비용** | GPU 간 네트워크 통신 | CPU-GPU 메모리 전송 |
| **확장성** | GPU 수에 비례하여 절약 | GPU 수와 무관하게 절약 |

#### RALO의 장점
1. **단일 GPU에서도 효과적**: GPU 간 통신 불필요
2. **CPU 메모리 활용**: 대용량 CPU 메모리 활용 가능
3. **비동기 전송**: 계산과 오프로딩 오버랩 가능
4. **간단한 구현**: 복잡한 분산 통신 로직 불필요

#### ZeRO의 장점
1. **더 빠른 통신**: GPU 간 고속 네트워크 (InfiniBand) 활용
2. **확장성**: GPU 수에 비례하여 메모리 절약
3. **ZeRO-3**: 파라미터까지 분산하여 최대 절약

---

## 개선 제안: 하이브리드 접근

### ZeRO-2 스타일 Gradient Sharding + CPU Offloading

현재 RALO는 모든 gradient를 CPU로 오프로드하지만, ZeRO-2처럼 GPU 간 분산도 고려할 수 있습니다:

```python
# 하이브리드 접근 (의사코드)
def hybrid_gradient_offload():
    if distributed_training:
        # ZeRO-2 스타일: GPU 간 gradient 분산
        reduce_scatter(gradients)  # GPU 간 분산
        # 각 GPU는 자신의 부분만 CPU로 오프로드
        offload_to_cpu(my_gradient_shard)
    else:
        # 단일 GPU: 현재 방식 (전체를 CPU로)
        offload_to_cpu(all_gradients)
```

### 장점
- **분산 학습 시**: GPU 간 분산 + CPU 오프로딩으로 이중 절약
- **단일 GPU 시**: 현재 방식 유지
- **유연성**: 환경에 따라 자동 선택

---

## 결론

1. **ZeRO-2**: Gradient sharding으로 backward 중 메모리 절약
2. **ZeRO-3**: Parameter sharding으로 forward/backward 모두에서 메모리 절약 (통신 증가)
3. **RALO**: CPU 오프로딩으로 GPU 메모리 절약 (단일/분산 모두 지원)

현재 RALO 구현은 ZeRO의 핵심 아이디어(분산 저장)를 CPU 오프로딩으로 구현한 것으로, 단일 GPU 환경에서도 효과적이며 분산 환경에서는 ZeRO와 결합 가능합니다.

---

## NeMo Megatron 메모리 최적화 기법

NVIDIA NeMo Megatron은 대규모 언어 모델(LLM) 학습을 위한 프레임워크로, 여러 병렬화 기법과 메모리 최적화 전략을 제공합니다.

### 1. Tensor Parallelism (TP)

#### 핵심 아이디어
- **레이어 내부 연산을 GPU 간 분산**
- Attention과 MLP의 행렬 연산을 여러 GPU에 분산

#### 구현 방식
```python
# 의사코드: Attention의 QKV 계산을 분산
def tensor_parallel_attention():
    # GPU 0: Q 계산 (Q 행렬의 일부)
    # GPU 1: K 계산 (K 행렬의 일부)
    # GPU 2: V 계산 (V 행렬의 일부)
    # GPU 3: Output projection (Output 행렬의 일부)
    
    # All-Gather로 결과 수집
    q = all_gather(q_shard)
    k = all_gather(k_shard)
    v = all_gather(v_shard)
    
    # Attention 계산
    attn_output = attention(q, k, v)
    
    # Reduce-Scatter로 output 분산
    output = reduce_scatter(attn_output)
```

#### 메모리 절약
- **각 GPU의 파라미터**: `1/TP_size` (TP_size = 텐서 병렬 GPU 수)
- **예**: 7B 모델, TP=4 → 각 GPU: 1.75B 파라미터 (3.5GB BF16)

#### 통신 비용
- **Forward**: All-Gather + Reduce-Scatter per layer
- **Backward**: All-Gather + Reduce-Scatter per layer
- **총 통신**: 2 × (All-Gather + Reduce-Scatter) per layer per step

---

### 2. Pipeline Parallelism (PP)

#### 핵심 아이디어
- **모델 레이어를 여러 스테이지로 분할**
- 각 GPU가 모델의 일부 레이어만 담당

#### 구현 방식
```python
# 의사코드: 4 GPU, 24 레이어 → 각 GPU 6 레이어
def pipeline_parallel_forward():
    # GPU 0: 레이어 0-5
    # GPU 1: 레이어 6-11
    # GPU 2: 레이어 12-17
    # GPU 3: 레이어 18-23
    
    # Micro-batch 0 처리
    stage0_output = gpu0.forward(input)
    stage1_output = gpu1.forward(stage0_output)
    stage2_output = gpu2.forward(stage1_output)
    output = gpu3.forward(stage2_output)
    
    # Micro-batch 1 처리 (파이프라인 오버랩)
    # GPU 0은 micro-batch 1 처리, GPU 1-3은 micro-batch 0 처리
```

#### 메모리 절약
- **각 GPU의 파라미터**: `1/PP_size` (PP_size = 파이프라인 병렬 GPU 수)
- **Activation 메모리**: 각 GPU는 자신의 스테이지 activation만 저장
- **예**: 7B 모델, PP=4 → 각 GPU: 1.75B 파라미터 + 해당 스테이지 activation만

#### 통신 비용
- **Forward**: 스테이지 간 activation 전송 (PP_size - 1)
- **Backward**: 스테이지 간 gradient 전송 (PP_size - 1)
- **파이프라인 버블**: 첫/마지막 micro-batch에서 일부 GPU가 idle

---

### 3. Sequence Parallelism (SP)

#### 핵심 아이디어
- **시퀀스 길이를 GPU 간 분산**
- 각 GPU가 시퀀스의 일부만 처리

#### 구현 방식
```python
# 의사코드: 4 GPU, 시퀀스 길이 4096 → 각 GPU 1024 토큰
def sequence_parallel_attention():
    # GPU 0: 토큰 0-1023
    # GPU 1: 토큰 1024-2047
    # GPU 2: 토큰 2048-3071
    # GPU 3: 토큰 3072-4095
    
    # 각 GPU에서 로컬 attention 계산
    local_attn = local_attention(query_shard, key_shard, value_shard)
    
    # All-Gather로 전체 시퀀스 attention 수집
    global_attn = all_gather(local_attn)
```

#### 메모리 절약
- **Activation 메모리**: `1/SP_size` (SP_size = 시퀀스 병렬 GPU 수)
- **긴 시퀀스에서 효과적**: 시퀀스 길이에 비례하는 activation 메모리 절약
- **예**: 14K 시퀀스, SP=4 → 각 GPU: 3.5K 토큰 activation만 저장

#### 통신 비용
- **Attention 계산**: All-Gather로 전체 attention matrix 수집
- **통신량**: O(seq_len²) per attention layer

---

### 4. Selective Activation Recomputation

#### 핵심 아이디어
- **전체 activation 재계산 대신 선택적 재계산**
- 메모리 사용량과 계산 비용의 균형

#### 구현 방식
```python
# 의사코드
def selective_checkpointing():
    # 중요한 activation만 저장 (예: attention output)
    # 덜 중요한 activation은 재계산 (예: intermediate MLP activations)
    
    if is_checkpoint_layer:
        # Activation 저장
        saved_activation = layer_output
    else:
        # Activation 재계산
        recomputed_activation = recompute(layer)
```

#### 메모리 절약
- **전체 checkpointing**: Activation 메모리 ~70% 절약, 계산 ~30% 증가
- **선택적 checkpointing**: Activation 메모리 ~50% 절약, 계산 ~15% 증가
- **균형**: 메모리와 계산의 최적 균형점 선택

---

### 5. Mixed Precision Training

#### 핵심 아이디어
- **Forward: BF16/FP16, Backward: BF16/FP16, Optimizer: FP32**
- 메모리 사용량 절반으로 감소

#### 메모리 절약
- **Parameters**: FP32 → BF16: 50% 절약
- **Activations**: FP32 → BF16: 50% 절약
- **Gradients**: FP32 → BF16: 50% 절약 (일부는 FP32 유지)

---

### 6. Memory Offloading

#### 핵심 아이디어
- **중간 activation을 CPU로 오프로딩**
- 필요 시 GPU로 다시 로드

#### 구현 방식
```python
# 의사코드
def activation_offloading():
    # Forward pass
    activation = layer(input)
    
    # CPU로 오프로딩
    cpu_activation = activation.cpu()
    del activation  # GPU 메모리 해제
    
    # Backward pass에서 필요 시 다시 로드
    if needed:
        activation = cpu_activation.cuda()
        grad = backward(activation)
```

---

## NeMo Megatron 병렬화 전략 조합

### 일반적인 조합

1. **Data Parallelism (DP)**: 여러 GPU에 데이터 배치 분산
2. **Tensor Parallelism (TP)**: 레이어 내부 연산 분산
3. **Pipeline Parallelism (PP)**: 레이어를 스테이지로 분할
4. **Sequence Parallelism (SP)**: 시퀀스 길이 분산

### 예시: 8 GPU 설정

```
Option 1: TP=4, PP=2
- TP 그룹 0 (GPU 0-3): 모델의 절반 레이어
- TP 그룹 1 (GPU 4-7): 모델의 나머지 절반 레이어

Option 2: TP=2, PP=4
- 4개의 파이프라인 스테이지
- 각 스테이지는 2 GPU로 텐서 병렬

Option 3: TP=2, PP=2, SP=2
- 텐서 병렬 + 파이프라인 병렬 + 시퀀스 병렬
```

### 메모리 사용량 계산 (7B 모델, 8 GPU 예시)

| 병렬화 전략 | Parameters/GPU | Activations/GPU | 총 메모리/GPU |
|------------|----------------|-----------------|---------------|
| DP only | 14GB | 28GB (4K seq) | ~42GB |
| TP=4, PP=2 | 3.5GB | 7GB (4K seq) | ~10.5GB |
| TP=2, PP=4 | 7GB | 3.5GB (4K seq) | ~10.5GB |
| TP=2, PP=2, SP=2 | 7GB | 1.75GB (4K seq) | ~8.75GB |

---

## RALO vs ZeRO vs NeMo Megatron 비교

| 특징 | ZeRO-2/3 | NeMo Megatron | RALO (현재) |
|------|----------|---------------|-------------|
| **병렬화 방식** | Gradient/Parameter Sharding | Tensor/Pipeline/Sequence Parallelism | CPU Offloading |
| **단일 GPU 지원** | 제한적 | 제한적 | 완벽 지원 |
| **분산 학습** | 필수 | 필수 | 선택적 |
| **메모리 절약** | GPU 간 분산 | GPU 간 분산 + 선택적 재계산 | CPU 메모리 활용 |
| **통신 비용** | 중간 (All-Gather) | 높음 (다양한 통신) | 낮음 (CPU-GPU) |
| **구현 복잡도** | 중간 | 높음 | 낮음 |
| **확장성** | GPU 수에 비례 | GPU 수에 비례 | GPU 수와 무관 |
| **긴 시퀀스** | 제한적 | Sequence Parallelism 지원 | CPU offloading으로 지원 |

---

## 하이브리드 접근: RALO + NeMo Megatron 기법

### 제안: Sequence Parallelism + CPU Offloading

NeMo의 Sequence Parallelism과 RALO의 CPU Offloading을 결합:

```python
# 의사코드: 하이브리드 접근
def hybrid_sequence_parallel():
    if distributed_training and long_sequence:
        # Sequence Parallelism: 시퀀스를 GPU 간 분산
        sequence_shard = split_sequence(input, num_gpus)
        
        # 각 GPU에서 처리
        local_output = process_local_shard(sequence_shard)
        
        # CPU로 오프로딩 (추가 메모리 절약)
        if grad_offload:
            offload_to_cpu(local_gradients)
    else:
        # 단일 GPU: 현재 RALO 방식
        offload_to_cpu(all_gradients)
```

### 장점
1. **분산 환경**: Sequence Parallelism으로 activation 메모리 절약
2. **단일 GPU**: CPU Offloading으로 gradient 메모리 절약
3. **하이브리드**: 두 기법 결합으로 최대 절약

---

## 결론

1. **ZeRO**: Optimizer state, gradient, parameter를 GPU 간 분산
2. **NeMo Megatron**: Tensor/Pipeline/Sequence 병렬화 + 선택적 재계산
3. **RALO**: CPU 오프로딩으로 단일/분산 모두 지원

각 접근 방식은 서로 다른 장단점을 가지며, 환경과 요구사항에 따라 선택하거나 결합할 수 있습니다. RALO는 단일 GPU 환경에서도 효과적이며, 분산 환경에서는 NeMo Megatron의 병렬화 기법과 결합하여 더 큰 메모리 절약을 달성할 수 있습니다.

