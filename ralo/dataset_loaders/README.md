# Dataset Loaders

데이터셋 로더 시스템은 다양한 HuggingFace 데이터셋을 로드하고 변환하는 유연한 구조를 제공합니다.

## 구조

- **BaseDatasetLoader**: 모든 로더의 추상 베이스 클래스
- **CompetitionMathLoader**: competition_math 데이터셋 전용 로더
- **GenericHFDatasetLoader**: 일반적인 HuggingFace 데이터셋 자동 감지 로더
- **CombinedDatasetLoader**: 여러 데이터셋을 조합하는 로더

## 사용자 정의 필터 함수

시스템은 자동 필터링 로직을 제공하지 않습니다. 대신 사용자가 직접 필터링 로직을 제공해야 합니다.

### 필터 함수 작성 예제

```python
# my_filters.py
def filter_levels(example):
    """Level 3, 4, 5만 필터링"""
    level = example.get("level")
    return level in ["Level 3", "Level 4", "Level 5"]

def filter_by_complexity(example):
    """문제 길이로 필터링"""
    problem = example.get("problem", "")
    word_count = len(problem.split())
    return 50 <= word_count <= 200
```

### 설정 파일에서 사용

```yaml
dataset:
  name: qwedsacf/competition_math
  split: train
  filter_fn: "my_filters:filter_levels"  # 모듈 경로로 지정
  shuffle_seed: 42
```

### 다중 데이터셋 예제

```yaml
dataset:
  shuffle_seed: 42
  datasets:
    - name: qwedsacf/competition_math
      split: train
      filter_fn: "my_filters:filter_levels"
    - name: gsm8k
      split: train
      filter_fn: "my_filters:filter_by_complexity"
```

## 필터 함수 규칙

- 필터 함수는 `Dict[str, Any]` (dataset example)를 받아서 `bool`을 반환해야 합니다
- `True`를 반환하면 해당 예제가 포함됩니다
- `False`를 반환하면 제외됩니다
- 필터 함수가 없으면 모든 데이터가 포함됩니다

## 기존 설정 파일

기존 설정 파일들 (`my_exp.yaml`, `example.yaml` 등)은 그대로 유지됩니다.
`filter_levels`가 설정되어 있어도 무시되며, 모든 데이터가 포함됩니다.
필터링이 필요하면 `filter_fn`을 추가하세요.

