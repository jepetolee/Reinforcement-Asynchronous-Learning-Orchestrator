# RALO Project - Class Diagrams

이 디렉토리는 RALO (Reinforcement Asynchronous Learning Orchestrator) 프로젝트의 모든 클래스를 개별 PlantUML 다이어그램으로 문서화합니다.

## 디렉토리 구조

```
Class_Diagrams/
├── 00_overview.puml                    # 전체 시스템 개요 다이어그램
├── Core/                               # 핵심 클래스
│   ├── OrchestratorServer.puml        # 중앙 오케스트레이터 서버
│   ├── RALO.puml                       # 메인 학습 오케스트레이터
│   └── CPUOffloadTrainer.puml         # CPU 기반 트레이너
├── OrchestratorComponents/             # 오케스트레이터 컴포넌트
│   ├── ProblemProvider.puml           # 문제 제공자
│   ├── SampleQueueManager.puml        # 샘플 큐 관리자
│   └── GradientAggregator.puml        # 그래디언트 집계기
├── Services/                           # 서비스 레이어
│   ├── OrchestratorService.puml       # 오케스트레이터 서비스
│   ├── ModelService.puml              # 모델 서비스
│   ├── SamplingService.puml           # 샘플링 서비스
│   └── TrainingService.puml           # 학습 서비스
├── Clients/                            # HTTP 클라이언트
│   ├── TrainerClient.puml             # 트레이너 클라이언트
│   └── SamplerClient.puml             # 샘플러 클라이언트
├── Algorithms/                         # 알고리즘
│   ├── SamplerAlgorithm.puml          # 샘플러 알고리즘 (추상)
│   ├── TrainerAlgorithm.puml          # 트레이너 알고리즘 (추상)
│   ├── TreePOSamplerAlgorithm.puml    # TreePO 샘플러 구현
│   ├── TreePOTrainerAlgorithm.puml    # TreePO 트레이너 구현
│   ├── SampleSchema.puml              # 샘플 스키마 (추상)
│   └── TreePOSampleSchema.puml        # TreePO 샘플 스키마
├── Config/                             # 설정 클래스
│   ├── SamplerConfig.puml             # 샘플러 설정
│   ├── TrainerConfig.puml             # 트레이너 설정
│   ├── OrchestratorConfig.puml        # 오케스트레이터 설정
│   ├── WandbConfig.puml               # WandB 설정
│   ├── DatasetConfig.puml             # 데이터셋 설정
│   └── LogControlConfig.puml          # 로그 제어 설정
└── Utilities/                          # 유틸리티 클래스
    ├── ExperimentLogger.puml          # 실험 로거 계층
    ├── CPUAdamW.puml                  # CPU AdamW 옵티마이저 계층
    └── TreeNode.puml                  # 트리 노드 (TreePO용)
```

## 사용 방법

### PlantUML 렌더링

각 `.puml` 파일은 PlantUML로 렌더링할 수 있습니다:

#### VS Code
1. PlantUML 확장 프로그램 설치
2. `.puml` 파일 열기
3. `Alt + D` 또는 `Cmd + Option + D`로 미리보기

#### 온라인 도구
- [PlantUML Online Server](http://www.plantuml.com/plantuml/uml/)
- 파일 내용을 복사하여 붙여넣기

#### 명령줄
```bash
# Java가 필요합니다
java -jar plantuml.jar *.puml
```

## 다이어그램 설명

### 00_overview.puml
전체 시스템의 주요 컴포넌트와 관계를 보여주는 개요 다이어그램입니다.

### Core/
- **OrchestratorServer**: 중앙 오케스트레이터 서버. 문제 큐, 샘플 큐, 그래디언트 집계, 가중치 버전 관리를 담당
- **RALO**: 메인 학습 오케스트레이터. 샘플러와 트레이너를 조율
- **CPUOffloadTrainer**: GPU 메모리를 절약하기 위해 그래디언트를 CPU로 오프로드하는 트레이너

### OrchestratorComponents/
- **ProblemProvider**: 샘플러에 문제 데이터를 제공하고 추적
- **SampleQueueManager**: 샘플러와 트레이너 간의 샘플 큐 관리
- **GradientAggregator**: 트레이너로부터 받은 그래디언트를 집계하고 옵티마이저 스텝 수행

### Services/
- **OrchestratorService**: 오케스트레이터 서버와의 통신을 추상화하는 서비스 레이어
- **ModelService**: 모델과 토크나이저 관리
- **SamplingService**: vLLM 기반 텍스트 생성 관리
- **TrainingService**: 학습 작업 관리 및 그래디언트 수집

### Clients/
- **TrainerClient**: 트레이너 프로세스용 HTTP 클라이언트
- **SamplerClient**: 샘플러 워커용 HTTP 클라이언트

### Algorithms/
- **SamplerAlgorithm**: 샘플러 알고리즘의 추상 기본 클래스
- **TrainerAlgorithm**: 트레이너 알고리즘의 추상 기본 클래스
- **TreePOSamplerAlgorithm**: TreePO 샘플러 구현
- **TreePOTrainerAlgorithm**: TreePO 트레이너 구현
- **SampleSchema**: 샘플 스키마의 추상 기본 클래스
- **TreePOSampleSchema**: TreePO 샘플 스키마 구현

### Config/
모든 설정 클래스는 dataclass를 사용하여 타입 안전한 설정을 제공합니다.

### Utilities/
- **ExperimentLogger**: 실험 로깅을 위한 추상 기본 클래스
  - **NoOpLogger**: 로깅 비활성화용 구현
  - **WandbLogger**: Weights & Biases 통합 구현
- **CPUAdamW**: CPU 기반 AdamW 옵티마이저 팩토리
  - **SoloCPUAdamW**: 단일 GPU용 구현
  - **DistributedCPUAdamW**: 다중 GPU용 구현
- **TreeNode**: TreePO 알고리즘에서 사용하는 트리 노드 구조

## 관계 표기법

- `--|>`: 상속 (Inheritance)
- `*--`: 컴포지션 (Composition)
- `o--`: 집약 (Aggregation)
- `..>`: 의존성 (Dependency)
- `-->`: 연관 (Association)

## 업데이트

새로운 클래스를 추가하거나 기존 클래스를 수정할 때는 해당 다이어그램 파일을 업데이트하세요.

## 참고 자료

- [PlantUML 공식 문서](https://plantuml.com/)
- [PlantUML 클래스 다이어그램 가이드](https://plantuml.com/class-diagram)
- RALO 프로젝트 메인 README: `/README.md`

