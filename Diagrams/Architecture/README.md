# RALO System - Architecture Diagrams

이 디렉토리는 RALO (Reinforcement Asynchronous Learning Orchestrator) 시스템의 전체 아키텍처를 보여주는 다이어그램을 포함합니다. 아키텍처 다이어그램은 시스템의 구조, 레이어, 데이터 흐름, 통합, 배포를 종합적으로 시각화합니다.

## 디렉토리 구조

```
Architecture/
├── 00_system_architecture.puml      # 전체 시스템 아키텍처
├── 01_layered_architecture.puml     # 레이어 아키텍처
├── 02_data_flow_architecture.puml   # 데이터 흐름 아키텍처 (시퀀스)
├── 03_integration_architecture.puml # 통합 아키텍처
├── 04_sequence_architecture.puml    # 시퀀스 아키텍처
├── 05_deployment_architecture.puml  # 배포 아키텍처
├── 06_data_flow_detailed.puml      # README 데이터 흐름 상세 (컴포넌트 뷰)
└── README.md                        # 이 문서
```

## 다이어그램 설명

### 00_system_architecture.puml
**목적**: 전체 시스템의 구조와 주요 컴포넌트 관계를 보여주는 종합 아키텍처

**포함 내용**:
- **Application Layer**: RALO Coordinator, Algorithm Registry, Config Manager
- **Orchestration Layer**: Orchestrator Server와 서브컴포넌트들
- **Worker Layer**: Sampler Workers와 Trainer Workers (DDP)
- **Storage Layer**: Weight Storage, Gradient Storage, Problem Dataset
- 레이어 간 연결 및 데이터 흐름
- DDP 통신 (NCCL AllReduce)

**주요 특징**:
- 전체 시스템을 레이어별로 구조화
- 컴포넌트 간 의존성과 통신 패턴 명확화
- 확장 가능한 아키텍처 (여러 워커 인스턴스)

### 01_layered_architecture.puml
**목적**: 시스템을 레이어별로 분해하여 각 레이어의 역할과 책임을 명확히 표현

**포함 레이어**:
1. **Presentation Layer**: HTTP REST API
2. **Application Layer**: OrchestratorServer, RALO Coordinator, Algorithm Layer
3. **Service Layer**: OrchestratorService, ModelService, SamplingService, TrainingService
4. **Business Logic Layer**: ProblemProvider, SampleQueueManager, GradientAggregator, TreePO Algorithm
5. **Data Access Layer**: Weight Storage, Gradient Storage, Problem Dataset
6. **Infrastructure Layer**: vLLM Engine, PyTorch DDP, CPU Optimizer, HTTP Client

**주요 특징**:
- 계층적 아키텍처 패턴
- 각 레이어의 명확한 책임 분리
- 레이어 간 의존성 방향 (상위 → 하위)

### 02_data_flow_architecture.puml
**목적**: 시스템 전체의 데이터 흐름을 시퀀스 다이어그램으로 표현

**주요 데이터 흐름**:
1. **Problem Distribution Flow**: 문제 데이터셋 → Orchestrator → Sampler
2. **Sample Generation Flow**: Sampler 내부 처리 (TreePO, vLLM)
3. **Batch Distribution Flow**: Orchestrator → Trainer
4. **Gradient Computation Flow**: Trainer 내부 처리 (Forward, Backward)
5. **Gradient Upload Flow**: Trainer → Disk (청크 단위) → Orchestrator
6. **Optimizer Step Flow**: Orchestrator 내부 처리 (집계, 옵티마이저 스텝)
7. **Weight Synchronization Flow**: Orchestrator → Disk → Workers

**주요 특징**:
- 전체 데이터 생명주기 추적
- 디스크 기반 저장소의 역할 강조
- 청크 단위 업로드 프로세스 상세화

### 03_integration_architecture.puml
**목적**: RALO 시스템과 외부 시스템 및 인프라스트럭처의 통합을 보여줌

**포함 내용**:
- **External Systems**: HuggingFace Datasets, WandB, File System
- **RALO System**: Orchestrator, Sampler Cluster, Trainer Cluster, RALO Coordinator
- **Infrastructure**: HTTP Network, NCCL, CUDA, vLLM Runtime, PyTorch Runtime

**주요 통합**:
- 외부 데이터 소스 (HuggingFace)
- 실험 추적 (WandB)
- 파일 시스템 (가중치, 그래디언트 저장)
- 네트워크 통신 (HTTP)
- GPU 런타임 (CUDA, vLLM, PyTorch)
- 분산 통신 (NCCL)

### 04_sequence_architecture.puml
**목적**: 전체 학습 루프의 시퀀스를 상세히 표현

**주요 시퀀스 그룹**:
1. **System Initialization**: 시스템 초기화
2. **Problem Distribution**: 문제 분배
3. **Sample Generation**: 샘플 생성
4. **Batch Distribution**: 배치 분배
5. **Gradient Computation**: 그래디언트 계산
6. **Gradient Upload (Chunked)**: 그래디언트 업로드 (청크 단위)
7. **Optimizer Step**: 옵티마이저 스텝
8. **Weight Synchronization**: 가중치 동기화
9. **Monitoring & Control**: 모니터링 및 제어

**주요 특징**:
- 전체 학습 사이클의 완전한 시퀀스
- 조건부 흐름 (버전 체크 등)
- 반복 루프 표현

### 05_deployment_architecture.puml
**목적**: 실제 배포 환경에서의 물리적 구조를 보여줌

**포함 내용**:
- **Orchestrator Node**: CPU 기반, 디스크 저장소
- **Sampler Nodes**: 각 노드당 1개 GPU, 독립 프로세스
- **Trainer Nodes**: 각 노드당 4개 GPU, DDP 프로세스
- 네트워크 연결
- GPU 할당
- 저장소 위치

**배포 시나리오**:
- Orchestrator: 단일 노드, CPU만 사용
- Samplers: 여러 노드, 각 노드 1 GPU
- Trainers: 여러 노드, 각 노드 4 GPU (DDP)

### 06_data_flow_detailed.puml
**목적**: README에 명시된 데이터 흐름을 컴포넌트 다이어그램으로 정확히 표현

**포함 내용**:
- **Problem dataset → Orchestrator (ProblemProvider)**: 문제 데이터셋 로딩
- **Samplers → GET /problem/get**: 샘플러가 문제 요청
- **Samplers → POST /upload → SampleQueueManager**: 샘플 업로드
- **SampleQueueManager → GET /get → Trainers**: 배치 분배
- **Trainers → POST /gradient/upload_chunk + /gradient/upload_finalize → GradientAggregator**: 그래디언트 업로드 (청크 단위)
- **GradientAggregator → optimizer/weights**: 옵티마이저 스텝 및 가중치 저장
- **Samplers & Trainers → GET /weights/***: 가중치 동기화

**주요 특징**:
- README의 ASCII 다이어그램을 PlantUML로 변환
- 각 단계별 HTTP 엔드포인트 명시
- 컴포넌트 간 데이터 흐름을 화살표로 표현
- 여러 워커 인스턴스 표시

## 아키텍처 핵심 원칙

### 1. 중앙 집중식 오케스트레이션
- Orchestrator가 모든 상태를 중앙에서 관리
- 일관된 모델 업데이트 보장
- A3C 스타일의 중앙 집중식 업데이트

### 2. 비동기 처리
- Sampler와 Trainer가 독립적으로 동작
- HTTP를 통한 느슨한 결합
- 확장 가능한 아키텍처

### 3. 디스크 기반 저장소
- 메모리 효율성
- 대규모 모델 지원 (7B+ 파라미터)
- 안정성 (OOM 방지)

### 4. 청크 단위 업로드
- 대용량 그래디언트 처리
- 네트워크 효율성
- 점진적 처리

### 5. 가중치 버전 관리
- 모든 워커의 일관성 보장
- 버전 기반 동기화
- 롤백 가능성

## 시스템 특징

### 확장성
- **Sampler Workers**: 수평 확장 가능 (독립 프로세스)
- **Trainer Workers**: DDP를 통한 수평 확장
- **Orchestrator**: 단일 인스턴스 (중앙 집중식)

### 안정성
- 디스크 기반 저장소로 OOM 방지
- 타임아웃 처리 및 재시도 메커니즘
- 자동 정리 (stale 파일)

### 성능
- 멀티스레드 HTTP 서버
- 비동기 처리
- GPU 가속 (vLLM, PyTorch)
- CPU 옵티마이저 (메모리 효율)

### 유연성
- 알고리즘 플러그인 아키텍처
- 설정 기반 구성
- 독립적인 워커 스케일링

## 데이터 흐름 요약

1. **문제 → 샘플러**: Orchestrator가 샘플러에게 문제 분배
2. **샘플러 → Orchestrator**: 생성된 샘플을 큐에 업로드
3. **Orchestrator → 트레이너**: 샘플 큐에서 배치 분배
4. **트레이너 → Orchestrator**: 계산된 그래디언트를 청크 단위로 업로드
5. **Orchestrator**: 그래디언트 집계 및 옵티마이저 스텝 수행
6. **Orchestrator → 모든 워커**: 업데이트된 가중치 동기화

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

## 관련 문서

- **클래스 다이어그램**: `/Diagrams/Class_Diagrams/` - 개별 클래스 구조
- **컴포넌트 다이어그램**: `/Diagrams/Component/` - 컴포넌트 구조 및 통신
- **메인 README**: `/README.md` - 프로젝트 전체 문서

## 아키텍처 결정 사항

### 왜 중앙 집중식 Orchestrator인가?
1. 일관된 모델 업데이트 보장
2. 효율적인 리소스 관리
3. 확장 가능한 분산 학습
4. 명확한 책임 분리

### 왜 디스크 기반 저장소인가?
1. 메모리 효율성 (대규모 모델 지원)
2. 안정성 (OOM 방지)
3. 확장성 (디스크 공간만큼 확장)

### 왜 청크 단위 업로드인가?
1. 대용량 그래디언트 처리
2. 네트워크 효율성
3. 점진적 처리 가능

## 참고 자료

- [PlantUML 공식 문서](https://plantuml.com/)
- [PlantUML 아키텍처 다이어그램 가이드](https://plantuml.com/architecture-diagram)
- RALO 프로젝트 메인 README: `/README.md`

