# RALO System - Component Diagrams

이 디렉토리는 RALO (Reinforcement Asynchronous Learning Orchestrator) 시스템의 컴포넌트 다이어그램을 포함합니다. 컴포넌트 다이어그램은 시스템의 물리적/논리적 컴포넌트, 컴포넌트 간 인터페이스, 포트, 의존성 관계를 시각화합니다.

## 디렉토리 구조

```
Component/
├── 00_system_overview.puml              # 전체 시스템 컴포넌트 개요
├── 01_orchestrator_component.puml      # Orchestrator 서버 컴포넌트 상세
├── 02_sampler_component.puml           # Sampler 컴포넌트 상세
├── 03_trainer_component.puml          # Trainer 컴포넌트 상세
├── 04_ralo_coordinator.puml            # RALO 코디네이터 컴포넌트
├── 05_deployment_view.puml             # 배포 뷰 (물리적 배치)
├── 06_communication_flow.puml          # 컴포넌트 간 통신 흐름
└── README.md                           # 이 문서
```

## 다이어그램 설명

### 00_system_overview.puml
**목적**: 전체 시스템의 주요 컴포넌트와 관계를 보여주는 high-level 뷰

**포함 내용**:
- Orchestrator Server Component (중앙 서버)
- Sampler Components (여러 인스턴스의 샘플러 워커)
- Trainer Components (여러 인스턴스의 트레이너 워커)
- 컴포넌트 간 HTTP 통신
- 데이터 흐름 (문제, 샘플, 그래디언트, 가중치)

**주요 특징**:
- 시스템의 전체 구조를 한눈에 파악
- 컴포넌트 간 통신 패턴 표시
- 확장 가능한 아키텍처 (여러 워커 인스턴스)

### 01_orchestrator_component.puml
**목적**: Orchestrator 서버의 내부 구조와 서브컴포넌트

**포함 내용**:
- ProblemProvider Component: 문제 데이터 관리
- SampleQueueManager Component: 샘플 큐 관리
- GradientAggregator Component: 그래디언트 집계
- HTTP API Server Component: HTTP 서버 (Bottle, Thread Pool)
- Weight Storage Component: 가중치 저장소
- ExperimentLogger Component: 실험 로깅

**주요 인터페이스**:
- Problem API: `/problem/get`
- Sample API: `/upload`
- Batch API: `/get`
- Gradient API: `/gradient/upload_chunk`, `/gradient/upload_finalize`
- Weight API: `/weights/download`, `/weights/version`
- Control API: `/stats`, `/trainer/register`, `/trainer/heartbeat`, `/step/next`, `/lock/*`

### 02_sampler_component.puml
**목적**: Sampler 워커의 내부 구조

**포함 내용**:
- vLLM Engine Component: 고성능 텍스트 생성 엔진
- SamplingService Component: 샘플링 서비스 관리
- TreePOSamplerAlgorithm Component: TreePO 알고리즘 구현
- SamplerClient Component: HTTP 클라이언트
- ModelService Component: 모델 관리
- OrchestratorService Component: 오케스트레이터 통신 추상화

**주요 흐름**:
1. OrchestratorService를 통해 문제 가져오기
2. TreePO 알고리즘으로 문제 처리
3. vLLM 엔진으로 텍스트 생성
4. 샘플을 OrchestratorService를 통해 업로드
5. 주기적으로 가중치 동기화

### 03_trainer_component.puml
**목적**: Trainer 워커의 내부 구조

**포함 내용**:
- TrainingService Component: 학습 서비스 관리
- CPUOffloadTrainer Component: CPU 오프로드 트레이너
- TreePOTrainerAlgorithm Component: TreePO 알고리즘 구현
- TrainerClient Component: HTTP 클라이언트
- Model Component: 학습 모델
- CPUAdamW Optimizer Component: CPU 기반 옵티마이저
- OrchestratorService Component: 오케스트레이터 통신 추상화

**주요 흐름**:
1. OrchestratorService를 통해 배치 가져오기
2. TreePO 알고리즘으로 손실 계산
3. CPUOffloadTrainer로 그래디언트 계산
4. 그래디언트를 OrchestratorService를 통해 업로드 (청크 단위)
5. 주기적으로 가중치 동기화
6. DDP를 통한 그래디언트 동기화 (AllReduce)

### 04_ralo_coordinator.puml
**목적**: RALO 코디네이터의 구조

**포함 내용**:
- OrchestratorService Component: 오케스트레이터 서비스
- ModelService Component: 모델 서비스
- Algorithm Registry Component: 알고리즘 레지스트리
- Config Manager Component: 설정 관리

**주요 역할**:
- 샘플러와 트레이너 워커 생성 및 관리
- 알고리즘 선택 및 등록
- 설정 관리
- 생명주기 관리

### 05_deployment_view.puml
**목적**: 물리적 배포 구조

**포함 내용**:
- 노드/서버 배치
- GPU 할당
- 네트워크 연결
- 프로세스 배치
- 저장소 위치

**배포 시나리오**:
- **Orchestrator Node**: CPU에서 실행, 디스크 저장소 사용
- **Sampler Nodes**: 각 노드당 1개 GPU, 독립 프로세스
- **Trainer Nodes**: 각 노드당 4개 GPU (DDP), `torchrun --nproc_per_node=4`

### 06_communication_flow.puml
**목적**: 컴포넌트 간 통신 시퀀스

**포함 내용**:
- 문제 요청/응답 흐름
- 샘플 업로드 흐름
- 그래디언트 업로드 흐름 (청크 단위)
- 가중치 동기화 흐름
- 트레이너 등록 및 하트비트
- 통계 및 제어 API

**주요 시퀀스**:
1. **Problem Distribution**: Sampler가 문제 요청
2. **Sample Generation & Upload**: 샘플 생성 후 업로드
3. **Batch Distribution**: Trainer가 배치 요청
4. **Gradient Upload**: 그래디언트 청크 업로드 및 최종화
5. **Optimizer Step**: Orchestrator에서 옵티마이저 스텝 수행
6. **Weight Synchronization**: 워커들이 가중치 동기화

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

## 컴포넌트 관계 요약

### Orchestrator Server
- **역할**: 중앙 집중식 오케스트레이터
- **주요 기능**: 문제 분배, 샘플 큐 관리, 그래디언트 집계, 가중치 버전 관리
- **인터페이스**: HTTP REST API
- **서브컴포넌트**: ProblemProvider, SampleQueueManager, GradientAggregator, HTTP API Server, Weight Storage

### Sampler Worker
- **역할**: 텍스트 생성 및 샘플링
- **주요 기능**: 문제 처리, vLLM 기반 생성, 샘플 업로드
- **인터페이스**: OrchestratorService (HTTP 클라이언트)
- **서브컴포넌트**: vLLM Engine, SamplingService, TreePOSamplerAlgorithm, SamplerClient

### Trainer Worker
- **역할**: 그래디언트 계산 및 전송
- **주요 기능**: 배치 처리, 그래디언트 계산, 그래디언트 업로드
- **인터페이스**: OrchestratorService (HTTP 클라이언트), DDP (NCCL)
- **서브컴포넌트**: TrainingService, CPUOffloadTrainer, TreePOTrainerAlgorithm, TrainerClient, Model, CPUAdamW Optimizer

## 주요 인터페이스

### HTTP API 엔드포인트

#### Problem API
- `GET /problem/get`: 문제 가져오기

#### Sample API
- `POST /upload`: 샘플 업로드

#### Batch API
- `GET /get`: 배치 가져오기

#### Gradient API
- `POST /gradient/upload_chunk`: 그래디언트 청크 업로드
- `POST /gradient/upload_finalize`: 그래디언트 업로드 완료

#### Weight API
- `GET /weights/download`: 가중치 다운로드
- `GET /weights/version`: 가중치 버전 조회

#### Control API
- `GET /stats`: 통계 조회
- `POST /trainer/register`: 트레이너 등록
- `POST /trainer/heartbeat`: 트레이너 하트비트
- `POST /step/next`: 글로벌 스텝 증가
- `POST /lock/acquire`: 락 획득
- `POST /lock/release`: 락 해제

## 데이터 흐름

1. **문제 → 샘플러**: Orchestrator의 ProblemProvider가 샘플러에게 문제 분배
2. **샘플러 → Orchestrator**: 생성된 샘플을 SampleQueueManager에 업로드
3. **Orchestrator → 트레이너**: SampleQueueManager가 트레이너에게 배치 분배
4. **트레이너 → Orchestrator**: 계산된 그래디언트를 GradientAggregator에 업로드
5. **Orchestrator → 모든 워커**: 업데이트된 가중치를 Weight Storage에서 제공

## 확장성

- **Sampler Workers**: 독립적으로 확장 가능 (수평 확장)
- **Trainer Workers**: DDP를 통한 수평 확장
- **Orchestrator**: 단일 인스턴스 (중앙 집중식)

## 참고 자료

- [PlantUML 공식 문서](https://plantuml.com/)
- [PlantUML 컴포넌트 다이어그램 가이드](https://plantuml.com/component-diagram)
- RALO 프로젝트 메인 README: `/README.md`
- 클래스 다이어그램: `/Diagrams/Class_Diagrams/`

