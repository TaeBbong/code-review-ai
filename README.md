# ReviewBot Backend

LLM 기반 자동 코드 리뷰 시스템의 백엔드입니다. Git diff를 분석하여 코드 이슈, 테스트 제안, 패치 제안 등을 JSON 형태로 반환합니다.

---

## 목차

1. [프로젝트 구조](#프로젝트-구조)
2. [아키텍처 개요](#아키텍처-개요)
3. [핵심 컴포넌트 상세](#핵심-컴포넌트-상세)
4. [데이터 흐름](#데이터-흐름)
5. [Variant 시스템](#variant-시스템)
6. [새로운 Variant 추가하기](#새로운-variant-추가하기)
7. [성능 평가 시스템](#성능-평가-시스템)
8. [환경 설정](#환경-설정)
9. [실행 방법](#실행-방법)

---

## 프로젝트 구조

```
backend/
├── main.py                      # FastAPI 앱 진입점
├── config/
│   └── settings.py              # 환경 설정 (pydantic-settings)
├── api/
│   └── routes.py                # HTTP 엔드포인트 정의
├── domain/
│   ├── prompts/
│   │   ├── registry.py          # PromptPack 로딩 및 관리
│   │   └── packs/               # Variant별 프롬프트 팩
│   │       ├── G0-baseline/
│   │       │   ├── manifest.yaml
│   │       │   ├── review.system.txt
│   │       │   ├── review.user.txt
│   │       │   ├── repair.system.txt
│   │       │   └── repair.user.txt
│   │       ├── G1-mapreduce/
│   │       │   └── ...
│   │       ├── G2-iterative/
│   │       │   └── ...
│   │       └── G3-multipersona/
│   │           └── ...
│   ├── schemas/
│   │   ├── review.py            # Pydantic 스키마 (Request/Response)
│   │   └── diff.py              # DiffChunk 스키마 (map-reduce용)
│   ├── services/
│   │   └── review_service.py    # ReviewService (Facade 패턴)
│   └── tools/
│       ├── git_diff.py          # Git diff 유틸리티
│       ├── get_symbols.py       # diff에서 심볼 추출 (함수, 클래스 등)
│       └── ripgrep_refs.py      # ripgrep 기반 심볼 레퍼런스 검색
├── pipelines/
│   ├── base.py                  # ReviewPipeline ABC (Template Method)
│   ├── registry.py              # PipelineRegistry
│   ├── evidence/
│   │   └── refs_builder.py      # evidence_pack.refs 빌더
│   ├── presets/                 # Variant별 파이프라인 설정
│   │   ├── g0-baseline.yaml
│   │   ├── g1-mapreduce.yaml
│   │   ├── g2-iterative.yaml
│   │   └── g3-multipersona.yaml
│   └── variants/                # 파이프라인 구현체
│       ├── g0_baseline.py       # 단순 diff → LLM
│       ├── g1_mapreduce.py      # 파일별 분할 + evidence 수집
│       ├── g2_iterative.py      # 2-pass 오탐 필터링
│       └── g3_multipersona.py   # 다중 관점 병렬 리뷰
├── evaluation/                  # 성능 평가 시스템
│   ├── schemas.py               # 평가 스키마 (EvalSample, SampleScore 등)
│   ├── loader.py                # YAML 데이터셋 로더
│   ├── scorer.py                # 매칭 및 스코어링 로직
│   ├── evaluator.py             # 평가 실행기
│   ├── langsmith_integration.py # LangSmith 연동
│   ├── cli.py                   # CLI 스크립트
│   └── datasets/                # 평가 데이터셋
│       └── v1_initial.yaml      # 초기 데이터셋 (20개 샘플)
├── llm/
│   ├── base.py                  # LLMAdapter ABC, AdapterChatModel
│   ├── provider.py              # get_llm_adapter() 팩토리
│   ├── ollama.py                # Ollama 어댑터
│   ├── openai_compat.py         # OpenAI 호환 어댑터 (vLLM 등)
│   └── invoke.py                # Chain 실행 헬퍼
├── middleware/
│   └── request_context.py       # run_id 생성 미들웨어
├── exceptions/
│   └── handlers.py              # 전역 예외 핸들러
└── shared/
    ├── context.py               # ContextVar (run_id)
    ├── helpers.py               # JSON 추출 유틸리티
    ├── logging.py               # 로깅 설정
    └── parser.py                # validate_or_repair 로직
```

---

## 아키텍처 개요

이 프로젝트는 **확장 가능한 코드 리뷰 파이프라인**을 목표로 설계되었습니다. 핵심 설계 원칙:

### 1. Variant 기반 확장성

- 동일한 API로 여러 리뷰 전략(variant)을 지원
- **PromptPack**: LLM에게 전달할 프롬프트 집합
- **Pipeline**: diff 처리, LLM 호출, 후처리 로직

```
┌─────────────────────────────────────────────────────────────┐
│                      ReviewService                          │
│                    (Facade / Orchestrator)                  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      PromptPackRegistry  PipelineRegistry  LLMProvider
              │               │               │
              ▼               ▼               ▼
         PromptPack     ReviewPipeline   LLMAdapter
         (프롬프트)      (실행 로직)      (LLM 호출)
```

### 2. 사용된 디자인 패턴

| 패턴 | 적용 위치 | 목적 |
|------|-----------|------|
| **Facade** | `ReviewService` | 복잡한 내부 로직을 단순한 인터페이스로 제공 |
| **Template Method** | `ReviewPipeline` | 파이프라인 골격은 고정, variant별 변경은 hook으로 |
| **Adapter** | `LLMAdapter` | 다양한 LLM 백엔드를 동일한 인터페이스로 추상화 |
| **Registry** | `PromptPackRegistry`, `PipelineRegistry` | 런타임에 variant 컴포넌트를 동적 로딩 |
| **Factory** | `get_llm_adapter()` | 설정에 따라 적절한 LLM 어댑터 생성 |

---

## 핵심 컴포넌트 상세

### 1. `main.py` - 애플리케이션 진입점

```python
def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="ReviewBot", version="0.1.0")
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(router)
    return app
```

- FastAPI 앱을 생성하고 미들웨어, 예외 핸들러, 라우터를 등록합니다.
- `RequestContextMiddleware`는 매 요청마다 고유한 `run_id`를 생성합니다.

### 2. `config/settings.py` - 환경 설정

`pydantic-settings`를 사용하여 환경 변수와 `.env` 파일에서 설정을 로드합니다.

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `llm_provider` | `"ollama"` | LLM 백엔드 (`ollama` \| `openai_compat`) |
| `ollama_model` | `"qwen3:4b"` | Ollama 모델명 |
| `ollama_base_url` | `"http://localhost:11434"` | Ollama 서버 URL |
| `openai_compat_base_url` | `"http://localhost:8000/v1"` | vLLM 등 OpenAI 호환 서버 URL |
| `temperature` | `0.3` | LLM 응답 다양성 |
| `max_tokens` | `2048` | 최대 출력 토큰 |
| `review_default_variant` | `"G0-baseline"` | 기본 variant |
| `review_packs_dir` | `"backend/domain/prompts/packs"` | 프롬프트 팩 디렉토리 |
| `review_presets_dir` | `"backend/pipelines/presets"` | 파이프라인 프리셋 디렉토리 |
| `review_repo_path` | `None` | 리뷰할 Git 저장소 경로 (None이면 현재 디렉토리) |

### 3. `api/routes.py` - API 엔드포인트

```
GET  /health         → { "ok": true }
GET  /schema/review  → Request/Response JSON Schema
POST /review         → 코드 리뷰 실행
```

**POST /review 요청 예시:**
```json
{
  "diff": "diff --git a/...",      // 직접 diff 전달 (선택)
  "diff_target": "staged",         // staged | worktree | commit..commit
  "variant_id": "g0-baseline"      // 사용할 variant
}
```

### 4. `domain/services/review_service.py` - ReviewService

**얇은 오케스트레이터(Facade)** 역할을 합니다:

```python
async def review(self, req: ReviewRequest) -> ReviewResult:
    # 1) variant_id로 PromptPack 로드
    variant_id = self._prompt_registry.resolve_variant(req.variant_id)
    pack = self._prompt_registry.get(variant_id)

    # 2) preset YAML에서 Pipeline 클래스 로드 및 인스턴스화
    spec = self._pipeline_registry.load_spec(variant_id)
    pipeline = self._pipeline_registry.build_pipeline(spec, pack=pack)

    # 3) 파이프라인 실행
    result = await pipeline.run(req)
    return result
```

**왜 이렇게 구현했나요?**

- 모든 복잡한 로직(diff 수집, chunking, LLM 호출, repair)은 Pipeline 내부에 있음
- ReviewService는 "어떤 variant를 실행할지"만 결정
- 새 variant 추가 시 ReviewService 수정 불필요

### 5. `pipelines/base.py` - ReviewPipeline (Template Method)

파이프라인의 **골격**을 정의하는 추상 클래스입니다:

```python
class ReviewPipeline(ABC):
    async def run(self, req: ReviewRequest) -> ReviewResult:
        # 1) diff 준비 (hook)
        diff, diff_target = await self.resolve_diff(req)

        # 2) 청크 분할 (hook) - map-reduce용
        chunks = await self.split_chunks(diff)

        # 3) LLM + Parser 준비
        adapter = get_llm_adapter()
        llm = AdapterChatModel(adapter)

        # 4) 리뷰 실행 (단일 vs map-reduce)
        if len(chunks) <= 1:
            result = await self._review_single(...)
        else:
            result = await self._review_map_reduce(...)

        # 5) 메타 정보 주입
        result.meta.variant_id = variant_id
        ...

        # 6) 후처리 (hook)
        await self.after_run(req=req, result=result, ...)
        return result

    # ========== Hooks (variant별 오버라이드) ==========
    @abstractmethod
    async def resolve_diff(self, req) -> tuple[str, str]:
        """diff 텍스트와 타겟 레이블 반환"""

    async def split_chunks(self, diff: str) -> List[DiffChunk]:
        """diff를 청크로 분할 (기본: 분할 안 함)"""

    async def build_review_payload(self, *, req, diff, diff_target, chunk) -> dict:
        """LLM에 전달할 payload 구성 (확장 가능)"""

    async def reduce_results(self, results: List[ReviewResult]) -> ReviewResult:
        """여러 리뷰 결과를 병합 (기본: 모든 이슈 합치기)"""

    async def after_run(self, *, req, result, raw_text, ...) -> None:
        """후처리 hook (로깅, 저장 등)"""
```

**왜 Template Method 패턴인가요?**

- 파이프라인 실행 순서(diff 준비 → 청크 분할 → LLM 호출 → 병합 → 후처리)는 **고정**
- variant별로 다른 부분만 **hook 메서드**로 오버라이드
- 코드 중복 최소화, 일관된 실행 흐름 보장

**Map-Reduce 동작:**
- `split_chunks()`가 단일 청크 반환 → 기존처럼 단일 LLM 호출
- `split_chunks()`가 여러 청크 반환 → 병렬 리뷰 후 `reduce_results()`로 병합

### 6. `pipelines/variants/g0_baseline.py` - BaselinePipeline

가장 기본적인 파이프라인 구현:

```python
class BaselinePipeline(ReviewPipeline):
    async def resolve_diff(self, req):
        # 요청에 diff가 직접 있으면 그대로 사용
        raw = (req.diff or "").strip()
        if raw:
            return raw, "raw"

        # 없으면 git에서 diff 가져오기
        diff_target = req.diff_target or self.params.get("diff_source") or "staged"
        diff = get_git_diff(diff_target=diff_target, ...)
        return diff, diff_target
```

### 7. `domain/prompts/registry.py` - PromptPackRegistry

프롬프트 팩을 로드하고 관리합니다:

```
packs/
└── G0-baseline/
    ├── manifest.yaml      # 메타정보 + 템플릿 매핑
    ├── review.system.txt  # 리뷰용 system 프롬프트
    ├── review.user.txt    # 리뷰용 user 프롬프트
    ├── repair.system.txt  # repair용 system 프롬프트
    └── repair.user.txt    # repair용 user 프롬프트
```

**manifest.yaml 구조:**
```yaml
id: g0-baseline
description: "Baseline strict JSON output"
templates:
  review_system: review.system.txt
  review_user: review.user.txt
  repair_system: repair.system.txt
  repair_user: repair.user.txt
params:
  bad_max_chars: 4000
```

**왜 프롬프트를 파일로 분리했나요?**

- 프롬프트 수정 시 코드 변경 불필요
- A/B 테스트를 위한 variant 간 비교 용이
- 비개발자도 프롬프트 튜닝 가능

### 8. `llm/` - LLM 추상화 계층

```
┌─────────────────────────────────────────┐
│            LLMAdapter (ABC)             │
│  - provider: str                        │
│  - model_name: str                      │
│  - ainvoke(messages) -> str             │
└─────────────────────────────────────────┘
          ▲                    ▲
          │                    │
┌─────────────────┐  ┌─────────────────────┐
│  OllamaAdapter  │  │ OpenAICompatAdapter │
│   (Ollama API)  │  │  (vLLM, LocalAI 등) │
└─────────────────┘  └─────────────────────┘
```

**AdapterChatModel**: LangChain의 `BaseChatModel`을 상속하여 LangChain 생태계(프롬프트 템플릿, 체인 등)와 호환됩니다.

**왜 Adapter 패턴인가요?**

- 새 LLM 백엔드 추가 시 `LLMAdapter`만 구현하면 됨
- 비즈니스 로직은 LLM 백엔드에 의존하지 않음
- 테스트 시 Mock Adapter 주입 가능

### 9. `shared/parser.py` - validate_or_repair

LLM 출력이 스키마에 맞지 않을 때 자동으로 repair합니다:

```python
async def validate_or_repair(*, raw_text, repair_chain, bad_max_chars):
    raw_json = extract_json_text(raw_text)

    try:
        # 1차 파싱 시도
        parsed = ReviewResult.model_validate_json(raw_json)
        return parsed, False, raw_json, None  # repair 미사용
    except ValidationError:
        # 실패 시 repair chain 호출
        fixed_msg = await repair_chain.ainvoke({"bad": raw_json[:bad_max_chars]})
        fixed_json = extract_json_text(fixed_msg.content)
        parsed = ReviewResult.model_validate_json(fixed_json)
        return parsed, True, raw_json, fixed_json  # repair 사용됨
```

**왜 repair 메커니즘이 필요한가요?**

- 소형 LLM은 종종 스키마를 완벽히 따르지 못함
- repair chain을 통해 잘못된 JSON을 수정
- `meta.repair_used`로 repair 사용 여부 추적 가능

---

## 데이터 흐름

```
┌─────────┐    POST /review    ┌─────────────┐
│ Client  │ ────────────────▶  │   routes.py │
└─────────┘                    └──────┬──────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │ ReviewService │
                              └───────┬───────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
    PromptPackRegistry        PipelineRegistry         LLMProvider
              │                       │                       │
              ▼                       ▼                       ▼
    Load PromptPack           Load Pipeline            Get LLMAdapter
    (system/user prompts)     (BaselinePipeline)       (Ollama/vLLM)
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │   pipeline    │
                              │    .run()     │
                              └───────┬───────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
   resolve_diff()              build_chain()               invoke_chain()
   (Git diff 가져오기)          (프롬프트 구성)              (LLM 호출)
                                                                    │
                                                                    ▼
                                                          ┌─────────────────┐
                                                          │ validate_or_    │
                                                          │ repair()        │
                                                          └────────┬────────┘
                                                                   │
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │  ReviewResult   │
                                                          │     (JSON)      │
                                                          └─────────────────┘
```

---

## Variant 시스템

Variant는 **프롬프트 팩 + 파이프라인**의 조합입니다.

### 현재 구현된 Variant

| Variant ID | 프롬프트 팩 | 파이프라인 | 특징 |
|------------|-------------|------------|------|
| `g0-baseline` | `G0-baseline` | `BaselinePipeline` | 단순 diff → LLM → JSON |
| `g1-mapreduce` | `G1-mapreduce` | `MapReducePipeline` | 파일별 분할 + ripgrep evidence 수집 |
| `g2-iterative` | `G2-iterative` | `IterativeRefinementPipeline` | 2-pass 리뷰로 오탐 필터링 |
| `g3-multipersona` | `G3-multipersona` | `MultiPersonaPipeline` | 다중 관점 병렬 리뷰 |

#### G1-mapreduce 상세

G1-mapreduce는 두 가지 핵심 기능을 제공합니다:

**1. Evidence 수집 (ripgrep 기반)**
- diff에서 변경된 심볼(함수, 클래스 등)을 추출
- ripgrep으로 해당 심볼의 사용처(레퍼런스)를 검색
- `evidence_pack.refs`에 검색 결과를 포함하여 LLM에 전달
- LLM이 변경의 영향 범위를 더 정확히 파악 가능

**2. Map-Reduce 병렬 처리**
- 파일 수가 `min_files_for_split` 이상이면 파일별로 diff를 분할
- 각 파일을 병렬로 리뷰 (최대 `max_concurrency`개 동시 실행)
- 개별 리뷰 결과를 `reduce_results()`로 병합
- 대규모 diff에서 토큰 제한 회피 + 파일별 집중 리뷰 가능

```yaml
# g1-mapreduce.yaml 설정 예시
params:
  min_files_for_split: 2   # 최소 파일 수 (미만이면 분할 안 함)
  max_files_for_split: 20  # 최대 파일 수 (초과 시 상위 N개만)
  max_concurrency: 4       # 병렬 LLM 호출 수
  max_symbols: 12          # 검색할 심볼 최대 개수
  top_k_per_symbol: 6      # 심볼당 레퍼런스 최대 개수
```

#### G2-iterative 상세

G2-iterative는 자기 검증을 통해 오탐(false positive)을 줄입니다:

**동작 방식:**
1. **1차 리뷰**: 포괄적으로 이슈 수집 (의심스러운 것도 포함)
2. **2차 리뷰**: 1차 결과를 다시 검토하여 오탐 필터링
3. **결과**: 검증된 이슈만 최종 출력

**출력 특징:**
- summary.key_points에 필터링 정보 추가: `Filtered N potential false positive(s) from initial M issue(s)`

#### G3-multipersona 상세

G3-multipersona는 여러 관점의 리뷰어를 시뮬레이션합니다:

**기본 페르소나 3종:**
| 페르소나 | 집중 영역 |
|---------|----------|
| Security Reviewer | 보안 취약점, OWASP Top 10, injection 공격 |
| Performance Reviewer | 성능 병목, N+1 쿼리, 알고리즘 복잡도 |
| Maintainability Reviewer | 코드 복잡도, 네이밍, SOLID 원칙 |

**동작 방식:**
- 각 페르소나가 병렬로 diff를 리뷰 (최대 `max_concurrency`개 동시 실행)
- 개별 리뷰 결과를 `reduce_persona_results()`로 병합
- 이슈 카테고리에 페르소나 태그 추가: `[Security Reviewer] SQL Injection`

```yaml
# g3-multipersona.yaml 설정 예시
params:
  personas:              # 사용할 페르소나 목록
    - security
    - performance
    - maintainability
  max_concurrency: 3     # 병렬 LLM 호출 수
```

### Variant 선택 우선순위

1. 요청의 `variant_id` 파라미터
2. 환경 변수 `REVIEW_DEFAULT_VARIANT`
3. 하드코딩된 기본값 `"G0-baseline"`

> **참고**: `variant_id`는 대소문자를 구분하지 않습니다. `g1-mapreduce`, `G1-mapreduce`, `G1-MAPREDUCE` 모두 동일하게 처리됩니다.

### 허용된 Variant 제한

운영 환경에서 특정 variant만 허용하려면:

```bash
REVIEW_ALLOWED_VARIANTS_RAW="g0-baseline,g1-mapreduce,g2-iterative,g3-multipersona"
```

허용되지 않은 variant 요청 시 기본 variant로 폴백됩니다.

---

## 새로운 Variant 추가하기

### Step 1: 프롬프트 팩 생성

`backend/domain/prompts/packs/` 아래에 새 폴더 생성:

```
packs/
└── G2-myvariant/
    ├── manifest.yaml
    ├── review.system.txt
    ├── review.user.txt
    ├── repair.system.txt
    └── repair.user.txt
```

**manifest.yaml:**
```yaml
id: g2-myvariant
description: "My custom review strategy"
templates:
  review_system: review.system.txt
  review_user: review.user.txt
  repair_system: repair.system.txt
  repair_user: repair.user.txt
params:
  bad_max_chars: 4000
  my_custom_param: "value"
```

**review.user.txt 예시:**
```
Review the following git diff and output JSON that matches the schema.

{format_instructions}

variant_id: {variant_id}

git diff:
{diff}
```

- `{format_instructions}`: Pydantic 스키마에서 자동 생성된 JSON 형식 지침
- `{variant_id}`, `{diff}`: 파이프라인에서 주입하는 변수

### Step 2: 파이프라인 구현

`backend/pipelines/variants/g2_myvariant.py`:

```python
from backend.pipelines.base import ReviewPipeline
from backend.config.settings import settings
from backend.domain.tools.git_diff import get_git_diff


class MyVariantPipeline(ReviewPipeline):
    async def resolve_diff(self, req):
        """
        diff를 준비하는 hook.
        Returns: (diff_text, diff_target_label)
        """
        raw = (getattr(req, "diff", None) or "").strip()
        if raw:
            return raw, "raw"

        diff_target = (
            getattr(req, "diff_target", None)
            or self.params.get("diff_source")
            or "staged"
        )
        repo_path = str(settings.review_repo_path) if settings.review_repo_path else None

        diff = get_git_diff(
            diff_target=diff_target,
            repo_path=repo_path,
            context_lines=int(self.params.get("context_lines", 3)),
        )
        return diff, diff_target

    async def build_review_payload(self, *, req, diff, diff_target):
        """
        LLM에 전달할 payload를 확장하는 hook.
        기본 payload에 커스텀 데이터 추가 가능.
        """
        payload = await super().build_review_payload(
            req=req, diff=diff, diff_target=diff_target
        )
        # 커스텀 필드 추가
        payload["my_custom_field"] = self.params.get("my_custom_param", "default")
        return payload

    async def after_run(self, *, req, result, raw_text, raw_json, fixed_json):
        """
        후처리 hook.
        로깅, DB 저장, 외부 알림 등.
        """
        # 예: 결과 로깅
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Review completed: {len(result.issues)} issues found")
```

### Step 3: 프리셋 YAML 작성

`backend/pipelines/presets/g2-myvariant.yaml`:

```yaml
id: g2-myvariant
pipeline: backend.pipelines.variants.g2_myvariant:MyVariantPipeline
params:
  diff_source: staged
  context_lines: 5
  my_custom_param: "custom_value"
  bad_max_chars: 6000
```

- `pipeline`: `모듈경로:클래스명` 형식
- `params`: 파이프라인 생성자에 전달될 파라미터

### Step 4: 테스트

```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "variant_id": "g2-myvariant",
    "diff_target": "staged"
  }'
```

---

## 성능 평가 시스템

Variant별 리뷰 품질을 측정하기 위한 평가 프레임워크입니다.

### 평가 지표

| 지표 | 설명 | 계산 방법 |
|------|------|----------|
| **Precision** | 발견한 이슈 중 실제 이슈 비율 | TP / (TP + FP) |
| **Recall** | 실제 이슈 중 발견한 비율 | TP / (TP + FN) |
| **F1 Score** | Precision-Recall 조화평균 | 2 × (P × R) / (P + R) |

### 데이터셋 구조

평가 데이터셋은 YAML 형식으로 `backend/evaluation/datasets/`에 저장됩니다:

```yaml
# v1_initial.yaml 예시
samples:
  - id: "correctness-001"
    metadata:
      primary_category: "correctness"
      difficulty: "easy"
      tags: ["null-check", "runtime-error"]
    input:
      diff: |
        diff --git a/service.py b/service.py
        ...
    expected:
      issues:
        - category: "correctness"
          severity_min: "medium"
          title_keywords: ["None", "null", "NoneType"]
          description_keywords: ["체크", "null"]
      min_issues: 1
      max_issues: 3
```

**현재 데이터셋 (v1_initial):**
| 카테고리 | 개수 | 예시 |
|---------|------|------|
| correctness | 6 | None 체크 누락, off-by-one, 타입 불일치 |
| maintainability | 5 | 매직넘버, 코드중복, 중첩조건, SRP 위반 |
| performance | 5 | N+1 쿼리, O(n²) 알고리즘, 메모리 과사용 |
| security | 4 | SQL Injection, 하드코딩 시크릿, Path Traversal |

### CLI 사용법

```bash
# 데이터셋 목록 확인
uv run python -m backend.evaluation.cli list

# 단일 variant 평가
uv run python -m backend.evaluation.cli run-local v1_initial g1-mapreduce

# 여러 variant 비교
uv run python -m backend.evaluation.cli compare v1_initial \
    g0-baseline g1-mapreduce g2-iterative g3-multipersona

# 결과를 JSON으로 저장
uv run python -m backend.evaluation.cli run-local v1_initial g1-mapreduce -o result.json
```

### 코드에서 사용

```python
from backend.evaluation import Evaluator, load_dataset_by_name
from backend.pipelines.registry import get_pipeline
from backend.domain.schemas.review import ReviewRequest

# 데이터셋 로드
dataset = load_dataset_by_name("v1_initial")
evaluator = Evaluator(dataset=dataset)

# 리뷰 함수 정의
async def review_fn(diff: str, variant_id: str):
    pipeline = get_pipeline(variant_id)
    req = ReviewRequest(diff=diff, variant_id=variant_id)
    return await pipeline.run(req)

# 평가 실행
result = await evaluator.run(
    review_fn=review_fn,
    variant_id="g1-mapreduce",
    max_concurrency=4,
)

# 결과 확인
print(f"Precision: {result.overall_precision:.2%}")
print(f"Recall: {result.overall_recall:.2%}")
print(f"F1 Score: {result.overall_f1:.2%}")
```

### 이슈 매칭 로직

예상 이슈와 예측 이슈의 매칭 기준:

1. **카테고리 일치** (필수)
2. **심각도 >= 최소 심각도** (필수)
3. **제목 키워드 포함** (OR 조건 - 하나라도 포함되면 OK)
4. **설명 키워드 포함** (OR 조건)
5. **파일 패턴 매칭** (선택적)
6. **라인 번호 ±tolerance** (선택적, 기본 ±3)

### LangSmith 연동 (선택적)

LangSmith를 사용하면 실험 결과를 웹 UI에서 추적할 수 있습니다:

```bash
# 환경 변수 설정
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=lsv2_pt_xxxxx

# 데이터셋 업로드
uv run python -m backend.evaluation.cli upload v1_initial

# LangSmith 실험 실행
uv run python -m backend.evaluation.cli run-langsmith \
    code-review-eval-v1_initial g1-mapreduce
```

---

## 환경 설정

### .env 파일 예시

```bash
# LLM Provider
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://localhost:11434

# 또는 vLLM 사용 시
# LLM_PROVIDER=openai_compat
# OPENAI_COMPAT_BASE_URL=http://localhost:8000/v1
# OPENAI_COMPAT_MODEL=my-model

# LLM 파라미터
TEMPERATURE=0.3
MAX_TOKENS=4096

# 리뷰 설정
REVIEW_DEFAULT_VARIANT=G0-baseline
REVIEW_REPO_PATH=/path/to/target/repo
REVIEW_ALLOWED_VARIANTS_RAW=g0-baseline,g1-mapreduce
```

---

## 실행 방법

### 개발 환경

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### API 테스트

```bash
# 헬스체크
curl http://localhost:8000/health

# 스키마 확인
curl http://localhost:8000/schema/review

# 코드 리뷰 요청 (staged diff)
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"diff_target": "staged"}'

# 직접 diff 전달
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "diff": "diff --git a/file.py b/file.py\n...",
    "variant_id": "g0-baseline"
  }'
```

---

## 응답 스키마

`ReviewResult`의 주요 필드:

```json
{
  "meta": {
    "variant_id": "g0-baseline",
    "run_id": "uuid",
    "repair_used": false,
    "llm_provider": "ollama",
    "model": "qwen3:4b",
    "diff_target": "staged",
    "generated_at": "2024-01-01T00:00:00"
  },
  "summary": {
    "intent": "변경 의도 요약",
    "overall_risk": "low|medium|high",
    "key_points": ["핵심 포인트 1", "핵심 포인트 2"]
  },
  "issues": [
    {
      "id": "ISS-001",
      "title": "이슈 제목",
      "severity": "blocker|high|medium|low",
      "category": "correctness|security|performance|...",
      "description": "이슈 설명",
      "suggested_fix": "수정 제안",
      "locations": [{"file": "path/to/file.py", "line_start": 10, "line_end": 15}]
    }
  ],
  "test_suggestions": [...],
  "questions_to_author": [...],
  "merge_blockers": ["블로커 사유"],
  "patch_suggestions": [...]
}
```

---

## 확장 포인트 정리

| 확장 대상 | 방법 |
|-----------|------|
| 새로운 LLM 백엔드 | `llm/` 아래 `LLMAdapter` 구현체 추가 |
| 새로운 리뷰 전략 | 프롬프트 팩 + 파이프라인 + 프리셋 YAML 추가 |
| 출력 스키마 변경 | `domain/schemas/review.py` 수정 |
| diff 수집 방식 변경 | `domain/tools/git_diff.py` 확장 또는 파이프라인 hook 오버라이드 |
| 후처리 로직 추가 | 파이프라인의 `after_run()` hook 오버라이드 |
| 평가 데이터셋 추가 | `evaluation/datasets/`에 YAML 파일 추가 |
| 평가 지표 추가 | `evaluation/scorer.py`에 새 evaluator 함수 추가 |
