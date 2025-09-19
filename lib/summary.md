# `lib` Folder Summary

이 폴더는 `code-review-bot` 애플리케이션의 핵심 로직과 기능을 구현한 라이브러리 파일들을 포함하고 있습니다. 각 파일은 특정 도메인(Git, Ollama 클라이언트, 리뷰 로직)에 대한 책임을 가집니다.

---

## File: `git.dart`

### Purpose

Git 관련 커맨드를 실행하고 그 결과를 반환하는 기능을 담당합니다. 현재는 스테이징된(staged) 파일들의 diff 정보를 가져오는 데 사용됩니다.

### Key Components

- **`Git` class**:
  - **`static Future<String> getStagedDiff()`**: `git diff --cached` (또는 `--staged`) 명령을 실행하여 스테이징된 변경사항에 대한 diff 텍스트를 비동기적으로 가져옵니다. Git 프로세스 실행에 실패하면 `ProcessException`을 던집니다.

---

## File: `ollama_client.dart`

### Purpose

Ollama API와 통신하는 HTTP 클라이언트입니다. 주어진 모델과 프롬프트를 사용하여 코드 리뷰 생성을 요청하고, 응답을 받아 파싱하는 역할을 합니다.

### Key Components

- **`OllamaClient` class**:
  - **Constructor**: `host`와 `port`를 인자로 받아 Ollama 서버의 주소를 설정합니다.
  - **`Future<String> generate(...)`**: Ollama의 `/api/generate` 엔드포인트에 요청을 보냅니다. 스트리밍이 아닌 단일 응답을 요청하며, 응답 본문의 `response` 필드를 반환합니다. (주로 `chat` API의 폴백으로 사용됩니다.)
  - **`Future<String> chat(...)`**: Ollama의 `/api/chat` 엔드포인트에 요청을 보냅니다. `system` 역할과 `user` 역할의 메시지를 함께 전송하여 더 정교한 대화형 응답을 유도합니다. 응답 본문의 `message.content` 필드를 우선적으로 파싱하여 반환합니다.

---

## File: `reviewer.dart`

### Purpose

코드 리뷰 프로세스의 핵심 로직을 관장하는 클래스입니다. Git diff를 가져와 Ollama API에 보낼 프롬프트를 구성하고, 리뷰 결과를 받아 파싱하며, 최종 출력과 커밋 차단 여부를 결정합니다. 이 클래스는 `constants.dart`에 정의된 상수들을 적극적으로 활용하여 코드의 일관성과 유지보수성을 높입니다.

### Key Components

- **`Reviewer` class**:
  - **Constructor**: `model`, `host`, `port` 등 Ollama 클라이언트 설정과 `failOnSeverity`, `strict` 등 리뷰 정책 설정을 인자로 받습니다.
  - **`Future<ReviewResult> reviewStagedChanges()`**: 전체 리뷰 워크플로우를 실행합니다.
    1. `Git.getStagedDiff()`를 호출하여 diff를 가져옵니다.
    2. diff가 너무 길면 `maxChars`에 맞춰 자릅니다.
    3. `_buildPrompt()`를 호출하여 Ollama에 보낼 프롬프트를 생성합니다.
    4. `OllamaClient`를 사용하여 `chat` API를 호출하고, 실패 시 `generate` API로 폴백합니다.
    5. `_parseResponse()`를 호출하여 LLM의 응답에서 리뷰 마크다운과 JSON 요약 정보를 추출합니다.
    6. 최종 결과(`ReviewResult`)를 생성하여 반환합니다. 이 결과에는 터미널 출력용 문자열, 탐지된 최대 심각도, 커밋 차단 여부가 포함됩니다.
  - **`String _buildPrompt(String diff)`**: LLM에게 원하는 결과물(마크다운 리뷰, JSON 요약)을 명확히 지시하는 프롬프트를 생성합니다. `constants.dart`의 `reviewPromptTemplate`을 사용합니다.
  - **`_ParsedResponse _parseResponse(String response)`**: LLM의 응답 문자열에서 `jsonBlockRegExp` 정규식을 사용하여 JSON 블록을 찾아내고, 이를 바탕으로 `max_severity`와 순수 리뷰 마크다운 텍스트를 분리합니다.
  - **`static Future<void> installHook()`**: 현재 Git 저장소의 `.git/hooks/` 디렉토리에 `pre-commit` 훅 스크립트를 설치합니다. `preCommitHookScript` 상수를 사용합니다.

- **`ReviewResult` class**:
  - 리뷰 결과를 담는 데이터 클래스입니다. `renderedOutput`, `maxSeverity`, `blockCommit` 필드를 가집니다.

---

## File: `constants.dart`

### Purpose

애플리케이션 전반에서 사용되는 상수들을 관리합니다. 코드의 일관성을 유지하고 수정을 용이하게 합니다.

### Key Components

- **`severityOrder`**: 리뷰의 심각도 순서를 정의한 리스트입니다. (`['low', 'medium', 'high', 'block']`)
- **`reviewPromptTemplate`**: Ollama API에 전달할 프롬프트의 템플릿입니다.
- **`preCommitHookScript`**: `pre-commit` Git 훅에 사용될 셸 스크립트 내용입니다.
- **`jsonBlockRegExp`**: LLM의 응답에서 JSON 블록을 추출하기 위한 정규식입니다.
- **`noStagedChangesMessage`**: 스테이징된 변경사항이 없을 때 사용자에게 표시할 메시지입니다.
- **`defaultSeverity`**: 파싱 실패 또는 값 부재 시 사용될 기본 심각도입니다.
- **`reviewHeader`, `reviewSubHeader`, `maxSeverityDetected`, `lineSeparator`**: 리뷰 결과 출력의 헤더 및 구분선 문자열들입니다.
