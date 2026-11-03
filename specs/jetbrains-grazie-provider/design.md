# JetBrains AI (Grazie Gateway) Provider Design

## 목표와 범위

AI Model Interop에 **JetBrains AI (Grazie Gateway)** provider를 추가한다. 서비스의 OpenAI-compatible API 표면(`/v1/chat/completions`, `/v1/messages`)을 통해 JetBrains AI가 제공하는 OpenAI, Anthropic (Claude), xAI (Grok), Google (Gemini) 모델을 호출할 수 있게 한다.

### 포함 범위
- JetBrains Grazie Gateway를 새로운 provider adapter로 추가
- 프로바이더별 endpoint 라우팅 (OpenAI/Anthropic/Grok/Google)
- JetBrains OAuth 인증 (JWT 토큰)
- Grazie 라우팅 헤더
- 모델 ID 매핑 (JetBrains 모델명 → Grazie 내부 모델명)
- 비용 추적 (cost gate) 통합
- Anthropic ingress (`/v1/messages`) 지원
- transform 로직은 JetBrains provider 자체에 구현 (외부 provider 코드에 의존하지 않음)

### 제외 범위
- JetBrains OAuth PKCE flow 자체 (bridge 외부에서 토큰 획득, 환경변수로 주입)
- Embeddings / Rerank (JetBrains Grazie는 chat만 지원)
- DeepSeek / Kimi / Muse (Grazie에서 접근 불가 확인됨)

## 배경: JetBrains Grazie Gateway 구조

JetBrains AI API는 `https://ingrazzio-cloud-prod.labs.jb.gg` base URL의 **Grazie Gateway**를 통해 접근한다. 각 프로바이더는 **다른 endpoint 경로**를 사용하며, `X-LLM-Model` 헤더로 라우팅된다.

### 프로바이더별 Endpoint

| 프로바이더 | Endpoint 경로 | 요청 형식 | 응답 형식 |
|---|---|---|---|
| OpenAI | `/v1/responses` | OpenAI Responses API | OpenAI Responses API |
| Anthropic (Claude) | `/v1/messages` | Anthropic Messages API | Anthropic Messages API |
| Grok (xAI) | `/v1/responses` | OpenAI Responses API (Grok 라우팅) | OpenAI Responses API |
| Google (Gemini) | `/v1beta1/projects/jetbrains-grazie/locations/global/publishers/google/models/:generateContent` | Google native generateContent | Google native generateContent |

### 공통 라우팅 헤더

| 헤더 | 값 |
|---|---|
| `Grazie-Agent` | `{"name":"junie:cli","version":" "}` |
| `X-LLM-Model` | `openai` / `anthropic` / `grok` / `google` |
| `X-Keep-Path` | `true` |
| `X-Accept-EAP-License` | `false` |
| `X-Free-Google-Api` | `true` (조건부) |

### 인증

- JetBrains OAuth (PKCE flow) → JWT 토큰 획득
- `Authorization: Bearer <jwt_token>`
- 토큰은 환경변수 `JETBRAINS_AI_TOKEN` (또는 alias별 `api_key_env`)로 주입
- bridge는 토큰 갱신을 하지 않음 (외부에서 관리)

### 모델 ID 매핑

JetBrains 모델 ID와 Grazie 내부 모델 ID가 다름:

| JetBrains 모델 ID | Grazie 내부 모델 ID | 프로바이더 |
|---|---|---|
| `openai-gpt-5-4` | `gpt-5.4` | OpenAI |
| `openai-gpt-5-5` | `gpt-5.5` | OpenAI |
| `grok-4-5` | `grok-4.5` | Grok |
| `claude-sonnet-5` | `claude-sonnet-5` | Anthropic (동일) |
| `gemini-3.1-pro-preview` | `gemini-3.1-pro-preview` | Google (dots 유지) |

## 아키텍처 설계

### 접근 방식: transform 로직을 별도 모듈로 추출하고, Foundry와 JetBrains 양쪽에서 참조

현재 transform 로직은 `foundry.py` 안에 인라인으로 구현되어 있다. 이것을 `core/transforms.py`로 추출하여, Foundry와 JetBrains 양쪽에서 참조한다. 코드 중복을 피하고, Foundry endpoint 접근 여부와 무관하게 동작한다.

```
core/transforms.py (공유 모듈)
  ├── openai_chat_to_responses()   → OpenAI Responses API 변환
  ├── openai_chat_to_anthropic()   → Anthropic Messages API 변환
  └── openai_chat_to_google()      → Google generateContent 변환
       ↑                    ↑
  foundry.py          jetbrains.py
  (참조)              (참조)
```

bridge는 클라이언트로부터 **OpenAI Chat Completions 형식**으로 요청을 받는다. 이를 각 프로바이더의 native 형식으로 변환해야 한다. 변환 로직은 `core/transforms.py`에 구현하고, Foundry와 JetBrains 양쪽에서 참조한다.

```
Client (OpenAI Chat Completions format)
  ↓
Bridge /v1/chat/completions
  ↓
jetbrains provider
  ├── protocol=openai_responses  → core/transforms.py::openai_chat_to_responses()
  ├── protocol=anthropic_messages → core/transforms.py::openai_chat_to_anthropic()
  ├── protocol=xai_responses     → core/transforms.py::openai_chat_to_responses() (Grok 라우팅)
  └── protocol=google_generate_content → core/transforms.py::openai_chat_to_google()
  ↓
Grazie Gateway (https://ingrazzio-cloud-prod.labs.jb.gg)
  ├── /v1/responses (OpenAI/Grok)
  ├── /v1/messages (Anthropic)
  └── /v1beta1/.../generateContent (Google)
```

### 핵심 설계 결정

**1. transform 로직은 `core/transforms.py`에 추출하여 Foundry와 JetBrains 양쪽에서 참조**

현재 transform 로직은 `foundry.py` 안에 인라인으로 구현되어 있다. 이것을 `core/transforms.py`로 추출한다. Foundry와 JetBrains 양쪽에서 이 모듈을 참조한다. 코드 중복을 피하고, Foundry endpoint 접근 여부와 무관하게 동작한다.

추출할 3가지 transform 함수:
- `openai_chat_to_responses()` — OpenAI Chat Completions → OpenAI Responses API 형식 변환
- `openai_chat_to_anthropic()` — OpenAI Chat Completions → Anthropic Messages API 형식 변환
- `openai_chat_to_google()` — OpenAI Chat Completions → Google generateContent 형식 변환

**2. HTTP transport는 JetBrains 전용**

base URL, 인증, 헤더는 JetBrains 전용으로 구현한다.
- Base URL: `JETBRAINS_AI_BASE_URL` (기본값: `https://ingrazzio-cloud-prod.labs.jb.gg`)
- 인증: `JETBRAINS_AI_TOKEN` (JWT)
- 라우팅 헤더: `X-LLM-Model`, `Grazie-Agent` 등

**3. 모델 registry 통합**

JetBrains 모델은 `jetbrains:` prefix를 사용한다:

```json
{
  "jetbrains:openai-gpt-5-4": {
    "provider": "jetbrains",
    "kind": "chat",
    "provider_model": "gpt-5.4",
    "protocol": "openai_responses",
    "llm_model_header": "openai"
  },
  "jetbrains:claude-sonnet-5": {
    "provider": "jetbrains",
    "kind": "chat",
    "provider_model": "claude-sonnet-5",
    "protocol": "anthropic_messages",
    "llm_model_header": "anthropic"
  },
  "jetbrains:grok-4-5": {
    "provider": "jetbrains",
    "kind": "chat",
    "provider_model": "grok-4.5",
    "protocol": "xai_responses",
    "llm_model_header": "grok"
  },
  "jetbrains:gemini-3.1-pro-preview": {
    "provider": "jetbrains",
    "kind": "chat",
    "provider_model": "gemini-3.1-pro-preview",
    "protocol": "google_generate_content",
    "llm_model_header": "google"
  }
}
```

**4. 동적 모델 prefix**

`jetbrains:<native-model>` 동적 prefix를 지원한다 (ollama/compat과 동일 패턴). registry에 등록하지 않아도 `jetbrains:gpt-5.4` 형태로 직접 호출 가능하다.

## 구조적 개선: resolve + select 통합

현재 `_resolve_chat_model`과 `_select_chat_client`가 분리되어 있어 2번의 분기/수렴이 발생한다. 이를 통합하여 단일 흐름으로 만든다.

```
Before:
  model → _resolve_chat_model → config → _select_chat_client → client → generate → transform → response

After:
  model → _resolve_chat_and_client → (provider, client, config) → generate → transform → response
```

**변경 대상**: `main.py`
- `_resolve_chat_model()`과 `_select_chat_client()`를 `_resolve_chat_and_client()`로 통합
- JetBrains provider 추가 시 이 통합을 함께 수행
- 모든 provider (vertex, ollama, foundry, openai_compatible, jetbrains)에 적용

## 이름 변경: openai_compat → openai_compatible

`openai_compat` → `openai_compatible`로 변경한다. 영향 범위:
- `providers/openai_compat.py` → `providers/openai_compatible.py`
- `OpenAICompatChatClient` → `OpenAICompatibleChatClient`
- `provider_adapter.py`: SUPPORTED_PROVIDERS, parse_usage_for_provider, map_http_error_for_provider
- `main.py`: import, lifespan, _select_chat_client, _install_cost_clients
- `anthropic_ingress.py`: provider name 참조
- 테스트 파일: `test_openai_compat.py` → `test_openai_compatible.py`
- README, .env.example, docker-compose.yml

### Phase 0: Transform 로직 추출 (1-2일)

**파일**: `core/transforms.py` (신규)

Foundry에 인라인으로 구현된 transform 로직을 별도 모듈로 추출한다. Foundry와 JetBrains 양쪽에서 참조한다.

- `openai_chat_to_responses()` — OpenAI Chat Completions → OpenAI Responses API
- `openai_chat_to_anthropic()` — OpenAI Chat Completions → Anthropic Messages API
- `openai_chat_to_google()` — OpenAI Chat Completions → Google generateContent

Foundry의 `_xai_input_messages`, `_google_contents_and_system`, `_build_anthropic_request_body` 등 헬퍼 함수를 이 모듈로 이동하고, Foundry는 이 모듈을 import하여 사용하도록 변경한다.

### Phase 1: Provider Adapter 기본 구조

**파일**: `openai_compatible_bridge/providers/jetbrains.py`

```python
JETBRAINS_AI_BASE_URL = os.getenv("JETBRAINS_AI_BASE_URL", "https://ingrazzio-cloud-prod.labs.jb.gg").strip().rstrip("/")
JETBRAINS_AI_TOKEN = os.getenv("JETBRAINS_AI_TOKEN", "").strip()
HTTP_TIMEOUT_SECONDS = float(os.getenv("JETBRAINS_AI_HTTP_TIMEOUT_SECONDS", os.getenv("HTTP_TIMEOUT_SECONDS", "60")))

# 프로토콜 상수
JETBRAINS_OPENAI_PROTOCOL = "openai_responses"
JETBRAINS_ANTHROPIC_PROTOCOL = "anthropic_messages"
JETBRAINS_XAI_PROTOCOL = "xai_responses"
JETBRAINS_GOOGLE_PROTOCOL = "google_generate_content"

# endpoint 경로
JETBRAINS_OPENAI_PATH = "/v1/responses"
JETBRAINS_ANTHROPIC_PATH = "/v1/messages"
JETBRAINS_GOOGLE_PATH_PREFIX = "/v1beta1/projects/jetbrains-grazie/locations/global/publishers/google/models/"

GRAZIE_AGENT_HEADER = '{"name":"junie:cli","version":" "}'
```

**핵심 클래스**: `JetBrainsChatClient`

- `__init__`: httpx.AsyncClient 생성
- `close`: 클라이언트 종료
- `generate`: 비스트리밍 요청
- `stream_chat`: 스트리밍 요청
- 내부적으로 protocol에 따라 `core/transforms.py`의 transform 함수를 호출

**transform 로직은 `core/transforms.py`에서 참조**:

1. **OpenAI Responses API transform** — `core/transforms.py::openai_chat_to_responses()`
   - 입력: OpenAI Chat Completions messages, tools, max_tokens, temperature 등
   - 출력: OpenAI Responses API 형식 (input, tools, reasoning 등)
   - 응답: OpenAI Responses API → OpenAI Chat Completions 형식 변환

2. **Anthropic Messages API transform** — `core/transforms.py::openai_chat_to_anthropic()`
   - 입력: OpenAI Chat Completions messages, tools, max_tokens, temperature 등
   - 출력: Anthropic Messages API 형식 (messages, tools, system, max_tokens 등)
   - 응답: Anthropic Messages API → OpenAI Chat Completions 형식 변환

3. **xAI Responses API transform** — `core/transforms.py::openai_chat_to_responses()` (Grok 라우팅)
   - OpenAI Responses API와 동일한 변환, Grok 라우팅 헤더만 다름

4. **Google generateContent transform** — `core/transforms.py::openai_chat_to_google()`
   - 입력: OpenAI Chat Completions messages, tools, max_tokens, temperature 등
   - 출력: Google generateContent 형식 (contents, systemInstruction, generationConfig, toolConfig)
   - 응답: Google generateContent → OpenAI Chat Completions 형식 변환

### Phase 2: main.py 통합

**변경 사항**:

1. **import 추가**:
```python
from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
```

2. **lifespan에 client 초기화**:
```python
app.state.jetbrains_chat_client = JetBrainsChatClient()
```

3. **_select_chat_client에 분기 추가**:
```python
if provider == "jetbrains":
    return request.app.state.jetbrains_chat_client
```

4. **_install_cost_clients에 jetbrains 추가**:
```python
("jetbrains_chat_client", "jetbrains"),
```

5. **_resolve_chat_model에 jetbrains 동적 prefix 처리**:
```python
if model.startswith("jetbrains:"):
    # compat/ollama와 동일 패턴
```

6. **_main_allowed_models에 jetbrains 모델 추가**:
```python
def _main_allowed_models() -> set[str]:
    return _vertex_allowed_models() | _COMPAT_MODEL_IDS | _JETBRAINS_MODEL_IDS
```

### Phase 3: provider_adapter.py 통합

1. **SUPPORTED_PROVIDERS에 "jetbrains" 추가**
2. **parse_usage_for_provider에 jetbrains 분기 추가** (OpenAI/Anthropic/Google usage 파싱)
3. **PROVIDER_ENDPOINTS에 "jetbrains": ("chat",) 추가**
4. **map_http_error_for_provider에 jetbrains 분기 추가**

### Phase 4: Anthropic ingress 통합

`/v1/messages` 엔드포인트에서 `jetbrains` provider를 지원하도록 추가:
- `PROVIDER_TOOL_SUPPORT`에 `"jetbrains": True` 추가
- anthropic_ingress.py에서 jetbrains provider 처리

### Phase 5: 비용 추적 통합

- `COST_PROVIDER_BILLING_JSON`에 `"jetbrains"` 키 추가
- `MeteredHTTPClient`에 jetbrains provider 등록
- 가격 fallback: `jetbrains:*` 동적 모델 지원

### Phase 6: 테스트

- `tests/test_jetbrains_chat.py`
- 각 프로바이더별 transform 검증
- 라우팅 헤더 검증
- 모델 ID 매핑 검증

### Phase 7: 문서화

- README 업데이트
- .env.example 업데이트
- docker-compose.yml 업데이트

## 인증 설계 상세

### 토큰 주입 방식

Option A: 환경변수 직접 주입
```bash
JETBRAINS_AI_TOKEN=<jwt_token>
```

Option B: alias별 api_key_env (registry)
```json
{
  "jetbrains:openai-gpt-5-4": {
    "provider": "jetbrains",
    "api_key_env": "JETBRAINS_AI_TOKEN",
    ...
  }
}
```

**권장**: Option B (기존 openai_compat 패턴과 일관성 유지). `api_key_env`가 없으면 `JETBRAINS_AI_TOKEN` 환경변수를 fallback으로 사용.

### 토큰 갱신

bridge는 토큰 갱신을 하지 않는다. 토큰 만료 시:
- 401 응답 → client에 에러 전달
- 운영자가 외부에서 토큰을 갱신하고 환경변수를 업데이트
- bridge 재시작 또는 hot-reload로 반영

## 비용 추적 설계

### provider billing 분류

JetBrains AI는 구독형 서비스이므로, 기본적으로 `subscription` 또는 `nonbillable`로 분류한다. 운영자가 실제 계약을 확인하여 설정한다.

```bash
COST_PROVIDER_BILLING_JSON='{"jetbrains":"subscription"}'
```

### 가격 fallback

동적 모델 `jetbrains:<native>`에 대한 가격 fallback:
- `jetbrains:*` exact key로 fallback (ollama/compat과 동일 패턴)

## 리스크와 제약

### 1. Transform 로직 추출 복잡도

`foundry.py`에 인라인으로 구현된 transform 로직을 `core/transforms.py`로 추출하는 과정에서, Foundry provider의 기존 동작을 깨뜨리지 않아야 한다. 추출 후 Foundry provider가 이 모듈을 참조하도록 변경하고, 기존 테스트가 모두 통과해야 한다.

완화 방법:
- 추출 전 Foundry 테스트를 먼저 실행하여 baseline 확보
- 추출 후 Foundry 테스트를 재실행하여 회귀 확인
- 점진적으로 추출 (한 번에 하나의 transform 함수씩)

### 2. JetBrains OAuth 토큰 만료

JWT 토큰은 만료된다. bridge가 토큰 갱신을 하지 않으므로, 토큰 만료 시 수동 갱신이 필요하다. 향후 자동 갱신 기능을 추가할 수 있다.

### 3. Grazie Gateway 변경

JetBrains 내부 API이므로, endpoint 경로나 헤더가 변경될 수 있다. pi-jetbrains-junie-bridge 프로젝트의 JUNIE_INSIGHTS.md를 참고하여 변경 사항을 추적한다.

### 4. Google 모델의 Vertex 스타일 경로

Google 모델은 Vertex 스타일 경로(`/v1beta1/projects/.../generateContent`)를 사용한다. project ID가 `jetbrains-grazie`로 고정되어 있다. 환경변수로 override 가능하게 설계한다.

## 마일스톤

### M1: Transform 로직 추출 (1-2일)
- `core/transforms.py` 생성
- Foundry의 transform 헬퍼 함수를 이 모듈로 이동
- Foundry provider가 이 모듈을 import하도록 변경
- Foundry 테스트로 회귀 확인

### M2: Provider Adapter 기본 구조 (2-3일)
- `providers/jetbrains.py` 생성
- JetBrainsChatClient 클래스 구현
- `core/transforms.py` 참조
- Grazie 라우팅 헤더

### M3: main.py 통합 (1일)
- lifespan client 초기화
- _select_chat_client 분기
- _resolve_chat_model 동적 prefix
- _main_allowed_models

### M4: provider_adapter.py 통합 (0.5일)
- SUPPORTED_PROVIDERS 추가
- parse_usage_for_provider 분기
- PROVIDER_ENDPOINTS 추가
- map_http_error_for_provider 분기

### M5: Anthropic ingress 통합 (0.5일)
- PROVIDER_TOOL_SUPPORT 추가
- anthropic ingress에서 jetbrains 처리

### M6: 비용 추적 통합 (0.5일)
- MeteredHTTPClient 등록
- 가격 fallback

### M7: 테스트 (2-3일)
- tests/test_jetbrains_chat.py
- 각 프로바이더별 transform 검증
- 라우팅 헤더 검증
- 모델 ID 매핑 검증

### M8: 문서화 (0.5일)
- README 업데이트
- .env.example 업데이트
- docker-compose.yml 업데이트

## 예상 파일 변경 목록

| 파일 | 변경 유형 | 설명 |
|---|---|---|
| `core/transforms.py` | 신규 | 공유 transform 모듈 (OpenAI/Anthropic/Google 변환) |
| `providers/foundry.py` | 수정 | transform 로직을 core/transforms.py에서 import하도록 변경 |
| `providers/jetbrains.py` | 신규 | JetBrains provider adapter (core/transforms.py 참조) |
| `main.py` | 수정 | client 초기화, 라우팅, 동적 prefix |
| `core/provider_adapter.py` | 수정 | SUPPORTED_PROVIDERS, usage parser, error mapper |
| `anthropic_ingress.py` | 수정 | PROVIDER_TOOL_SUPPORT 추가 |
| `tests/test_jetbrains_chat.py` | 신규 | 테스트 |
| `README.md` | 수정 | 문서화 |
| `.env.example` | 수정 | 환경변수 추가 |
| `docker-compose.yml` | 수정 | 환경변수 추가 |

## 열린 질문

1. **Grazie Gateway의 Google 모델 project ID가 항상 `jetbrains-grazie`인가?**
   - pi-jetbrains-junie-bridge에서 확인됨
   - 운영 환경에서 변경 가능성 있음 → 환경변수로 override 가능하게 설계

2. **Transform 로직 추출 시 Foundry provider의 기존 동작을 깨뜨리지 않을 수 있는가?**
   - 추출 전 Foundry 테스트 baseline 확보 필요
   - 점진적 추출 (한 번에 하나의 함수씩)로 위험 최소화
