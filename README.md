<a id="top"></a>

<div align="center">
  <img src="./assets/header.svg" width="100%" alt="AI Model Interop — multi-protocol AI model API gateway, provider-native routing, private / local-only"/>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Vertex_AI-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white" alt="Vertex AI"/>
  <img src="https://img.shields.io/badge/Ollama-111827?style=for-the-badge&logoColor=white" alt="Ollama"/>
  <img src="https://img.shields.io/badge/Palantir_Foundry-000000?style=for-the-badge&logo=palantir&logoColor=white" alt="Palantir Foundry"/>
</p>

<div align="center">
  <h3><b>AI Model Interop</b>은 OpenAI-compatible API와 Anthropic Messages 요청을 Vertex AI, Ollama, Palantir Foundry, JetBrains AI 등의 provider-native AI model API로 변환·라우팅하는 <b>로컬/사내망 전용 multi-protocol gateway</b>입니다.</h3>
</div>

> [!WARNING]
> **배포 위치 주의**: 이 bridge는 Vertex service account, local model endpoint, cost ledger 같은 운영 자원을 연결합니다. 보안과 과금 보호를 위해 **public internet에 노출하지 말고**, local machine 또는 private Docker network 안에서만 운용하십시오.

<p align="center">
  <a href="#arch"><b>🏛️ 시스템 아키텍처</b></a> &nbsp;·&nbsp;
  <a href="#quickstart"><b>🚀 빠른 시작</b></a> &nbsp;·&nbsp;
  <a href="#config"><b>⚙️ 환경 변수</b></a> &nbsp;·&nbsp;
  <a href="#api"><b>📡 API 참조</b></a>
</p>

---

## 💡 개발 배경 (Why?)

OpenAI-compatible client가 여러 provider를 직접 다루게 만들면 인증, payload, streaming, batching, 비용 추적 책임이 client마다 흩어집니다. 이 bridge는 client에는 `/v1/...` endpoint와 `model` alias만 노출하고, 내부에서 provider-native API 호출을 분기합니다.

* **Provider routing 단일화**: client 요청에는 provider field를 넣지 않습니다. `model` alias가 registry를 통해 Vertex, Ollama, Foundry provider adapter로 해석됩니다.
* **Vertex 운영 복잡도 흡수**: Vertex service account 인증, model별 batching, embeddings/rerank/chat payload 변환을 bridge가 담당합니다.
* **Local model 연결**: Ollama chat completions를 같은 `/v1/chat/completions` 표면으로 연결해 local model과 Vertex model을 같은 client 설정에서 다룰 수 있습니다.
* **Foundry 이기종 프로토콜 & 멀티턴 Tool Calling 완결**: Palantir Foundry의 5종 upstream 규격(OpenAI, Claude, Grok, Astra, Google Gemini)을 단일 규격으로 표준화하고 multi-turn structured tool call / tool result 사이클과 스트리밍을 투명하게 중계합니다.
* **비용 방어선 유지**: cost tracking을 켜면 `metered`로 분류된 모든 실제 provider HTTP 시도가 전송 직전에 forecast budget gate를 통과해야 합니다. 실제 청구액의 절대 상한을 보장하지는 않습니다.

---

<a id="arch"></a>

## 🏛️ 시스템 아키텍처

<div align="center">
  <img src="./assets/architecture.svg" width="100%" alt="AI Model Interop architecture"/>
</div>

### 🎨 핵심 설계 포인트

<table width="100%">
  <tr>
    <td width="50%" valign="top">

#### 🟦 Drop-in API Shape
<p>OpenAI-compatible client는 기존처럼 <code>/v1/embeddings</code>, <code>/v1/chat/completions</code>, <code>/v1/rerank</code>를 호출합니다.</p>
    </td>
    <td width="50%" valign="top">

#### 🟩 Provider Adapter
<p>Bridge 내부에서 model alias를 Vertex, Ollama 또는 Palantir Foundry provider-native model id로 해석합니다.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">

#### 🟪 Vertex Native Support
<p>Vertex embeddings, chat completions, Search Ranking rerank를 OpenAI-compatible response shape으로 변환합니다.</p>
    </td>
    <td width="50%" valign="top">

#### 🟧 Private Cost Gate
<p>Cost tracking이 켜진 경우 과금 분류가 불명확하거나 종량제 모델의 가격 설정이 불완전하면 fail-closed 처리합니다. PostgreSQL 공유 원장에서 forecast 예산 검사와 예약을 원자적으로 수행한 뒤에만 유료 호출을 전송합니다.</p>
    </td>
  </tr>
</table>

---

## 🧭 지원 범위

| Endpoint | Provider | 세부 프로토콜 / 기능 | 상태 |
|---|---|---|---|
| `POST /v1/embeddings` | Vertex | Google predict embeddings (`text-embedding-005` 등) | 지원 |
| `POST /v1/chat/completions` | Vertex | Gemini generateContent / Gemma MaaS | 지원 |
| `POST /v1/chat/completions` | Ollama | Local chat / dynamic `ollama:<model>` / JSON repair | 지원 |
| `POST /v1/chat/completions` | Foundry | `openai_chat_completions` (OpenAI proxy) | 지원 |
| `POST /v1/chat/completions` | Foundry | `anthropic_messages` (Claude proxy) | 지원 |
| `POST /v1/chat/completions` | Foundry | `xai_responses` (Grok proxy) | 지원 |
| `POST /v1/chat/completions` | Foundry | `openai_responses` (GPT-6 Astra proxy) | 지원 |
| `POST /v1/chat/completions` | Foundry | `google_generate_content` (Google Gemini native proxy) | 지원 |
| `POST /v1/messages` | Bridge | Anthropic Messages ingress → 내부 OpenAI-shape 파이프라인으로 라우팅 (Claude Code / Claude Desktop용) | 지원 |
| `GET /v1/models` | Bridge | `context_length`·`capabilities` 포함 (Desktop 3P validator 대응) | 지원 |
| `POST /v1/rerank` | Vertex | Vertex AI Search Ranking rerank | 지원 |

Ollama embeddings와 Ollama rerank는 현재 범위가 아닙니다. Vertex 임베딩의 429/503에는 아래에 설명한 제한된 재시도를 적용하며, 다른 endpoint의 재시도 정책은 변경하지 않습니다.

### Foundry 이기종 프로토콜 및 Structured Tool Calling

Palantir Foundry의 이기종 upstream LLM Proxy를 단일 OpenAI 규격(`/v1/chat/completions`)으로 통합하며, 멀티턴 도구 호출(Tool Calling)과 스트리밍을 중계합니다.

* **Foundry protocol route**:
  - `openai_chat_completions`, `openai_responses`, `anthropic_messages`, `xai_responses`는 각 Foundry compatible proxy로 라우팅합니다.
  - `google_generate_content`는 OpenAI Chat Completions 요청을 Google native `generateContent`/`streamGenerateContent?alt=sse` 요청으로 변환합니다. Google native 경로에서 지원 여부가 확인되지 않은 reasoning override는 전달하지 않고 명시적으로 거부합니다.
  - 이 protocol은 Enrollment나 OpenAI proxy 노출을 자동으로 변경하지 않습니다. Foundry Enrollment가 활성화되어 있고 실제 Google native model 호출이 확인된 뒤, `MODEL_REGISTRY_JSON`에 명시적인 alias를 추가해야 합니다.


```mermaid
sequenceDiagram
    autonumber
    actor Client as OpenAI-compatible Client
    participant Bridge as AI Model Interop
    participant Foundry as Palantir Foundry LLM Proxy

    Client->>Bridge: POST /v1/chat/completions (tools, tool_choice)
    Note over Bridge: Model Registry 조회 및 protocol 라우팅

    alt protocol = openai_chat_completions
        Bridge->>Foundry: /api/v2/llm/proxy/openai/v1/chat/completions
    else protocol = anthropic_messages (Claude)
        Note over Bridge: tools ↔ input_schema, multi-tool results 병합
        Bridge->>Foundry: /api/v2/llm/proxy/anthropic/v1/messages
        Foundry-->>Bridge: tool_use 응답
        Note over Bridge: tool_use ↔ OpenAI tool_calls 변환
    else protocol = xai_responses (Grok)
        Note over Bridge: reasoning_effort clamp (low/medium/high)
        Bridge->>Foundry: /api/v2/llm/proxy/xai/v1/responses
        Foundry-->>Bridge: function_call 응답 (output_item.done 파싱)
    else protocol = openai_responses (GPT-6 Astra)
        Note over Bridge: /v1/responses 포맷 정규화 (tools 보존, sampling 배제)
        Bridge->>Foundry: /api/v2/llm/proxy/openai/v1/responses
    else protocol = google_generate_content (Gemini)
        Note over Bridge: OpenAI messages/tools → Google contents/toolConfig 변환
        Bridge->>Foundry: /api/v2/llm/proxy/google/v1/models/{model}:generateContent
        Bridge->>Foundry: /api/v2/llm/proxy/google/v1/models/{model}:streamGenerateContent?alt=sse
    end

    Bridge-->>Client: 표준 OpenAI chat.completion 응답 (finish_reason="tool_calls")
```

1. **5종 이기종 프로토콜 자동 변환**:
   - `openai_chat_completions` (기본값): 표준 OpenAI 규격 프록시. `tools`, `tool_choice`, `parallel_tool_calls`를 있는 그대로 전달합니다.
   - `anthropic_messages`: Claude 모델군을 위한 변환 어댑터. OpenAI `tools`를 Anthropic `input_schema`로 변환하고, 모델의 `tool_use` 블록을 OpenAI 표준 `tool_calls` 배열로 변환합니다. 여러 도구 실행 결과(`tool_result`)가 연속 전달되면 Anthropic 규격에 맞춰 단일 `user` 턴 내 블록들로 안전하게 병합합니다.
   - `xai_responses`: Grok 모델군을 위한 Responses 프록시. OpenAI 도구 정의를 xAI `function_call` 구조로 변환하며, Foundry SSE 스트림의 `output_item.done` 이벤트에서 도구 호출 메타데이터를 정밀 파싱합니다. xAI 모델의 `reasoning_effort`는 `low`, `medium`, `high` 3단계로 엄격히 clamp됩니다.
   - `openai_responses`: GPT-6 Astra 계열 등 `/v1/responses`를 요구하는 모델 전용 프로토콜. `/v1/chat/completions`에서 도구 호출 시 거부되는 upstream 동작을 방지하며, sampling/reasoning 파라미터를 배제한 검증된 responses payload 규격으로 호출합니다.
   - `google_generate_content`: Gemini 모델군을 위한 Google native adapter. `messages`/`tools`/생성 파라미터를 `contents`/`systemInstruction`/`generationConfig`/`toolConfig`로 변환하고 `candidates`/`usageMetadata`/Google SSE를 OpenAI 응답으로 되돌립니다. Foundry Google proxy에서 검증되지 않은 reasoning override는 전달하지 않습니다.
2. **Streaming & Non-Streaming 완결성**:
   - 모든 프로토콜에서 SSE 토큰 스트리밍 중 실시간 `tool_calls` 델타(`index`, `id`, `name`, `arguments`)를 표준 규격으로 스트리밍하며 마지막에 `finish_reason: "tool_calls"`를 보장합니다.

### Ollama 특화 기능

Ollama chat adapter는 OpenAI-compatible `response_format.type=json_object`를 Ollama JSON mode(`format: "json"`)로 전달하고, `response_format.type=json_schema`는 wrapper의 `name`/`strict`를 제외한 `json_schema.schema` object만 Ollama native structured output `format`으로 전달합니다. `json_schema` 응답은 bridge가 post-validation하며, Ollama/backend가 schema와 다른 JSON을 반환하면 raw content 없이 non-stream은 HTTP 502, stream은 SSE error의 `invalid_schema_output`으로 실패시킵니다. `STRUCTURED_OUTPUT_REPAIR_ENABLED=true`이면 dynamic Ollama Cloud `json_schema` 실패에 한해 지정된 Ollama Cloud 모델을 순서대로 최대 3회 추가 호출하고, 최종 validation을 통과한 JSON만 200으로 반환합니다. Ollama `think`는 기본 활성화 상태로 호출하며, 요청별 `reasoning_effort` 또는 `reasoning.effort`가 있으면 해당 요청에만 `high`, `medium`, `low`, `none`으로 override합니다. 응답의 `<think>...</think>` reasoning block은 OpenAI-compatible `message.content`와 streaming `delta.content`에서 제거합니다. Reasoning 제거 후 visible content가 비면 model-side upstream error로 처리합니다.

### 호출별 Ollama 모델 지정

Ollama chat model은 registry/env 추가 없이 요청마다 native model을 직접 지정할 수 있습니다. `model` 값을 `ollama:<native-model>` 형태로 보내면 bridge가 Ollama provider로 라우팅하고, prefix를 제거한 값을 Ollama native model id로 전달합니다.

```json
{
  "model": "ollama:deepseek-v4-flash:cloud",
  "messages": [
    {"role": "user", "content": "Reply with OK only."}
  ],
  "reasoning_effort": "medium",
  "max_tokens": 64,
  "temperature": 0
}
```

`ollama:`처럼 native model이 비어 있으면 HTTP 400 `invalid_model`로 거절합니다. `/v1/models`는 dynamic Ollama namespace를 열거하지 않고 registry에 등록된 모델만 반환합니다.
`reasoning_effort` 또는 `reasoning.effort`는 요청별 Ollama `think` override입니다. 허용값은 `high`, `medium`, `low`, `none`이며, `none`은 명시 요청일 때만 `think=false`로 전달됩니다. 필드가 없으면 `OLLAMA_THINK` env default를 그대로 사용하므로 runtime 재기동 없이 canary별 reasoning level을 바꿀 수 있습니다.

비용 추적이 켜져 있고 `COST_PROVIDER_BILLING_JSON`에서 `ollama`를 `metered`로 분류했다면 dynamic model에도 user-facing model id 기준의 input/output 가격이 필수입니다. Exact key가 우선이며, `COST_PRICING_JSON`에 `ollama:*` chat 가격을 명시하면 `ollama:<native-model>` 전체에 fallback으로 적용됩니다. 유효한 `subscription`/`nonbillable` 분류는 비용 경로를 완전히 우회하며, Ollama라는 이름만으로 구독형이라고 가정하지 않습니다.

### Ollama structured output repair

Ollama Cloud는 strict structured output을 보장하지 않으므로 repair는 기본 OFF입니다. `STRUCTURED_OUTPUT_REPAIR_ENABLED=true`를 설정하면 dynamic `ollama:<native>` Cloud 모델의 `response_format.type=json_schema` 실패에 한해 repair chain을 실행합니다. 기본 repair chain은 다음 순서입니다.

1. `ollama:qwen3.5:cloud`
2. `ollama:gemma4:31b-cloud`
3. `ollama:glm-5.2:cloud`

`STRUCTURED_OUTPUT_REPAIR_MODELS`로 쉼표 구분 목록을 지정할 수 있지만 최대 3개만 사용합니다. Repair prompt에는 실패한 raw output을 넣지 않고, original messages, JSON schema, redacted failure category만 사용합니다. `stream=true`와 `json_schema` repair가 함께 켜지면 bridge는 token-by-token streaming 대신 buffered validation을 수행하고 valid final JSON만 SSE content chunk로 내보냅니다. 모든 attempt가 실패하면 raw content 없이 `invalid_schema_output`으로 실패합니다.

### Anthropic ingress (`POST /v1/messages`)

Claude Code / Claude Desktop 연결용 Anthropic Messages ingress다. 요청은 `openai_compatible_bridge/anthropic_ingress.py`에서 OpenAI-shape으로 번역된 뒤 기존 registry 라우팅 + forecast budget gate를 그대로 탄다. upstream은 바뀌지 않는다.

* 인증: `Authorization: Bearer <BRIDGE_API_KEY>` 또는 `x-api-key: <BRIDGE_API_KEY>`. 키가 비어 있으면 강제하지 않는다 (기존 ingress와 동일).
* `system`(string|list), `tools`(`input_schema`→JSON Schema), `tool_choice`(`auto`/`any`/`tool`/`none`), `stop_sequences`, `temperature`/`top_p`, `stream` 지원. Claude Code의 `output_config.effort`(`low`/`medium`/`high`/`xhigh`/`max`)는 `openai_compat` provider에 `reasoning_effort`로 전달한다. 지원 단계는 선택한 OpenCode 모델과 해당 OpenAI-compatible endpoint에 따라 다르다. `thinking` 대화 이력 블록은 무시하고, image/document 등 미디어 블록은 400으로 거부한다.
* tool 호출 가능 provider: Vertex, Foundry(`PROVIDER_TOOL_SUPPORT`). Ollama는 `tools` 포함 시 400. Vertex function calling(`functionDeclarations`/`functionResponse`, id 합성 `call_<name>_<idx>`)은 `providers/vertex.py`가 담당하고, tool 포함 스트림은 buffered 단일 이벤트로 반환한다.
* 스트리밍은 Anthropic SSE(`message_start`/`content_block_delta`/`message_delta`/`message_stop`)이며 `[DONE]`을 보내지 않는다. `message_start`의 usage는 0으로 시작하고 최종 `message_delta`에 실제값을 싣는다.
* `count_tokens`는 제공하지 않는다 (없으면 client가 추정으로 fallback).
* `GET /v1/models` 각 항목에 `context_length`·`capabilities{tools,streaming,images}`를 포함한다. `context_length`는 registry alias의 `context_length` 키로 override, 기본값 `128000`. 정확한 값은 운영자가 기입한다.

Claude Code 연결 예 (`/v1`을 붙이지 않는다. Claude Code가 `/v1/messages`를 붙인다):

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=$BRIDGE_API_KEY
claude --model gemini-2.5-flash
```

`~/.claude/settings.json` 별명 매핑 예:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    "ANTHROPIC_AUTH_TOKEN": "bridge-key",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "gemini-2.5-flash",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gemini-2.5-flash"
  }
}
```

Claude Desktop은 Developer Mode → Configure Third-Party Inference에 gateway base URL로 이 bridge를 지정한다. 비-Claude 모델은 `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`이 필요할 수 있다.

### OpenCode Go 등 외부 OpenAI gateway 연결 (`openai_compatible`)

`openai_compatible` provider는 alias마다 외부 OpenAI-compatible `/chat/completions`를 바라본다. Claude Code → bridge `/v1/messages` → Go openai endpoint 구조로 Go 전 모델을 매핑할 수 있다. client가 URL/key를 지정할 수 없고 registry(env)만 본다.

```json
{
  "go-glm-5.3": {
    "provider": "openai_compatible",
    "kind": "chat",
    "provider_model": "glm-5.3",
    "base_url": "https://opencode.ai/inference/go/openai/v1",
    "api_key_env": "OPENCODE_GO_API_KEY"
  }
}
```

* `base_url`: 필수. `http(s)`만 허용. `api_key_env`가 가리키는 환경변수를 호출 시점에 읽는다 (재시작 없이 로테이션). `api_key` 직접 기재도 되지만 권장하지 않는다.
* 모델 하나하나 등록하지 않아도 된다. `compat:<upstream-model>` dynamic namespace는 레지스트리 추가 없이 `OPENAI_COMPAT_BASE_URL` 하나로 전 모델을 커버한다 (`ollama:<native>`와 같은 방식). 예: `claude --model compat:glm-5.3`. `/v1/models` 목록에는 dynamic 모델이 나오지 않는다. 키는 `OPENAI_COMPAT_API_KEY`로 주입한다.
* 과금: `COST_PROVIDER_BILLING_JSON`에 `"openai_compatible"` 키를 실제 계약(`metered` 등)으로 분류한다. 종량이면 `compat:*` chat 가격(`input_per_million`/`output_per_million`) 하나로 dynamic 전 모델을 커버할 수 있다. 키 누락은 fail-closed.
* Claude Code 매핑 예: `ANTHROPIC_DEFAULT_SONNET_MODEL=go-glm-5.3` 또는 `claude --model go-glm-5.3`. Anthropic 네이티브 Go 모델(`minimax-m3`, `qwen` 계열 등)은 Go anthropic endpoint에 직접 붙는 게 낫다.

### JetBrains AI (Grazie Gateway) 연결

JetBrains AI는 Grazie Gateway를 통해 OpenAI, Anthropic (Claude), xAI (Grok), Google (Gemini) 모델을 제공한다. 각 프로바이더는 다른 endpoint를 사용하며, `X-LLM-Model` 헤더로 라우팅된다.

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
  }
}
```

* 인증: `JETBRAINS_AI_TOKEN`으로 정적 토큰을 주입하거나 별도 OAuth grant를 발급해 `JETBRAINS_AI_OAUTH_FILE`로 갱신 저장소를 지정한다. 독립 grant 발급과 기존 Junie 세션에 대한 영향은 검증되지 않았으며, [로그인 및 보안 제약](specs/jetbrains-grazie-provider/oauth-refresh.md)을 확인해야 한다.
* Base URL: `JETBRAINS_AI_BASE_URL` (기본값: `https://ingrazzio-cloud-prod.labs.jb.gg`)
* 동적 모델: `jetbrains:<native-model>` prefix로 registry 없이 직접 호출 가능 (예: `jetbrains:gpt-5.4`)
* 과금: `COST_PROVIDER_BILLING_JSON`에 `"jetbrains"` 키를 실제 계약(`subscription` 등)으로 분류한다.
* reasoning override: JetBrains의 네 가지 프로토콜은 아직 검증된 변환 계약이 없으므로 OpenAI의 `reasoning_effort`/`reasoning` 및 Anthropic의 `output_config.effort` 요청을 스트리밍 시작과 upstream 호출 전에 HTTP 400으로 거부한다. 미지정 요청은 그대로 처리한다.

#### Junie ACP 모델 탐색과 OMP 연결

브리지를 Junie가 설치되고 로그인된 사용자 환경에서 실행하고 `JUNIE_DISCOVERY_ENABLED=true`를 설정한다. 선택적으로 `JUNIE_DISCOVERY_COMMAND`에 실행 파일 경로, `JUNIE_DISCOVERY_CWD`에 Junie 세션 작업 디렉터리를 지정한다. 브리지는 모델 목록이나 JetBrains 모델 요청 시 `junie --acp=true --skip-update-check`로 `initialize`와 `session/new`만 수행하며 추론 요청은 보내지 않는다. 탐색은 최대 10초, 성공 결과는 5분 동안 캐시한다. 실패하면 마지막 성공 목록을 유지하고 30초 후 재시도한다. `/v1/models`의 `X-Junie-Discovery-Status`는 `disabled`, `fresh`, `stale` 중 하나다. 첫 탐색이 실패했다면 stale 목록은 비어 있다.

ACP의 `model` 옵션에서 `jetbrains-ai` 그룹의 `v1:12:jetbrains-ai:<native>` 항목 중 안전한 ID를 모두 `jetbrains:<native>`로 노출한다. `claude-`, `gemini-`, `grok-`, `gpt-` 및 `openai-gpt-` 계열은 각 프로토콜로 연결하며 upstream에도 변환 없이 원래 native ID를 전달한다. 그 밖의 발견 모델은 목록에서 확인할 수 있지만, 프로토콜을 추측하지 않고 채팅 요청을 `unsupported_model` 오류로 거부한다. 명시적 registry로 정확한 `provider_model`과 `protocol`을 지정하면 탐색 결과보다 우선한다. ACP 목록은 실제 Grazie 모델 ID 호환성이나 호출 성공을 보증하지 않으며 upstream 오류는 기존 오류 처리 경로로 전달된다. ACP는 context 크기와 출력 한도를 보장하지 않으므로 발견 항목에 이를 광고하지 않는다.

탐색되지 않은 `jetbrains:<native>` 직접 호출은 기존 동작대로 기본 OpenAI 프로토콜을 사용한다. 이 동작은 임의 ID에 대한 호환성 보장이 아니며, 다른 프로토콜이 필요하면 registry에 명시해야 한다.

모델 발견 기능은 Junie 프로세스를 통해 목록만 조회하며 자격 증명을 직접 읽지 않는다. 기존 Grazie 채팅 경로에는 별도의 인증이 필요하다. 인증·모델 사용 권한과 upstream ID는 실제 호출로 별도 확인해야 한다.
로컬 실행 예시:

```sh
mkdir -m 700 -p "$HOME/.local/share/ai-model-interop"
uv run python -m openai_compatible_bridge.jetbrains_login --output "$HOME/.local/share/ai-model-interop/jetbrains-oauth.json"
export JETBRAINS_AI_OAUTH_FILE="$HOME/.local/share/ai-model-interop/jetbrains-oauth.json"
export JUNIE_DISCOVERY_ENABLED=true
export MODEL_LIST_INCLUDE='jetbrains:*'
uv run uvicorn openai_compatible_bridge.main:app --host 127.0.0.1 --port 8000
```

브라우저에서 별도 인증 grant를 승인해야 한다. 기존 Junie refresh token을 복사·공유하지 않는다. Compose 사용 시 Junie 실행 파일과 로그인 환경, OAuth 파일이 있는 전용 writable 디렉터리를 별도로 제공해야 한다. `MODEL_LIST_INCLUDE`는 목록에 노출할 모델을 필터링하며, 생략하면 다른 provider 모델도 OMP 목록에 나타날 수 있다.


예를 들어 인증을 사용하지 않는 로컬 bridge라면 OMP의 `~/.omp/agent/models.yml`에 다음을 추가한다. `baseUrl`의 `/v1`은 OMP discovery에서 다시 붙이지 않도록 `injectV1: false`를 설정한다.

```yaml
providers:
  jetbrains-bridge:
    baseUrl: http://127.0.0.1:8000/v1
    api: openai-completions
    auth: none
    discovery:
      type: openai-models-list
      injectV1: false
      timeoutMs: 15000
```

`BRIDGE_API_KEY`를 활성화한 경우에는 OMP provider에 해당 bridge 인증을 별도로 설정한다. OMP도 모델 목록을 캐시할 수 있으므로 변경 직후에는 OMP의 목록 새로고침이 필요할 수 있다.

### Model Alias

Client 요청에는 provider field를 넣지 않습니다. `model` 값이 registry alias이며, bridge가 provider를 고릅니다.

```json
{
  "llama-local": {
    "provider": "ollama",
    "kind": "chat",
    "provider_model": "llama3.1"
  },
  "gemini-2.5-flash": {
    "provider": "vertex",
    "api": "generateContent",
    "kind": "chat",
    "location": "us-central1"
  },
  "foundry:gemini-3.8-flash": {
    "provider": "foundry",
    "kind": "chat",
    "provider_model": "gemini-3.8-flash",
    "protocol": "google_generate_content"
  }
}
```

---

<a id="config"></a>

## ⚙️ 환경 변수 설정

`.env` 파일을 생성하거나 컨테이너 환경 변수로 다음 값을 주입합니다. 전체 키 목록과 주석은 [`.env.example`](./.env.example)을 참고하세요.

| Variable | Default | Description |
|---|---|---|
| `BRIDGE_API_KEY` | `""` | Client request 보호용 선택적 Bearer token. |
| `EMBEDDINGS_MAX_REQUEST_BYTES` | `1048576` | `/v1/embeddings` HTTP 본문 최대 바이트. JSON 파싱 전에 검사하며 초과 시 `413/request_too_large`. Content-Length가 없는 전송에도 적용. 모델의 토큰 제한과는 별개. |
| `EMBEDDINGS_MAX_INFLIGHT` | `32` | 프로세스당 embedding 요청 처리 상한과 실제 Vertex embedding HTTP 전송 상한에 같은 값을 적용(1 이상). 포화 시 슬롯이 빌 때까지 FIFO(도착 순서)로 대기하며 신규 요청이 대기자를 앞지르지 않음. 요청 상태는 `/readyz`의 `embeddings.inflight`, `embeddings.queue_depth`, `embeddings.capacity`로 계측(리미터가 없거나 metrics 실패 시 `embeddings: null`로 폴백하고 상태 코드는 `cost_tracking`만으로 결정). |
| `EMBEDDINGS_QUEUE_TIMEOUT_SECONDS` | `30` | embedding 슬롯 대기 상한(초, 유한한 0 이상). 상한 초과 시에만 `429/embedding_concurrency_limit`과 `Retry-After: 2`. `0`은 포화 시 즉시 거부를 복원. 음수·비정상 값은 기동 실패. 대기 중 본문도 요청별 바이트 제한 적용. |
| `EMBEDDINGS_RETRY_MAX_ATTEMPTS` | `4` | Vertex embedding 429/503의 총 전송 시도 상한. 최초 전송 포함 `1..8`의 정수. `1`은 자동 재시도 비활성화. |
| `EMBEDDINGS_RETRY_INITIAL_DELAY_SECONDS` | `1` | 최초 backoff의 기준 시간. 유한한 `0.01..30`초. 다음 기준 시간은 2배씩 증가하며 jitter ±20% 적용. |
| `EMBEDDINGS_RETRY_MAX_DELAY_SECONDS` | `8` | 계산한 backoff의 상한. 유한한 `0.01..60`초이며 최초 기준 시간 이상. 더 긴 upstream `Retry-After`는 재시도 전체 시간 안에서 존중. |
| `EMBEDDINGS_RETRY_TIMEOUT_SECONDS` | `30` | 최초 429/503 응답 이후 모든 재시도의 대기·HTTP 슬롯 획득·비용 admission·HTTP 전송을 합친 제한 시간. 유한한 `0.1..300`초. 최초 전송의 기존 HTTP timeout은 유지. |
| `MODEL_REGISTRY_JSON` | `""` | Model alias registry override JSON. |
| `EXTRA_MODELS` | `""` | Backward-compatible comma-separated Vertex predict model additions. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama native API base URL. Docker에서는 `http://host.docker.internal:11434` 권장. |
| `FOUNDRY_BASE_URL` | `""` | 고정된 Foundry OpenAI-compatible `/chat/completions` endpoint. `google_generate_content`는 이 URL의 host/root에서 Google native model endpoint를 안전하게 파생합니다. |
| `FOUNDRY_TOKEN` | `""` | Foundry bearer token. 운영에서는 Kubernetes Secret 또는 로컬 `.env`에만 저장. |
| `FOUNDRY_HTTP_TIMEOUT_SECONDS` | `HTTP_TIMEOUT_SECONDS` | Foundry 전용 HTTP timeout. |
| `OLLAMA_HTTP_TIMEOUT_SECONDS` | `HTTP_TIMEOUT_SECONDS` | Ollama native API 전용 HTTP timeout. reasoning-heavy model은 더 길게 잡을 수 있음. |
| `OLLAMA_THINK` | `true` | Ollama `think` request field 기본값. `true`, `false`, `low`, `medium`, `high`, `omit` 지원. 요청별 `reasoning_effort`/`reasoning.effort`가 있으면 해당 요청에서 override. |
| `STRUCTURED_OUTPUT_REPAIR_ENABLED` | `false` | Dynamic Ollama Cloud `json_schema` 실패에 한해 bounded repair chain 활성화. |
| `STRUCTURED_OUTPUT_REPAIR_MODELS` | `ollama:qwen3.5:cloud,ollama:gemma4:31b-cloud,ollama:glm-5.2:cloud` | Repair 후보 모델 순서. 최대 3개만 사용하며 `metered` 후보의 각 HTTP 시도도 cost/budget gate를 통과해야 함. |
| `GOOGLE_APPLICATION_CREDENTIALS` | `""` | Vertex service account JSON path. |
| `VERTEX_PROJECT` | *(Required)* | GCP project id for Vertex. |
| `VERTEX_LOCATION` | `us-central1` | Default Vertex region. |
| `VERTEX_TASK_TYPE_DEFAULT` | `RETRIEVAL_DOCUMENT` | Default Vertex embedding task type. |
| `VERTEX_AUTO_TRUNCATE` | `true` | Vertex embedding auto truncate flag. |
| `TOKEN_REFRESH_SKEW_SECONDS` | `300` | Vertex access token을 만료 몇 초 전에 선갱신할지. |
| `MAX_CONCURRENCY` | `8` | Vertex chat/rerank의 HTTP 동시 호출 상한. Vertex embedding은 `EMBEDDINGS_MAX_INFLIGHT`를 사용. |
| `HTTP_TIMEOUT_SECONDS` | `60` | Provider HTTP timeout. |
| `DEFAULT_MAX_INSTANCES` | `1` | 알 수 없는 Vertex predict model 호출 시 요청당 instance chunk 폴백. |
| `COST_TRACKING_ENABLED` | `false` | 비용 추적과 forecast budget gate 활성화 여부. `false`는 명시적 보호 해제. |
| `COST_PROVIDER_BILLING_JSON` | `""` | `vertex`/`ollama`/`foundry`의 실제 계약을 `metered`/`subscription`/`nonbillable`로 명시. 추정 기본값 없이 빈 값·누락·잘못된 분류는 fail-closed. |
| `COST_TRACKING_PROVIDERS` | `""` | legacy 호환 키. 새 runtime은 무시하며 billing map만 사용. 과거의 빈 값=전체 추적 의미는 더 이상 적용하지 않음. |
| `COST_LEDGER_PATH` | `""` | 컨테이너 내부 비용 원장 SQLite 파일 경로. Docker에서는 `/data/cost-ledger.db`. |
| `COST_LEDGER_BACKEND` | `sqlite` | `sqlite`는 로컬 호환용, `postgres`는 공유 단일 authority. PostgreSQL 선택 시 SQLite 파일을 열거나 fallback하지 않음. |
| `COST_LEDGER_POSTGRES_DSN` | `""` | 전용 bridge DB의 외부 주입 연결 문자열. `COST_LEDGER_BACKEND=postgres` 필요. 자격증명은 저장소에 기록하지 않음. |
| `COST_LEDGER_POOL_MAX_SIZE` | `8` | PostgreSQL connection pool 최대 크기. admission worker(기본 4) + record worker + admin/readiness 여유분. |
| `COST_LEDGER_POOL_TIMEOUT_SECONDS` | `3` | PostgreSQL connection pool 대기 상한. 선택한 admission timeout보다 짧게 설정하는 것을 권장. |
| `COST_ADMISSION_WORKERS` | `4` | 비용 admission worker 수. `1..64`의 정수만 허용하며, 초과 요청은 슬롯이 빌 때까지 기다립니다. |
| `COST_ADMISSION_TIMEOUT_SECONDS` | `10` | admission queue 대기와 DB 작업을 합친 전체 제한 시간(초). `0.5..30`만 허용하며 만료 시 metered 요청을 fail-closed 처리합니다. |
| `COST_SCHEMA_VERIFY_TTL_SECONDS` | `30` | 성공한 schema 검증 캐시 TTL. `0`이면 매 transaction마다 검증(fail-closed 최대, 느림). 실패는 캐시하지 않음. |
| `COST_LEDGER_DIR` | `./data` | (docker-compose 전용) 컨테이너 `/data`에 mount되는 host bind 경로. 원장 파일을 호스트에 보존. |
| `COST_CHAT_DEFAULT_MAX_OUTPUT_TOKENS` | `4096` | `max_tokens` 미지정 chat 요청의 비용 forecast용 응답 토큰 상한 추정값. |
| `COST_PRICING_JSON` | `""` | 종량제 모델/endpoint의 적용 가격 차원을 빠짐없이 명시하는 JSON. `COST_PRICING_PATH`와 둘 중 하나를 사용. |
| `COST_PRICING_PATH` | `""` | 가격 JSON 파일 경로. |
| `COST_SHORT_WINDOW_SECONDS` | `""` | 단기 budget window 길이(초). 종량제 비용 판정 시 필수. |
| `COST_SHORT_WINDOW_LIMIT_USD` | `""` | 단기 forecast limit. `unlimited`는 예산 한도 차단만 해제하며 분류/가격/DB 검사는 유지. |
| `COST_DAILY_LIMIT_USD` | `""` | UTC 일 단위 forecast limit. `unlimited`는 예산 한도 차단만 해제. |
| `COST_ADMIN_ENABLED` | `false` | Private cost admin API 활성화 여부. |
| `COST_ADMIN_API_KEY` | `""` | Cost admin 전용 Bearer token. `BRIDGE_API_KEY`와 별도 값이어야 함. |
| `COST_RECONCILIATION_ENABLED` | `false` | Cloud Billing BigQuery reconciliation 활성화 여부. |
| `COST_BILLING_BIGQUERY_PROJECT` | `""` | Cloud Billing export 조회용 BigQuery project id. |
| `COST_BILLING_BIGQUERY_DATASET` | `""` | Cloud Billing export dataset. |
| `COST_BILLING_BIGQUERY_TABLE` | `""` | Cloud Billing export table. |
| `COST_RETENTION_REQUEST_DAYS` | `90` | Request-level 비용 원장 보존 기간. |
| `COST_RETENTION_AGGREGATE_MONTHS` | `13` | Aggregate/reconciliation 보존 기간. |

<details>
<summary><b>💡 복잡한 모델 라우팅 추가 방법</b></summary>
<br/>

`MODEL_REGISTRY_JSON`으로 alias별 provider, API type, region, provider-native model id를 직접 제어합니다. 같은 `/v1/chat/completions` 표면에 Vertex, Ollama, Foundry를 alias만으로 섞을 수 있습니다. Foundry endpoint와 bearer token은 registry JSON이 아니라 `FOUNDRY_BASE_URL`/`FOUNDRY_TOKEN` runtime 설정으로 주입합니다.

Foundry chat alias는 선택적으로 `protocol`을 지정합니다. 허용값은 `openai_chat_completions`(기본값), `anthropic_messages`, `xai_responses`, `openai_responses`입니다. provider별 upstream endpoint는 배포된 Foundry host에서 고정 suffix로 파생되며, client 요청이 URL이나 bearer token을 바꿀 수 없습니다. 실제 운영 registry와 `FOUNDRY_TOKEN`은 저장소에 기록하지 않습니다.

```json
{
  "text-embedding-005": {
    "provider": "vertex",
    "api": "predict",
    "kind": "embeddings",
    "location": "us-central1"
  },
  "llama-local": {
    "provider": "ollama",
    "kind": "chat",
    "provider_model": "llama3.1"
  },
  "foundry:claude-3-7-sonnet": {
    "provider": "foundry",
    "kind": "chat",
    "protocol": "anthropic_messages",
    "provider_model": "claude-3-7-sonnet"
  },
  "foundry:grok-3": {
    "provider": "foundry",
    "kind": "chat",
    "protocol": "xai_responses",
    "provider_model": "grok-3"
  },
  "foundry:gpt-6-astra": {
    "provider": "foundry",
    "kind": "chat",
    "protocol": "openai_responses",
    "provider_model": "gpt-6-astra"
  }
}
```

</details>

#### Foundry Sonnet 5.5 / GPT-6.1 Sol

[`examples/foundry-sonnet55-sol61.json`](examples/foundry-sonnet55-sol61.json)은 두 신규 모델을 추가하는 등록 항목입니다. **기존 `MODEL_REGISTRY_JSON`에 병합해야 하며 전체 값을 이 파일로 교체하면 안 됩니다.** Sonnet은 `anthropic_messages`, GPT는 `openai_responses`를 사용합니다. RID는 제공된 Foundry 카탈로그 자료의 값을 그대로 사용했습니다.

로컬 회귀 테스트와 기존 어댑터를 이용한 실제 Foundry의 일반 응답·스트리밍·자동 도구 호출을 검증했습니다. 다만 이 파일을 내려받는 것만으로 운영 별칭이 등록되지는 않습니다. 운영 설정 변경과 재시작, 소비자 경로 확인은 Atlas가 수행합니다. Sonnet의 특정 도구 강제 선택에서 HTTP 400이 있었으며, 자동 선택은 성공했습니다. 검증 범위와 제한, 기존 모델 보존 방법은 [운영 인계 문서](specs/foundry-sonnet55-sol61/atlas-onboarding.md)에 있습니다.

### 비용 추적과 forecast budget gate

<div align="center">
  <img src="./assets/cost-tracking-flow.svg" width="100%" alt="Cost tracking hard budget flow"/>
</div>

`COST_TRACKING_ENABLED=true`이면 `COST_PROVIDER_BILLING_JSON`의 명시적 계약 분류로 비용 경로를 선택합니다. 현재 구현된 provider key는 `vertex`, `ollama`, `foundry`이며 OpenRouter는 구현되어 있지 않습니다.

| 분류 | 비용 경로 |
|---|---|
| `metered` | 적용 가격 차원을 모두 확인하고, 각 실제 HTTP 시도 직전에 DB 예산 검사와 forecast 예약을 완료해야 전송 |
| `subscription` / `nonbillable` | 유효한 분류가 확인되면 비용 경로 전체 우회. 가격·비용 설정 오류나 DB 장애와 무관하게 호출 가능 |
| 누락 / 알 수 없는 값 / 잘못된 분류 설정 | 과금 여부를 신뢰할 수 없어 fail-closed |

운영자가 **실제 payer 계약**을 확인하여 분류를 승인해야 합니다. 예를 들어 `{"vertex":"metered"}`는 Vertex 계약이 종량제일 때만 유효한 예이며 다른 provider의 호출은 분류 누락으로 차단합니다. Vertex는 일반적으로 종량제이지만 이름만으로 기본 분류를 채우지 않습니다. Foundry가 항상 종량제이거나 Ollama가 항상 구독형이라고 가정하지 않습니다. 이 map은 provider 전체 credential/baseURL 경로에 적용됩니다. 같은 provider 안에서 서로 다른 계약을 alias별로 섞을 수 없으며 **별도 deployment로 분리**해야 합니다.

`COST_TRACKING_PROVIDERS`의 빈 값=전체 추적 또는 특정 provider만 추적하던 동작은 legacy입니다. 새 runtime은 이 값을 무시하므로 종량제가 scope에서 빠지거나 구독형이 비용 원장에 기록되지 않습니다. `COST_TRACKING_ENABLED=false`는 기존처럼 **보호 기능 전체를 명시적으로 끈 상태**입니다. 두 limit을 `unlimited`로 설정하면 예산 한도 차단만 해제하며 종량제 분류·가격·DB 검사는 유지합니다.

#### 전송 직전의 공유 판정

PostgreSQL 사용 시 `bridge_cost.cost_events`가 유일한 admission authority입니다. 캐시 없이 같은 transaction의 advisory lock으로 `기간 내 확정 추정치 + 모든 미확정 예약 + 이번 forecast <= limit`을 검사하고 예약합니다. 단기 window와 UTC 일 단위 한도에 각각 적용하며 같은 금액은 허용합니다. 모든 pod가 같은 DB와 분류·가격·한도·lock 계약을 사용해야 합니다.

예약은 요청 입구에서 한 번만 하는 것이 아니라 **각 실제 provider HTTP 전송 직전**에 수행합니다. Embedding batch, repair, retry, streaming fallback도 각각 새 판정과 예약을 거칩니다. 분할 호출은 원 요청의 forecast 전체를 보수적으로 적용하고 repair는 변경된 모델과 forecast를 사용합니다. HTTP transport의 숨은 자동 재시도는 사용하지 않습니다.

Vertex embedding의 `predict`와 `embedContent`는 upstream이 응답한 **429/503**에만 명시적인 재시도를 적용합니다. 기본 기준 간격은 **1 → 2 → 4초**, jitter는 ±20%, 총 시도는 최초 전송 포함 **4회**입니다. `Retry-After`의 초 단위 값과 HTTP-date를 존중하며, 지시된 대기가 남은 전체 시간보다 길면 일찍 재전송하지 않고 마지막 오류를 반환합니다. Backoff 중에는 upstream HTTP 슬롯을 반환하고 다음 전송 때 다시 획득합니다.

`budget_exceeded`, `cost_config_error`, `cost_tracking_unavailable`, 다른 HTTP 오류, 결과가 불확실한 전송 timeout·연결 예외는 재시도하지 않습니다. 재시도 중 호출자 취소는 즉시 전파하고, 재시도 전체 시간이 만료되면 추가 전송 없이 마지막 429/503을 유지합니다. 분할 배치 일부가 실패하면 나머지 배치의 대기·후속 전송도 취소하고 종료를 기다립니다. 실제 재전송마다 비용 admission과 새 예약을 거치며, 실패·취소로 usage가 불확실한 예약은 기존 정책에 따라 `reserved`로 남깁니다. 잘못된 재시도 설정은 client 생성 시 실패합니다.

이는 forecast admission의 직렬화이지 실제 청구액의 절대 상한이 아닙니다. 실제 비용이 forecast보다 크면 `sum(max(actual - forecast, 0))`만큼 초과할 수 있으며, 알 수 없는 청구·가격 오차·사용량 누락 때문에 **실제 invoice 초과액의 유한한 상한은 보장하지 않습니다**. 보수적 forecast와 한도를 운영자가 정해야 합니다.

가격은 `COST_PRICING_JSON` 또는 `COST_PRICING_PATH`로 주입합니다. 종량제 chat은 `input_per_million`과 `output_per_million`, embeddings는 `embedding_per_million`, rerank는 `rerank_per_unit`을 빠짐없이 명시해야 합니다. 아래는 구조 예시이며 빈 문자열은 유효한 가격이 아니므로 차단됩니다. 숫자도 현재 계약 가격의 증거가 아니며 운영자가 확인해야 합니다.

```json
{
  "source": "manual",
  "version": "YYYY-MM-DD",
  "currency": "USD",
  "models": {
    "gemma-4-26b-a4b-it-maas": {
      "chat": {
        "input_per_million": "",
        "output_per_million": ""
      }
    },
    "text-embedding-005": {
      "embeddings": {
        "embedding_per_million": ""
      }
    },
    "text-multilingual-embedding-002": {
      "embeddings": {
        "embedding_per_million": ""
      }
    },
    "gemini-embedding-001": {
      "embeddings": {
        "embedding_per_million": "0.15"
      }
    },
    "gemini-embedding-2": {
      "embeddings": {
        "embedding_per_million": "0.20"
      }
    },
    "semantic-ranker-512@latest": {
      "rerank": {
        "rerank_per_unit": ""
      }
    },
    "ollama:*": {
      "chat": {
        "input_per_million": "",
        "output_per_million": ""
      }
    }
  }
}
```

#### 응답을 기다리게 하지 않는 기록

사용량 기록은 DB I/O 없는 동기 enqueue로 전달합니다.
**프로세스 내 queue 256개 + 전용 record worker 1개**의 best-effort 방식이며 재시도는 0회입니다.
기록 때문에 upstream을 다시 호출하지 않습니다. 알려진 유효한 usage만 가격표 기반 actual 추정액으로 finalize합니다.
명시적 0은 허용하되 누락을 가짜 0으로 채우지 않습니다. 이 추정액은 확정 청구액이 아닙니다.
사용량 누락/오류, upstream 오류, 취소, queue 유실은 `reserved` forecast를 남깁니다.
PostgreSQL에서는 기간 경계·prune·재시작으로 해제되지 않습니다.

Admission은 별도의 worker pool(기본 **4개**, `COST_ADMISSION_WORKERS`)을 사용합니다. 모든 worker가 사용 중이면 요청은 슬롯을 기다리고, queue 대기와 DB 작업을 합친 `COST_ADMISSION_TIMEOUT_SECONDS`(기본 **10초**)가 끝난 경우에만 fail-closed 됩니다. Timeout 이후에도 이미 시작된 DB 작업은 끝날 때까지 worker 슬롯을 점유합니다. PostgreSQL 연결은 **pool**이 소유하며(`COST_LEDGER_POOL_MAX_SIZE`, 기본 8) admission마다 새 TCP 연결을 만들지 않습니다. 풀 고갈·DB 장애도 유료 upstream 호출 전에 거부합니다. schema 검증은 성공 결과를 `COST_SCHEMA_VERIFY_TTL_SECONDS`(기본 30초) 동안 캐시하고 실패는 캐시하지 않습니다. DB connect/statement/lock timeout은 각각 **3/5/5초**입니다. 느린 기록과 별도 executor의 logging은 event loop나 응답 전달을 기다리게 하지 않습니다.

대기 중인 요청을 취소하면 대기 수에서 제외하며, 이미 시작된 DB 작업은 호출자 취소 후에도 완료까지 슬롯을 유지합니다. Worker에 작업을 제출하지 못하면 슬롯을 반환하고 `cost_tracking_unavailable`로 차단합니다. 풀의 물리 연결은 health 검사와 예산 예약 사이 및 후속 admission에서 재사용하지만, 예산 검사와 예약 자체는 매 호출마다 PostgreSQL transaction과 advisory lock으로 수행합니다. 풀 고갈 후 연결이 반환되면 다음 요청에서 다시 판정하며, 이전의 불확실한 예약을 자동 해제하거나 유료 요청을 자동 재전송하지 않습니다.

Shutdown 시 **1초** 동안 drain한 뒤 남은 queue job을 버립니다. 실행 중인 commit의 결과는 불확실할 수 있습니다. Queue는 durable outbox가 아니며 sticky 전역 latch도 없습니다. 이후 새 시도는 DB 상태를 다시 확인하므로 DB가 복구되면 자동으로 새 admission이 가능하지만, 이전 미확정 예약을 자동 해제하거나 replay하지는 않습니다.

Non-stream은 예산 초과 시 HTTP 429 `budget_exceeded`, admission 불가 시 HTTP 503 `cost_tracking_unavailable`로 유료 전송 전에 실패합니다. Streaming gate는 headers 전송 후 generator 안에서 실행되므로 HTTP status 변경 대신 SSE error `budget_exceeded` / `cost_tracking_unavailable` / `cost_config_error`와 `[DONE]`을 보냅니다. 거부된 시도는 유료 전송을 하지 않으며 성공 응답의 OpenAI-compatible shape에 비용 필드를 추가하지 않습니다.

#### SQLite 로컬 호환

`COST_LEDGER_BACKEND=sqlite`는 새 runtime에서도 선택 가능한 로컬 호환 기본값이며 **공유 multi-pod 예산 보장은 없습니다**. PostgreSQL 선택 시 SQLite 파일을 열거나 자동 fallback하지 않습니다. Docker Compose의 `/data` mount는 SQLite 호환용으로 유지하며, 기존 파일을 이관하거나 삭제하지 않습니다.

```bash
mkdir -p ./data
docker compose up -d --build
```

### PostgreSQL cost ledger 개발 후보

`COST_LEDGER_BACKEND=postgres`와 외부 주입 `COST_LEDGER_POSTGRES_DSN`을 명시합니다.
Neurons와 분리된 전용 논리 DB 및 고정 schema `bridge_cost`를 사용하고 runtime에는 DDL 권한을 주지 않습니다.
새 DB는 **빈 상태로 시작**하며 SQLite 데이터 복사·이관·역이관은 제공하지 않습니다.
Migrator 자격증명을 같은 DSN 환경 변수로 주입한 별도 승인 절차에서 schema만 적용합니다.
아래 명령은 절차 예시이며 이번 문서 작업에서 실행하지 않았습니다.

```bash
uv run python -m openai_compatible_bridge.core.cost_schema --apply
```

`/healthz`는 프로세스 생존, `/readyz`는 **admission DB 가용성**을 나타냅니다. Record 성공 여부가 readiness를 결정하지 않습니다. 전역 readiness를 rollout probe에 그대로 연결하면 유효한 구독형/비과금 요청까지 라우팅이 막힐 수 있으므로 Atlas의 별도 승인이 필요합니다. Compose의 liveness는 변경하지 않습니다.

상태의 `recording`에는 기존 `queue_depth`, `queue_capacity`, `record_in_flight`, `records_written`, `records_dropped`, `record_failures`, `usage_missing`, `last_record_success_at`, `admission_in_flight`, `admission_capacity`, `admission_timeouts`, `admission_rejected`, `admission_failures` 키가 그대로 유지됩니다. 새 `admission_queue_depth`는 worker 슬롯을 기다리는 요청 수입니다. 프로세스 로컬 지표는 abrupt kill 때 사라져 정확한 누락 건수를 알 수 없습니다. DB의 미정산 `reserved` 개수와 age를 함께 검토하고 임의 TTL로 해제하지 않습니다.

Rollback은 이미 소비한 PostgreSQL 예산을 우회하는 **과거 SQLite 자동 전환이 아닙니다**. 모든 pod의 유료 호출을 fence하고 PostgreSQL 기록/백업을 보존한 뒤 호환되는 PostgreSQL application revision으로만 되돌립니다. 그렇지 않으면 별도 승인된 reconciliation/carry-forward 전략이 마련될 때까지 유료 호출을 막습니다. 상세 경계는 [admission 계약](specs/cost-ledger-postgres/admission-contract.md), [Atlas 전환 초안](specs/cost-ledger-postgres/atlas-cutover.md), [검증 기록](specs/cost-ledger-postgres/verification.md)을 참고하세요. 이번 범위는 로컬 개발과 #17 갱신이며 서버 연결 확인·DB/Secret/PVC/배포/GitOps/이미지 게시/merge 작업은 하지 않습니다.

운영 확인용 private endpoint는 `COST_ADMIN_ENABLED=true`와 `COST_ADMIN_API_KEY`가 모두 설정된 경우에만 열립니다.

| Method | Endpoint | 용도 |
|---|---|---|
| `GET` | `/admin/cost/status` | 현재 spend, limit, reset time, health, reconciliation 상태 |
| `GET` | `/admin/cost/events` | Allowlist 기반 최근 비용 이벤트 |
| `GET` | `/admin/cost/reconciliation` | Cloud Billing export 대조 상태 |

Cloud Billing BigQuery reconciliation은 request path를 막지 않습니다. Export 미설정은 `unavailable`, 최근 billing row 지연은 `pending`, 권한/쿼리 오류는 `error`로 admin API에 노출됩니다. BigQuery export 연결 설정(`COST_BILLING_BIGQUERY_PROJECT` / `_DATASET` / `_TABLE`)은 [`.env.example`](./.env.example)과 [`docker-compose.yml`](./docker-compose.yml)에 선언되어 있습니다.

기존 [`SwiftBar Ollama 플러그인`](./scripts/swiftbar/ollama-cost.1m.py)은 admin API가 없으면 로컬 SQLite를 진단 fallback으로 읽는 legacy 동작이 있습니다. 이 값은 PostgreSQL의 현재 spend나 admission authority가 아닙니다. PostgreSQL 배포에서는 private admin 상태를 사용하며, 유효한 `subscription`/`nonbillable` 사용량은 새 원장에 기록하지 않습니다.

---

## 🎯 범용 AI 클라이언트 연동 가이드

Bridge를 로컬/사내망에 띄웠다면, OpenAI API 규격을 지원하는 도구에서 custom base URL만 이 bridge로 지정합니다.

| 설정 항목 | 입력할 값 | 비고 |
|---|---|---|
| **API Provider** | `OpenAI` 또는 compatible provider | 도구별 custom OpenAI-compatible provider 선택 |
| **Base URL** | `http://127.0.0.1:8000` | Docker host debug port를 쓰면 `http://127.0.0.1:8930` |
| **API Key** | `BRIDGE_API_KEY` 값 또는 dummy string | `BRIDGE_API_KEY`가 비어 있으면 인증은 강제되지 않음 |
| **Model Name** | Registry alias | 예: `gemini-2.5-flash`, `llama-local` |

Docker network 내부 client는 service DNS를 사용할 수 있습니다.

```text
http://openai-compatible-bridge
```

---

<a id="quickstart"></a>

## 🚀 빠른 시작

### ⚡ 요구사항

- uv
- Docker 및 Docker Compose
- Vertex 사용 시 GCP service account JSON key file (`roles/aiplatform.user`)
- Ollama 사용 시 local Ollama server

### 🧪 로컬 환경 (uv 사용)

```bash
uv run uvicorn openai_compatible_bridge.main:app --reload --port 8000
```

### 🐳 Docker Compose 환경

```bash
docker compose up -d --build
```

로컬 client는 `http://127.0.0.1:8000`을, Docker network 내부에서는 `http://openai-compatible-bridge`를 사용합니다.

---

<a id="api"></a>

## 📡 API 참조

| Method | Endpoint | 호환 규격 | Provider |
|---|---|---|---|
| `GET` | `/healthz` | 프로세스 생존 확인 | Bridge |
| `GET` | `/readyz` | Admission DB readiness. Record health와 별개이며 장애 시 503 | Bridge |
| `GET` | `/v1/models` | OpenAI-compatible | Registry |
| `GET` | `/v1/models/{model_id}` | OpenAI-compatible | Registry |
| `POST` | `/v1/embeddings` | OpenAI-compatible | Vertex |
| `POST` | `/v1/chat/completions` | OpenAI-compatible | Vertex / Ollama / Foundry |
| `POST` | `/v1/rerank` | Cohere / LocalAI-compatible | Vertex |
| `GET` | `/admin/cost/status` | Private admin | Cost ledger |
| `GET` | `/admin/cost/events` | Private admin | Cost ledger |
| `GET` | `/admin/cost/reconciliation` | Private admin | Billing reconciliation |

---

<p align="center">
  <sub><b>AI Model Interop</b> · private / local-only</sub><br/><br/>
  <a href="#arch">🏛️ 아키텍처</a> &nbsp;·&nbsp;
  <a href="#quickstart">🚀 빠른 시작</a> &nbsp;·&nbsp;
  <a href="#top">⬆️ 맨 위로</a>
</p>

---

### Ops notes

- 2026-10-12: End-to-end verification of the automated release chain (docs-only; no functional change).
