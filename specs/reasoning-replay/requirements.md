# 추론 연속성(Reasoning Item Replay) 요구사항

## 승인 대상

- Source of truth: `specs/reasoning-replay/requirements.md`
- 상태: 구현 완료, 로컬 브릿지 실경로 검증 완료. PR/CI 및 k3s 배포 대기.
- 대상 저장소: `openai-compatible-bridge`

## 문제 정의

브릿지의 chat completions 입구로 들어온 클라이언트가 이전 턴의 추론을 `reasoning_content`로 되돌려 보내면, 브릿지는 이를 업스트림에 전달할 방법이 없어 필드를 버린다. 그 결과 JetBrains Responses 경로에서는 턴 간 추론 연속성이 유지되지 않는다.

동시에 브릿지는 업스트림이 내려주는 `reasoning` 항목 자체를 현재 전부 버리고 있다. 즉 추론 모델을 쓰더라도 클라이언트에 추론 항목이 전달되지 않는다.

## 확인된 사실 (2026-11-22 프로브)

1. JetBrains 게이트웨이 `gpt-6-astra` + `/v1/responses`는 `reasoning` 출력 항목을 반환한다. 항목에는 `id`, `encrypted_content`, `summary`, `content`가 있다.
2. `output_item.added`의 `encrypted_content`는 부분값이고, `output_item.done`의 값이 완전체다. 재전송에는 완전체를 써야 한다.
3. `gpt-6.1-terra`는 `reasoning.effort=high`에서도 `reasoning` 항목을 반환하지 않았다. 항목이 없으면 기능은 아무 동작도 하지 않아야 한다.
4. 재전송 형식은 `{"type":"reasoning","id":...,"encrypted_content":...,"summary":[]}`다. `summary`를 빼면 400 `Missing required parameter: 'input[1].summary'`가 난다. `summary: []`를 포함하면 200이고 다음 턴 답변도 정상이다(프로브: 8 → 9).
5. 실패했던 OMP 요청 본문은 520개 메시지 중 assistant 1개에만 `reasoning_content`(1,424자)가 있었고 `thinking_blocks`는 없었다. 즉 클라이언트는 텍스트만 보존한다.
6. OMP에는 `openai-responses` API 모드와 네이티브 이력 재전송 모듈(`openai-responses-state`, `ReplaysNativeHistory`)이 있다.

## 현재 코드 갭

- `core/protocol_stream.py`: `response.output_item.*`에서 `function_call`만 처리하고 `reasoning` 항목은 무시한다.
- `responses_ingress.py`: 입력의 `reasoning` 항목을 검증 없이 건너뛴다(210행). 출력에는 `message`와 `function_call`만 넣는다.
- `main.py` chat 입구: `reasoning_content`는 폐기하고, `thinking_blocks`는 검증만 통과시킨 뒤 Anthropic 경로에서만 사용한다.
- 업스트림 요청 바디에 `include: ["reasoning.encrypted_content"]`가 없다.

## 설계 결정

### D1. 내부 표현

- 내부 이벤트/결과에 `reasoning_items: list[dict]`를 추가한다.
- 항목은 불투명하게 취급한다: `{id, encrypted_content, summary}`.
- `encrypted_content`는 로그, 에러 메시지, 감사 산출물에 절대 남기지 않는다. ID만 남긴다.

### D2. 업스트림 캡처

- 스트림: `response.output_item.added`에서 `type == "reasoning"`이면 ID와 부분 암호문을 기록하고, `response.output_item.done`에서 완전체로 교체한다.
- 비스트림: 응답의 `output` 배열에서 `reasoning` 항목을 순서대로 수집한다.
- 캡처한 항목은 원래 출력 순서(추론 → 메시지/도구 호출)를 유지한다.

### D3. Responses 인그레스 출력

- `/v1/responses`의 `output` 배열에 `reasoning` 항목을 `message`/`function_call` 앞에 포함한다.
- SSE에서도 `response.output_item.added`/`done`으로 같은 순서와 `output_index`를 사용한다.
- 항목이 없으면 기존 동작과 완전히 동일하다.

### D4. Chat 인그레스 출력 (호환 경로)

- assistant 메시지의 `thinking_blocks`에 `{"type":"reasoning_item","id":...,"encrypted_content":...,"summary":[...]}` 형태로 첨부한다.
- 스트리밍 delta에도 동일하게 싣는다.
- 기존 Anthropic `thinking`/`redacted_thinking` 블록과 타입으로 구분하며, Anthropic 변환기는 새 타입을 무시한다.
- 이 경로는 클라이언트가 불투명 블록을 보존할 때만 유효하다.

### D5. 재전송

- `/v1/responses` 입력의 `reasoning` 항목을 검증해 업스트림 `input` 배열의 같은 위치에 전달한다.
  - 필수: `id`, `encrypted_content`, `summary`(빈 배열 허용).
  - `summary`가 없으면 400이므로 브릿지가 `summary: []`를 보충한다.
- Chat 입력의 `thinking_blocks`에 `reasoning_item`이 있으면, 해당 assistant 메시지 직전에 대응하는 `reasoning` 입력 항목을 배치한다.
- 항목 ID 기준으로 중복을 제거한다. 클라이언트가 보낸 항목이 항상 우선한다.
- 텍스트 `reasoning_content`만으로는 항목을 복원할 수 없다. 현재처럼 폐기한다.

### D6. 업스트림 요청

- Responses 프로토콜 요청 바디에 `include: ["reasoning.encrypted_content"]`를 추가한다.
- 업스트림이 항목을 반환하지 않으면 아무 것도 추가하지 않는다.

### D7. 서버 캐시 (2차, chat 모드 한정)

- 목적: 불투명 블록을 보존하지 않는 클라이언트(현재 OMP chat 모드)에서도 연속성을 제공한다.
- 키: 마지막 assistant 턴까지의 메시지 접두사를 정규화해 SHA-256. (role, content 텍스트, tool_call id/인자, tool_call_id)
- 값: 해당 접두사 뒤에 생성된 `reasoning_items`와 위치.
- 다음 요청에서 접두사가 일치하고 클라이언트가 항목을 주지 않았을 때만 주입한다. 접두사가 다르면 주입하지 않는다.
- 메모리 전용, LRU 64개 / 총 8 MiB / TTL 30분. 재시작 시 소멸. 디스크 저장 없음.
- ID 중복 제거. 클라이언트 제공 항목이 있으면 캐시를 쓰지 않는다.

### D8. 실패 처리

- 재전송 항목 때문에 업스트림이 400을 내면, 항목을 제거한 같은 요청으로 1회만 재시도한다.
- 재시도 사실은 ID만 포함한 경고로 남긴다. 암호문과 추론 텍스트는 남기지 않는다.
- 그래도 실패하면 기존 오류 경로를 그대로 사용한다.

### D9. 플래그

- `BRIDGE_REASONING_REPLAY` 환경변수. 기본 ON, `0`이면 캡처·출력·재전송을 모두 끈다.
- 꺼진 상태는 현재 동작과 동일해야 한다.

## 클라이언트 경로 결정

- **권장 A (zero-loss, 변경 최소)**: OMP 제공자 설정을 `/v1/responses` + `api: openai-responses`로 바꾼다. D2~D6만으로 네이티브 이력 재전송이 성립한다.
- **호환 B**: chat completions를 유지해야 하면 D7 캐시가 필요하다. 캐시는 휴리스틱이므로 A보다 약하다.

## 데이터 흐름

```
업스트림 SSE reasoning(done)
  -> protocol_stream: reasoning_items 캡처
  -> responses_ingress: output/SSE에 reasoning 항목 emit
  -> 클라이언트가 다음 턴에 그대로 재전송
  -> responses_ingress: 입력 검증 후 upstream input에 재배치
  -> 업스트림: 이전 추론을 이어서 사용
```

## 보안

- `encrypted_content`와 추론 텍스트는 민감 정보로 취급한다. 채팅, 로그, Git, 스펙 문서에 값을 넣지 않는다.
- 캐시는 메모리 전용이며 프로세스 종료 시 사라진다.
- 오류 응답에는 항목 ID와 개수만 포함한다.

## 테스트

1. `output_item.added`(부분) → `done`(완전) 교체가 캡처되는지.
2. 비스트림 응답에서 `reasoning` 항목을 수집하는지.
3. Responses 출력에서 추론 항목이 메시지/도구 호출보다 앞서고 `output_index`가 연속인지.
4. Chat 출력의 `thinking_blocks`에 `reasoning_item`이 실리는지, 스트림 delta도 같은지.
5. 입력 재전송 시 순서가 보존되고 `summary` 누락이 보충되는지.
6. 같은 ID가 중복 주입되지 않는지.
7. 항목이 없을 때 응답이 기존과 바이트 수준에서 동일한지(회귀).
8. 캐시: 접두사 일치 시 1회 주입, 불일치 시 미주입, TTL/LRU 동작.
9. 로그·에러에 `encrypted_content`가 없는지.
10. 실패 재시도(항목 제거)가 1회만 발생하는지.

## 롤아웃

1. Phase 0 프로브: 완료(위 확인된 사실).
2. 구현 + 단위 테스트.
3. 로컬 브릿지에서 `gpt-6-astra` 2턴 검증: 추론 항목 출력 → 재전송 → 200 및 답변 연속성 확인.
4. PR/CI.
5. k3s 이미지 배포와 OMP 제공자 전환은 별도 승인 후 진행.

## 미확인 사항

- 게이트웨이가 `store=true` 상태에서 `id`만으로 재전송을 허용하는지. 현재 설계는 `encrypted_content`를 항상 사용한다.
- `summary`가 비어 있지 않은 경우의 정확한 허용 형식.
- OMP Responses 모드가 브릿지에 요구하는 `store`/`previous_response_id`/`item_reference` 범위.
- `gpt-6.1-terra`가 추론 항목을 전혀 반환하지 않는 이유(모델 특성인지 게이트웨이 정책인지).

## 구현 및 검증 기록

- 킬 스위치: `BRIDGE_REASONING_REPLAY` 기본 ON, `0`이면 캡처·출력·재전송을 모두 끈다.
- 단위 테스트: JetBrains 추론·Responses 인그레스·JetBrains chat 80개 통과. 전체(Postgres 환경 테스트 제외) 1,046개 통과, 25개 오류는 `ATLAS_TEST_PG_DSN` 미설정에 따른 환경 오류.
- 로컬 브릿지 `/v1/responses` 2턴 검증(2026-11-22):
  - 1턴: 출력 항목 `["reasoning", "message"]`, reasoning ID 있음, `encrypted_content` 2,616자, summary 빈 배열, 답변 "8".
  - 2턴: 같은 추론 항목을 입력으로 재전송 → 200, 답변 "9". 재전송 항목에 `summary: []`가 포함되지 않으면 업스트림이 400을 낸다.
- OMP 종단 검증(2026-11-22):
  - OMP 제공자를 `api: openai-responses`, baseUrl `http://127.0.0.1:18123/v1`로 전환.
  - OMP 요청: `reasoning.effort=high`, `store=false`, `include=["reasoning.encrypted_content"]`.
  - 도구 호출 후속 요청 입력에 `reasoning` 항목이 `encrypted_content`와 함께 포함됨을 확인. 이 항목에는 `id`가 없고 `summary`만 있으며, 브릿지는 암호문 기준으로 수용한다.
  - 해당 요청 200, OMP 답변 정상. 도구 호출 전후 추론 항목의 저장·재전송이 실제로 일어난다.
