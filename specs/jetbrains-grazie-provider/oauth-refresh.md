# JetBrains OAuth 갱신

`JETBRAINS_AI_TOKEN`이 있으면 기존 정적 토큰을 사용한다. 비어 있고 `JETBRAINS_AI_OAUTH_FILE`이 지정된 경우에만 해당 파일에서 OAuth 자격 증명을 읽는다. 생성자에 `token`을 명시하면 그 값이 우선하며 빈 문자열도 OAuth를 비활성화한다.

## 독립 자격 증명 준비

별도 로그인 커맨드는 제3자 참고 구현 [junie-api](https://github.com/fabienfleureau/junie-api/tree/c56c86a406ac0d6003458d2979ae81d3e73eeac5/src)의 `auth.ts`, `services/jetbrains/device-auth.ts`, `services/jetbrains/poll-token.ts`, `lib/api-config.ts`에 나온 `junie-cli` 공개 client의 PKCE 흐름을 재현한다. 이는 JetBrains의 공식 지원 문서가 아니며 endpoint 또는 동작이 변경될 수 있다. 별도 인증을 통해 새 grant를 요청하며 Junie CLI 저장소 또는 기존 refresh token을 읽거나 공유하지 않는다.

한 환경에서 별도 grant 발급·갱신과 Grazie 모델 호출이 성공했고 직후 기존 Junie 로그인으로 모델을 호출했다. 이는 JetBrains의 모든 계정이나 향후 동작을 보장하지 않는다. 기존 세션이 유지되어야 한다면 이 비공식 흐름의 위험과 JetBrains 정책을 확인한다.

전용 로컬 디렉터리를 실행 사용자 소유 `0700`으로 준비한 뒤 아래 명령을 사용자가 직접 실행하고 표시된 URL에서 승인한다. 기존 파일을 덮어쓰지 않으며 browser를 자동으로 열지 않는다. 파일이나 토큰 내용을 터미널에 표시하지 않는다.

```sh
uv run python -m openai_compatible_bridge.jetbrains_login --output /private/directory/oauth.json
```

생성한 파일 경로를 `JETBRAINS_AI_OAUTH_FILE`로 지정한다. 별도로 발급받은 호환 OAuth grant도 아래 형식으로 공급할 수 있다.

파일은 동일 사용자 소유 `0600`이어야 한다. 아래는 형식 예시이며 실제 자격 증명을 저장소, 환경변수, 로그 또는 명령행 인자에 넣지 않는다.

```json
{
  "token_url": "https://oauth.example.invalid/token",
  "client_id": "independent-client-id",
  "client_secret": "optional-client-secret",
  "refresh_token": "independently-issued-refresh-token",
  "access_token": "optional-current-access-token",
  "expires_at": 0
}
```

`expires_at`은 Unix epoch 초이며 없거나 유효하지 않으면 갱신한다. `client_secret`과 초기 `access_token`은 선택 사항이다. `client_secret`을 포함하면 form body에 전송한다. 서버 응답은 Bearer `access_token`, 양수 숫자 `expires_in`, 선택적 교체 `refresh_token`을 포함해야 한다. 만료 60초 전에 갱신하며, refresh 성공 시 이전 refresh token은 서버가 교체 토큰을 반환한 경우에만 교체된다. TLS HTTPS endpoint만 허용하고 리디렉션과 환경 proxy를 사용하지 않는다.

파일 경로를 `JETBRAINS_AI_OAUTH_FILE`에 지정한다. Compose에서는 별도 전용 writable 디렉터리를 컨테이너에 mount하고, 그 내부의 컨테이너 경로를 지정해야 한다. 파일 자체만 bind mount하면 atomic 교체가 불가능하다. 기존 compose는 이 비밀 디렉터리를 자동으로 mount하지 않는다.

## 동시성 및 장애

하나의 신뢰할 수 있는 로컬 파일시스템에서 동일 파일과 `.lock` 파일을 공유하는 프로세스끼리 `flock`으로 직렬화하고 잠금 안에서 파일을 다시 읽는다. 서로 다른 파일, 별도 호스트, `flock` 의미가 보장되지 않는 네트워크 파일시스템이나 독립 복제본 간 단일 작성자는 보장하지 않는다. 같은 grant를 그런 방식으로 배포하지 말고 하나의 갱신 소유자만 운영한다.

회전 응답 유실이나 프로세스 중단으로 이전 refresh token이 이미 무효화되었을 가능성을 막기 위해, 네트워크 요청 전에 `refresh_in_progress`를 디스크에 기록한다. 요청 실패, 응답 오류, 저장 실패 또는 중단 후 이 표시가 남으면 자동 재시도하지 않고 실패한다. 운영자는 독립 grant를 다시 발급하여 파일을 안전하게 교체해야 한다. 성공 응답을 받은 뒤 디스크에 안전하게 저장하기 전에 장애가 발생하면 새 토큰을 복구하지 못할 수 있다. 401 응답으로 요청을 자동 재전송하지 않는다.
