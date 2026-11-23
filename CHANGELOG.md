# Changelog

## Unreleased

- 재전송된 assistant message의 `status` 필드를 chat 변환 시 무시하도록 허용
- 재전송된 output_text의 `annotations` 필드 무시
- Responses 입력 이미지의 `detail`/`file_id`를 변환 시 제거
- Responses 추론 항목 캡처·출력·재전송 추가 (`BRIDGE_REASONING_REPLAY`, 기본 ON)
- Responses 입력 데이터 보존 회귀 수정
- 스트림 사용량 오류 처리 수정
- JetBrains auth_mode 검증 추가
