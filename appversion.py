"""빌드 버전 표시.

CI(.github/workflows/build.yml)가 PyInstaller로 빌드하기 직전에 이 파일의
COMMIT_SHA를 실제 커밋 SHA 문자열로 덮어씀. 로컬 개발 실행(uv run main.py)에서는
None으로 남아있어서 updatecheck.py가 업데이트 확인을 건너뜀 — 개발 중에 매번
"새 버전 있음" 알림이 뜨는 걸 방지함.
"""

COMMIT_SHA: str | None = None
