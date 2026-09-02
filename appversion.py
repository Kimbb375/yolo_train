"""빌드 버전 표시.

CI(.github/workflows/build.yml)가 PyInstaller로 빌드하기 직전에 이 파일의
COMMIT_SHA/BUILD_VERSION을 실제 값으로 덮어씀. 로컬 개발 실행(uv run main.py)에서는
둘 다 None으로 남아있어서 updatecheck.py가 업데이트 확인을 건너뜀 — 개발 중에 매번
"새 버전 있음" 알림이 뜨는 걸 방지함.

BUILD_VERSION은 "날짜.빌드번호"(예: 2026.09.02.3, UTC 기준 날짜 + GitHub Actions
run_number) 형식 — push마다 자동으로 매겨지는 사람이 읽을 수 있는 버전 표시.
실제 업데이트 유무 판단은 COMMIT_SHA(정확한 커밋 비교)로 함.
"""

COMMIT_SHA: str | None = None
BUILD_VERSION: str | None = None
