"""버전 체크 알림: 켤 때 GitHub Release의 version.json과 지금 실행 중인 빌드의
커밋 SHA(appversion.py, CI가 빌드 시점에 심어줌)를 비교해서 새 버전이 있으면 알려줌.

완전 자동 업데이트(파일 교체)는 아님 — "새 버전 있으니 링크 눌러서 다시 받으세요"
수준. 네트워크 실패 등은 조용히 무시(업데이트 확인 실패가 앱 사용을 막으면 안 됨).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Optional

import appversion

REPO = "Kimbb375/yolo_train"
VERSION_URL = f"https://github.com/{REPO}/releases/latest/download/version.json"
DOWNLOAD_URL = f"https://github.com/{REPO}/releases/latest/download/TrainingDataExtractor.exe"


def check_for_update(timeout: float = 5.0) -> Optional[str]:
    """새 버전이 있으면 안내 메시지(HTML), 없거나 확인 불가하면 None."""
    if not appversion.COMMIT_SHA:
        return None  # 로컬 개발 실행 - 체크 안 함

    try:
        with urllib.request.urlopen(VERSION_URL, timeout=timeout) as response:
            data = json.load(response)
        latest_sha = data.get("sha", "")
    except Exception:  # noqa: BLE001 - 네트워크/파싱 실패는 조용히 무시
        return None

    if latest_sha and latest_sha != appversion.COMMIT_SHA:
        return f'새 버전이 있습니다. <a href="{DOWNLOAD_URL}">여기서 다시 받으세요</a>.'
    return None
