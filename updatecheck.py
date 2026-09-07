"""버전 체크 알림: 켤 때 GitHub Release의 version.json과 지금 실행 중인 빌드의
커밋 SHA(appversion.py, CI가 빌드 시점에 심어줌)를 비교해서 새 버전이 있으면 알려줌.

의존성(pyproject.toml/uv.lock)이 그대로면 apply_update()로 소스(.py)만 담은
가벼운 update.zip을 받아 제자리에 덮어쓸 수 있음("빠른 업데이트") - python/
site-packages 폴더는 안 건드리므로 매번 몇백MB짜리 전체 zip을 다시 받을 필요가
없음. 의존성이 바뀐 버전이면 site-packages도 새로 깔아야 하므로 전체 zip
재다운로드를 안내함. 네트워크 실패 등은 조용히 무시(업데이트 확인 실패가 앱
사용을 막으면 안 됨).
"""

from __future__ import annotations

import io
import json
import os
import urllib.request
import zipfile
from typing import Optional

import appversion

REPO = "Kimbb375/yolo_train"
VERSION_URL = f"https://github.com/{REPO}/releases/latest/download/version.json"
DOWNLOAD_URL = f"https://github.com/{REPO}/releases/latest/download/TrainingDataExtractor.zip"
UPDATE_ZIP_URL = f"https://github.com/{REPO}/releases/latest/download/update.zip"


def check_for_update(timeout: float = 5.0) -> Optional[dict]:
    """새 버전이 있으면 {"message": html, "fast": bool}, 없거나 확인 불가하면 None.
    fast=True면 apply_update()로 재다운로드 없이 소스만 갱신 가능."""
    if not appversion.COMMIT_SHA:
        return None  # 로컬 개발 실행 - 체크 안 함

    try:
        with urllib.request.urlopen(VERSION_URL, timeout=timeout) as response:
            data = json.load(response)
        latest_sha = data.get("sha", "")
        latest_version = data.get("version", "")
        latest_deps_sha = data.get("depsSha", "")
    except Exception:  # noqa: BLE001 - 네트워크/파싱 실패는 조용히 무시
        return None

    if not latest_sha or latest_sha == appversion.COMMIT_SHA:
        return None

    label = f"({latest_version}) " if latest_version else ""
    fast = bool(latest_deps_sha) and latest_deps_sha == appversion.DEPS_SHA
    if fast:
        message = f"새 버전 {label}이 있습니다. (설치된 패키지는 그대로라 빠르게 적용 가능)"
    else:
        message = (f'새 버전 {label}이 있습니다. <a href="{DOWNLOAD_URL}">여기서 다시 받으세요</a> '
                    f'(zip 압축 풀고 안의 Run.bat 실행).')
    return {"message": message, "fast": fast}


def apply_update(log=print, timeout: float = 30.0) -> bool:
    """update.zip(소스 파일만)을 받아 이 앱이 실행 중인 폴더에 덮어씀. 같은 프로세스
    제약(gpu_setup.py 참고)으로 재시작해야 적용됨."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    log(f"[업데이트] {UPDATE_ZIP_URL} 받는 중...")
    with urllib.request.urlopen(UPDATE_ZIP_URL, timeout=timeout) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(app_dir)
    log("[업데이트] 적용 완료. 앱을 재시작하세요.")
    return True
