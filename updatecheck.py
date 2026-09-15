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
import ssl
import urllib.request
import zipfile
from typing import Optional

import appversion

REPO = "Kimbb375/yolo_train"
VERSION_URL = f"https://github.com/{REPO}/releases/latest/download/version.json"
DOWNLOAD_URL = f"https://github.com/{REPO}/releases/latest/download/TrainingDataExtractor.zip"
UPDATE_ZIP_URL = f"https://github.com/{REPO}/releases/latest/download/update.zip"

try:
    # 포터블 standalone CPython(uv가 받은 것, gpu_setup.py 참고)은 일반 python.org
    # 설치본과 달리 OS 인증서 저장소를 못 찾아서 urlopen이 "CERTIFICATE_VERIFY_FAILED:
    # unable to get local issuer certificate"로 실패하는 PC가 있었음(사용자 보고). certifi는
    # ultralytics(requests)의 전이 의존성이라 이미 항상 설치돼 있음 - 그 CA 번들을 명시적으로
    # 써서 이 PC의 인증서 저장소 상태와 무관하게 항상 검증 가능하게 함.
    import certifi
    _SSL_CONTEXT: Optional[ssl.SSLContext] = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # noqa: BLE001 - certifi가 없어도 기본 컨텍스트로 폴백
    _SSL_CONTEXT = None


def check_for_update(timeout: float = 5.0, silent: bool = True) -> Optional[dict]:
    """새 버전이 있으면 {"message": html, "fast": bool}, 없으면 None.
    fast=True면 apply_update()로 재다운로드 없이 소스만 갱신 가능.

    silent=True(기본, 앱 시작 시 자동 체크용): 네트워크/파싱 실패도 조용히 None 반환 -
    업데이트 확인 실패가 앱 사용을 막으면 안 됨.
    silent=False(상태 표시줄 "업데이트 확인" 수동 클릭용): 실패를 그대로 예외로 던짐 -
    안 그러면 방화벽 등으로 GitHub 접속 자체가 막힌 PC에서도 "최신 버전"으로만 보여서
    실제로는 확인이 실패했다는 걸 사용자가 알 방법이 없었음(사용자 보고: "다른 PC에서는
    업데이트가 계속 최신 버전이라고만 뜸" - 방화벽으로 매 요청이 조용히 실패하고 있었음)."""
    if not appversion.COMMIT_SHA:
        return None  # 로컬 개발 실행 - 체크 안 함

    try:
        with urllib.request.urlopen(VERSION_URL, timeout=timeout, context=_SSL_CONTEXT) as response:
            data = json.load(response)
        latest_sha = data.get("sha", "")
        latest_version = data.get("version", "")
        latest_deps_sha = data.get("depsSha", "")
    except Exception:
        if silent:
            return None
        raise

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
    with urllib.request.urlopen(UPDATE_ZIP_URL, timeout=timeout, context=_SSL_CONTEXT) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(app_dir)
    log("[업데이트] 적용 완료. 앱을 재시작하세요.")
    return True
