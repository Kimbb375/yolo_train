"""updatecheck.py 검증: 버전/의존성 비교 분기(fast 업데이트 가능 여부)와
apply_update()의 zip 압축 해제를 실제 네트워크 없이 흉내내서 확인함.

python test_updatecheck.py 로 직접 실행.
"""

import io
import json
import os
import tempfile
import urllib.request
import zipfile

import appversion
import updatecheck


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._data


def _with_fake_urlopen(payload: bytes):
    def _fake_urlopen(url, timeout=None):
        return _FakeResponse(payload)
    return _fake_urlopen


def check_no_update_when_same_commit() -> None:
    previous = (appversion.COMMIT_SHA, appversion.DEPS_SHA)
    appversion.COMMIT_SHA, appversion.DEPS_SHA = "abc123", "deps1"
    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = _with_fake_urlopen(
        json.dumps({"sha": "abc123", "version": "v1", "depsSha": "deps1"}).encode())
    try:
        assert updatecheck.check_for_update() is None
    finally:
        urllib.request.urlopen = real_urlopen
        appversion.COMMIT_SHA, appversion.DEPS_SHA = previous
    print("OK: 커밋 동일하면 알림 없음.")


def check_fast_update_when_deps_unchanged() -> None:
    previous = (appversion.COMMIT_SHA, appversion.DEPS_SHA)
    appversion.COMMIT_SHA, appversion.DEPS_SHA = "abc123", "deps1"
    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = _with_fake_urlopen(
        json.dumps({"sha": "def456", "version": "v2", "depsSha": "deps1"}).encode())
    try:
        info = updatecheck.check_for_update()
        assert info is not None and info["fast"] is True
    finally:
        urllib.request.urlopen = real_urlopen
        appversion.COMMIT_SHA, appversion.DEPS_SHA = previous
    print("OK: 커밋 다르고 의존성(depsSha) 같으면 fast=True.")


def check_full_download_when_deps_changed() -> None:
    previous = (appversion.COMMIT_SHA, appversion.DEPS_SHA)
    appversion.COMMIT_SHA, appversion.DEPS_SHA = "abc123", "deps1"
    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = _with_fake_urlopen(
        json.dumps({"sha": "def456", "version": "v2", "depsSha": "deps2"}).encode())
    try:
        info = updatecheck.check_for_update()
        assert info is not None and info["fast"] is False
        assert "다시 받으세요" in info["message"]
    finally:
        urllib.request.urlopen = real_urlopen
        appversion.COMMIT_SHA, appversion.DEPS_SHA = previous
    print("OK: 의존성(depsSha) 다르면 fast=False, 전체 재다운로드 안내.")


def check_apply_update_extracts_into_app_dir() -> None:
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w") as archive:
        archive.writestr("main.py", "print('new version')")
    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = _with_fake_urlopen(zip_bytes.getvalue())
    real_file = updatecheck.__file__
    try:
        with tempfile.TemporaryDirectory() as tmp:
            updatecheck.__file__ = os.path.join(tmp, "updatecheck.py")
            assert updatecheck.apply_update(log=lambda *_: None) is True
            with open(os.path.join(tmp, "main.py"), encoding="utf-8") as fh:
                assert fh.read() == "print('new version')"
    finally:
        urllib.request.urlopen = real_urlopen
        updatecheck.__file__ = real_file
    print("OK: apply_update가 update.zip 내용을 앱 폴더에 그대로 풀어씀.")


if __name__ == "__main__":
    check_no_update_when_same_commit()
    check_fast_update_when_deps_unchanged()
    check_full_download_when_deps_changed()
    check_apply_update_extracts_into_app_dir()
