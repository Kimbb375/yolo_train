"""agent._list_dir() 검증 (10번 중앙 제어의 원격 경로 탐색 다이얼로그가 쓰는 함수).

python test_agent.py 로 직접 실행.
"""

import os
import tempfile

import agent


def check_list_dir_existing_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "b_folder"))
        with open(os.path.join(tmp, "a_file.txt"), "w", encoding="utf-8") as fh:
            fh.write("x")

        entries, error = agent._list_dir(tmp)
        assert error is None
        names = [e["name"] for e in entries]
        assert "b_folder" in names and "a_file.txt" in names
        # 폴더가 파일보다 먼저 와야 함(정렬 규칙: isDir 내림차순, 이름 오름차순)
        assert entries[0]["name"] == "b_folder" and entries[0]["isDir"]
        assert entries[1]["name"] == "a_file.txt" and not entries[1]["isDir"]

    print("OK: 존재하는 폴더 목록(폴더 우선 정렬) 검증.")


def check_list_dir_missing_path_reports_error() -> None:
    entries, error = agent._list_dir(os.path.join(tempfile.gettempdir(), "no_such_dir_xyz_123"))
    assert entries == []
    assert error is not None

    print("OK: 없는 경로는 예외 대신 (빈 목록, 에러 문자열) 반환 검증.")


if __name__ == "__main__":
    check_list_dir_existing_folder()
    check_list_dir_missing_path_reports_error()
