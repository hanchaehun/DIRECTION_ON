#!/usr/bin/env bash
# 모의 함수 서버 시작: bash start_mock.sh <데이터 파일>  (측정 때는 mock_db_test.json). 8100번, 0.0.0.0
# 데이터 경로는 실행한 위치 기준 또는 이 폴더 기준 둘 다 받는다.
DB="${1:-data/mock_db.json}"
if [ -f "$DB" ]; then DB="$(cd "$(dirname "$DB")" && pwd)/$(basename "$DB")"; fi
cd "$(dirname "$0")"
exec python3 mock_tools/server.py --host 0.0.0.0 --port 8100 --db "$DB"
