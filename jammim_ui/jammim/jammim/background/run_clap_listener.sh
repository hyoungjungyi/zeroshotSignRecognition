#!/bin/bash
export PATH="/opt/homebrew/bin:$PATH"


# jamim 환경 활성화
source /Users/hyeongjeongyi/madcamp/jammim_ui/jammim/jammim/jamim_env/bin/activate
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/clap_listener.py" >> /tmp/claplistener.log 2>&1



# 로그 기록해보기
echo "$(date) - running clap_listener" >> /tmp/claplistener.debug.log
which python >> /tmp/claplistener.debug.log
which ffmpeg >> /tmp/claplistener.debug.log

