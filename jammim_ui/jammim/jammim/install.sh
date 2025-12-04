#!/bin/bash

# 공통 변수
PLIST_NAME="com.jammim.claplistener.plist"
PLIST_PATH="$HOME/Library/LaunchAgents/$PLIST_NAME"
PROJECT_DIR=$(pwd)

echo "🖥️ Detecting OS..."

UNAME_OUT="$(uname -s)"

case "${UNAME_OUT}" in
    Darwin*)    
        echo "🍎 macOS detected. Installing plist..."

        # plist 템플릿에서 현재 경로로 치환
        sed "s|__PROJECT_DIR__|$PROJECT_DIR|g" background/$PLIST_NAME > /tmp/$PLIST_NAME
        mv /tmp/$PLIST_NAME "$PLIST_PATH"

        # 권한 설정
        chmod 644 "$PLIST_PATH" 

        # launchctl 등록
        launchctl unload "$PLIST_PATH" 2>/dev/null
        launchctl load "$PLIST_PATH"

        echo "✅ macOS: Clap listener service installed and loaded!"
        ;;
    
    MINGW* | MSYS* | CYGWIN*)
        echo "🪟 Windows detected."

        echo "⚠️ Windows에서는 이 install.sh가 필요하지 않습니다."
        echo "   대신, 'background/clap_listener_win.exe'를 자동 시작하도록 레지스트리 등록 또는 바로가기 설정을 해야 합니다."
        ;;
    
    *)
        echo "❌ Unsupported OS: ${UNAME_OUT}"
        exit 1
        ;;
esac
