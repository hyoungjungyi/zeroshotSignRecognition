import sounddevice as sd
import numpy as np
import time
import os
import subprocess
from pydub import AudioSegment
from pydub.playback import _play_with_ffplay
import threading
print(sd.query_devices())
print(sd.default.device)

with open('/tmp/devices.txt', 'w') as f:
    f.write(str(sd.query_devices()))
THRESHOLD = float(os.environ.get("CLAP_THRESHOLD", 0.7))  # 박수 감지 임계값 (실험 필요)
CLAP_INTERVAL = 3  # 초, 박수 두 번 감지할 시간 범위
LOCK_FILE = "/tmp/ui_app.lock"

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
print("base dir: ",SCRIPT_DIR)
BGM_PATH = os.path.join(SCRIPT_DIR,"jamimBGM.mp3")
print("bgm path:;",BGM_PATH)

clap_times = []

def launch_electron_app():
    electron_app_path = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    log_file = "/tmp/electron_app.log"
    try:
        with open(log_file, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching Electron app...\n")

        log = open(log_file, "a")
        # shell=True로 실행 (npm run start는 쉘에서 실행 필요)
        subprocess.Popen(
            "npm run start",
            cwd=electron_app_path,
            stdout=log,
            stderr=log,
            shell=True,
            env=os.environ  # 현재 환경변수 전달 (npm, node 경로 포함)
        )
        with open(log_file, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Electron app launched.\n")
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error launching Electron app: {e}\n")


def play_bgm():
    print("playing bgm,,")
    sound = AudioSegment.from_file(BGM_PATH, format="mp3")
    threading.Thread(target=lambda: _play_with_ffplay(sound)).start()

def is_app_running():
    return os.path.exists(LOCK_FILE)

def mark_app_running():
    with open(LOCK_FILE, "w") as f:
        f.write("running")

def clear_app_running():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

def audio_callback(indata, frames, time_info, status):
    global clap_times
    volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
    current_time = time.time()
    
    if volume_norm > THRESHOLD:
        
        clap_times = [t for t in clap_times if current_time - t < CLAP_INTERVAL]
        clap_times.append(current_time)

        if len(clap_times) >= 2 and not is_app_running():
            print(f"volume: {volume_norm:.2f}")
            print("👏👏 박수 두 번 감지! 앱 실행 중...")
            play_bgm()
            mark_app_running()
            launch_electron_app() 
            clap_times.clear()
        elif is_app_running():
            print("앱이 이미 실행 중입니다. 무시됨.")

def main():
    print("🎤 박수 감지 대기 중... (Ctrl+C로 종료)")
    if is_app_running():
        clear_app_running()

    with sd.InputStream(callback=audio_callback):
        while True:
            sd.sleep(1000)

if __name__ == "__main__":
    main()
