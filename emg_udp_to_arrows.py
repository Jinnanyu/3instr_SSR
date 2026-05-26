# emg_udp_to_arrows.py
# 监听 UDP 的 EMG 解码事件，把动作映射成键盘方向键（↑↓←→）
# 依赖：pip install pynput

import argparse
import json
import socket
import time
from typing import Dict, Optional, Set

from pynput.keyboard import Controller, Key, Listener

KEYBOARD = Controller()

# DEFAULT_MAP = {
#     1: Key.up,
#     2: Key.down,
#     3: Key.left,
#     4: Key.right,
# }


DEFAULT_MAP = {
    1: 'a',
    2: 'd',
    3: 'w',
}

# 全局退出标志
STOP = False

def hotkey_listener():
    # 按 ESC 退出
    def on_press(key):
        global STOP
        if key == Key.esc:
            STOP = True
            return False
    with Listener(on_press=on_press) as listener:
        listener.join()

def press_tap(k):
    KEYBOARD.press(k)
    KEYBOARD.release(k)

def press_down(k):
    KEYBOARD.press(k)

def release_key(k):
    KEYBOARD.release(k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--ip", type=str, default="0.0.0.0", help="本机监听地址，默认 0.0.0.0")
    ap.add_argument("--mode", type=str, choices=["tap", "hold"], default="tap",
                    help="tap=事件start时按一下；hold=start按下,end松开")
    ap.add_argument("--cooldown_ms", type=int, default=120,
                    help="同一方向 start 触发的最小间隔（防止抖动连发）")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.ip, args.port))
    sock.settimeout(0.2)

    print(f"[OK] Listening UDP on {args.ip}:{args.port}")
    print("[OK] Press ESC to stop.")
    print(f"[OK] mode={args.mode}, cooldown_ms={args.cooldown_ms}")
    print("[Map] 1=↑ 2=↓ 3=← 4=→ (可按需改 DEFAULT_MAP)")

    # 启动 ESC 退出监听
    import threading
    th = threading.Thread(target=hotkey_listener, daemon=True)
    th.start()

    held: Set[int] = set()  # hold 模式下当前按住的 label
    last_fire_ts: Dict[int, float] = {}  # label -> last start time (sec)

    def trigger_start(label: int):
        now = time.time()
        last = last_fire_ts.get(label, 0.0)
        if (now - last) * 1000.0 < args.cooldown_ms:
            return
        last_fire_ts[label] = now

        key = DEFAULT_MAP.get(label)
        if key is None:
            return

        if args.mode == "tap":
            press_tap(key)
        else:
            # hold：避免重复 press
            if label not in held:
                press_down(key)
                held.add(label)

    def trigger_end(label: int):
        key = DEFAULT_MAP.get(label)
        if key is None:
            return
        if args.mode == "hold":
            if label in held:
                release_key(key)
                held.remove(label)

    # 主循环
    global STOP
    while not STOP:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break

        try:
            msg = json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            continue

        # 只处理 event 消息（你现在的解码脚本会发 type="event"）
        if msg.get("type") != "event":
            continue

        label = int(msg.get("label", 0))
        state = str(msg.get("state", ""))

        if args.debug:
            t = msg.get("t", None)
            name = msg.get("name", "")
            conf = msg.get("conf", None)
            print(f"[event] t={t} label={label} name={name} state={state} conf={conf}")
            print(f"[frame] t={t} label={label} name={name} state={state} conf={conf}")

        if label == 0:
            # rest 不映射方向键
            continue

        if state == "start":
            trigger_start(label)
        elif state == "end":
            trigger_end(label)
        elif state == "hold":
            # hold 状态可忽略；如果你想做“持续连发”，可以在这里加逻辑
            pass

    # 退出时释放所有按键（避免粘键）
    if args.mode == "hold":
        for lab in list(held):
            k = DEFAULT_MAP.get(lab)
            if k is not None:
                release_key(k)
        held.clear()

    try:
        sock.close()
    except Exception:
        pass
    print("[Done] stopped.")

if __name__ == "__main__":
    main()