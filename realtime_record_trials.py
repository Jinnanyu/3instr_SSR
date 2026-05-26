# -*- coding: utf-8 -*-
"""
Realtime EMG recorder (EPStudio WebSocket) -> per-trial NPZ files.

Goal:
- Use the same EPStudio ingestion flow as realtime_waveform.py, but instead of plotting,
  record labeled trials for later training/evaluation.

Key bindings:
- 1: UP    (label_id=0)
- 2: DOWN  (label_id=1)
- 3: LEFT  (label_id=2)
- 4: RIGHT (label_id=3)
- SPACE: start/stop recording one trial with current label
- ESC: stop program (and gracefully stop collection/websocket)

Output:
- out_dir/session_YYYYmmdd_HHMMSS/
  - meta.json
  - trial_0001_label0_up.npz  (contains emg[T,C], sfreq, channels, device_mac, label_id, label_name)

Dependencies:
- epstudiosdk
- pynput (pip install pynput)
"""

import os
import time
import json
import argparse
import configparser
from dataclasses import dataclass
from datetime import datetime
from typing import Union, List, Optional

import numpy as np

from pynput.keyboard import Listener, Key

from epstudiosdk.bean.collection.CollectionDeviceBean import CollectionDeviceBean
from epstudiosdk.client import EpClient
from epstudiosdk.collection.Collection import Collection
from epstudiosdk.websocket import DataReceive, DataType, EventMessage
from epstudiosdk.websocketclient import EpWebSocketClient

from epstudiosdk.request.user.UserLoginRequest import UserLoginRequest
from epstudiosdk.request.guineapig.GuineaPigSetCurrentRequest import GuineaPigSetCurrentRequest


LABELS = {
    0: "left",
    1: "right",
    2: "up",
}

KEY_TO_LABEL = {
    '1': 0,
    '2': 1,
    '3': 2,
}


@dataclass
class Trial:
    label_id: int
    label_name: str
    blocks: List[np.ndarray]  # list of [L,C] float32


class RecorderState:
    def __init__(self):
        self.cur_label_id: int = 0
        self.recording: bool = False
        self.trial: Optional[Trial] = None
        self.trial_idx: int = 0
        self.stop: bool = False

    def start_trial(self):
        if self.recording:
            return
        self.recording = True
        self.trial_idx += 1
        lab = self.cur_label_id
        self.trial = Trial(label_id=lab, label_name=LABELS[lab], blocks=[])
        print(f"[REC] START trial#{self.trial_idx:04d} label={lab}({LABELS[lab]})")

    def stop_trial_and_save(self, out_dir: str, sfreq: float, channels: List[int], device_mac: str):
        if (not self.recording) or (self.trial is None):
            return
        self.recording = False
        trial = self.trial
        self.trial = None

        if len(trial.blocks) == 0:
            print("[REC] STOP (empty) -> skipped")
            return

        emg = np.concatenate(trial.blocks, axis=0)  # [T,C]
        fname = f"trial_{self.trial_idx:04d}_label{trial.label_id}_{trial.label_name}.npz"
        fpath = os.path.join(out_dir, fname)

        np.savez_compressed(
            fpath,
            emg=emg.astype(np.float32),
            sfreq=float(sfreq),
            channels=np.array(channels, dtype=np.int32),
            device_mac=str(device_mac),
            label_id=int(trial.label_id),
            label_name=str(trial.label_name),
            saved_at=str(datetime.now().isoformat(timespec="seconds")),
        )
        dur = emg.shape[0] / float(sfreq)
        print(f"[REC] STOP -> saved {fpath}  shape={emg.shape}  dur={dur:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None,
                    help="config.ini path; if omitted, use ./data/config.ini next to this script")
    ap.add_argument("--sfreq", type=float, default=1000.0,
                    help="sampling rate (Hz) used for saving meta and later windowing")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="output folder; default creates session_YYYYmmdd_HHMMSS under ./records")
    ap.add_argument("--max_list_len", type=int, default=10000,
                    help="safety: drop a websocket chunk if any channel list is longer than this")
    args = ap.parse_args()

    # --- config load (same style as realtime_waveform.py) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.config is None:
        config_path = os.path.abspath(os.path.join(script_dir, "data", "config.ini"))
    else:
        config_path = os.path.abspath(args.config)

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    host = cfg.get("config-data", "host", fallback="http://localhost").split(";")[0]
    port = cfg.get("config-data", "port", fallback="8080")
    websocket_host = cfg.get("config-data", "websocket_host", fallback="ws://localhost").split(";")[0]
    websocket_port = cfg.get("config-data", "websocket_port", fallback="9000")
    login_id = cfg.get("config-data", "login_id", fallback="admin")
    password = cfg.get("config-data", "password", fallback="admin")
    guinea_pig_id = cfg.get("config-data", "guinea_pig_id", fallback="20220527")

    device_mac = cfg.get("app", "device_mac", fallback="FB:58:F8:40:9F:1C")
    channels_str = cfg.get("app", "channels", fallback="1,2,3,4,5")
    channels = [int(x.strip()) for x in channels_str.split(",") if x.strip()]

    # --- output session dir ---
    if args.out_dir is None:
        out_root = os.path.join(script_dir, "records")
        os.makedirs(out_root, exist_ok=True)
        session_name = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(out_root, session_name)
    else:
        out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": config_path,
        "host": host,
        "port": port,
        "websocket_host": websocket_host,
        "websocket_port": websocket_port,
        "device_mac": device_mac,
        "channels": channels,
        "sfreq": float(args.sfreq),
        "label_order": LABELS,
        "keymap": {k: int(v) for k, v in KEY_TO_LABEL.items()},
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] config:", config_path)
    print("[OK] device_mac:", device_mac)
    print("[OK] channels:", channels)
    print("[OK] out_dir:", out_dir)
    print("\n[Keys] 1=LEFT 2=RIGHT 3=UP | SPACE=start/stop trial | ESC=quit\n")

    state = RecorderState()

    # --- keyboard listener thread ---
    def on_press(key):
        # label switch
        try:
            ch = key.char
            if ch in KEY_TO_LABEL:
                state.cur_label_id = KEY_TO_LABEL[ch]
                print(f"[UI] current label -> {state.cur_label_id}({LABELS[state.cur_label_id]})")
                return
        except Exception:
            pass

        if key == Key.space:
            if not state.recording:
                state.start_trial()
            else:
                state.stop_trial_and_save(out_dir=out_dir, sfreq=args.sfreq,
                                          channels=channels, device_mac=device_mac)
            return

        if key == Key.esc:
            state.stop = True
            print("[UI] ESC pressed -> stopping...")
            return False

    listener = Listener(on_press=on_press)
    listener.start()

    # --- websocket event handler ---
    class MyEvent(EventMessage):
        def on_data(self, data: Union[str, dict, DataReceive]):
            if not (isinstance(data, DataReceive) and data.dataType == DataType.EP):
                return
            for device_data in data.data:
                if device_data.deviceId != device_mac:
                    continue

                # Build aligned block: [L,C]
                vals = []
                L = None
                for ch in channels:
                    arr = device_data.data.get(str(ch), None)
                    if arr is None or (not isinstance(arr, list)):
                        return
                    if len(arr) > args.max_list_len:
                        return
                    if L is None:
                        L = len(arr)
                    else:
                        if len(arr) != L:
                            L = min(L, len(arr))
                    vals.append(arr)

                if L is None or L <= 0:
                    return

                block = np.stack([np.asarray(v[:L], dtype=np.float32) for v in vals], axis=1)  # [L,C]

                if state.recording and state.trial is not None:
                    state.trial.blocks.append(block)

    # --- connect & start collection (same flow as realtime_waveform.py) ---
    client = EpClient(init_user_status=True)
    login_res = client.do_action_json(UserLoginRequest(login_id, password))
    print("Login result:", login_res)
    set_patient_res = client.do_action_json(GuineaPigSetCurrentRequest(guinea_pig_id))
    print("Set patient result:", set_patient_res)

    event_handler = MyEvent()
    websocket_client = EpWebSocketClient(event_msg=event_handler, enable_trace=False)
    websocket_client.start()

    device = CollectionDeviceBean(device_mac, channelStatus=channels)
    collection = Collection(client=client, device_list=[device], record_status=False)

    print("Starting collection...")
    result = collection.start_collection()
    print("Collection start result:", result)

    try:
        while not state.stop:
            time.sleep(0.1)
    finally:
        if state.recording:
            state.stop_trial_and_save(out_dir=out_dir, sfreq=args.sfreq,
                                      channels=channels, device_mac=device_mac)

        try:
            collection.stop_collection()
        except Exception:
            pass
        try:
            websocket_client.stop()
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass

        print("[Done] stopped. Session saved in:", out_dir)


if __name__ == "__main__":
    main()
