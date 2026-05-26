#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""realtime_decode_udp.py

把 realtime_waveform.py 从 EPStudio WebSocket 收到的实时 EMG 数据
接到 emg_replay_udp.py 的 1D-CNN 分类 + UDP 协议上。

输出 UDP JSON 与 emg_replay_udp.py 完全一致：
  - frame: {type:"frame", t, label, name, conf, active_ratio}
  - event: {type:"event", t, label, name, conf, state:"start"|"hold"|"end"}


依赖：epstudiosdk, torch, numpy

运行示例：
  python realtime_decode_udp.py --ckpt cnn_2actions_leftright.pth --sfreq 1000 \
      --device_mac F9:DC:B2:3E:B4:BD --channels 1,2 --ip 127.0.0.1 --port 5005

说明：
- --sfreq 必须填对（用于 win/step/rms 这些“秒 -> 点数”的换算）
- --channels 的顺序必须与你训练时喂给模型的通道顺序一致（现在通常为 1,2）
"""

import argparse
import configparser
import json
import os
import pickle
import queue
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .features_emg import extract_features_window

from epstudiosdk.bean.collection.CollectionDeviceBean import CollectionDeviceBean
from epstudiosdk.client import EpClient
from epstudiosdk.collection.Collection import Collection
from epstudiosdk.websocket import DataReceive, DataType, EventMessage
from epstudiosdk.websocketclient import EpWebSocketClient
from epstudiosdk.request.user.UserLoginRequest import UserLoginRequest
from epstudiosdk.request.guineapig.GuineaPigSetCurrentRequest import GuineaPigSetCurrentRequest


# ----------------- 模型（与训练脚本一致） -----------------
class EMGCNN(nn.Module):
    def __init__(self, n_channels: int = 5, n_classes: int = 2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


def load_ckpt(ckpt_path: Path, n_channels: int):
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=torch_device, weights_only=False)

    class_names = ckpt.get("class_names", None)
    if class_names is None:
        raise KeyError("ckpt 中找不到 class_names（请确认使用训练脚本保存的 .pth）")
    class_names = list(class_names)
    n_classes = len(class_names)

    model = EMGCNN(n_channels=n_channels, n_classes=n_classes).to(torch_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean = ckpt.get("mean", None)
    std = ckpt.get("std", None)
    per_window_norm = bool(ckpt.get("per_window_norm", False))

    if mean is not None:
        mean = np.asarray(mean, dtype=np.float32)
    if std is not None:
        std = np.asarray(std, dtype=np.float32)

    gesture_names = ["rest"] + class_names
    return model, torch_device, mean, std, per_window_norm, gesture_names


def load_svm(svm_path: Path):
    with open(svm_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        class_names = list(obj.get("class_names", ["left", "right", "up"]))
    else:
        model = obj
        class_names = ["left", "right", "up"]  # 默认顺序，训练时也必须保持一致
    return model, class_names


def infer_cnn_prob(model, torch_device, emg_win, per_window_norm, mean, std):
    if per_window_norm:
        mu = emg_win.mean(axis=0, keepdims=True)
        sigma = emg_win.std(axis=0, keepdims=True) + 1e-8
        emg_norm = (emg_win - mu) / sigma
        x_np = emg_norm.T[None, :, :]
    else:
        if mean is None or std is None:
            raise RuntimeError("全局归一化模式下 ckpt 必须提供 mean/std")
        win_tc = emg_win[None, :, :]
        win_norm = (win_tc - mean) / std
        x_np = np.transpose(win_norm.astype(np.float32), (0, 2, 1))
    x = torch.from_numpy(x_np.astype(np.float32)).to(torch_device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return prob.astype(np.float32)


def infer_svm_prob(svm_model, emg_win):
    feat, _ = extract_features_window(emg_win)
    prob = svm_model.predict_proba(feat[None, :])[0]
    return np.asarray(prob, dtype=np.float32)

# ----------------- UDP 消息 / 事件平滑 -----------------
@dataclass
class FrameMsg:
    type: str
    t: float
    label: int
    name: str
    conf: float
    active_ratio: float


@dataclass
class EventMsg:
    type: str
    t: float
    label: int
    name: str
    conf: float
    state: str


class TapEventSmoother:
    """
    只产生 start/end（tap 控制够用）：
    - 必须先稳定 rest 一段时间（min_rest_steps）才允许 start（armed）
    - end 后 refractory_sec 内屏蔽 opposite start（解决回弹反向误触发）
    - 动作期间不允许 action->action 直接切换（避免抖动连发）
    适配 2 类 left/right，opposite 映射为 {1:2, 2:1}
    """
    def __init__(
        self,
        gesture_names: List[str],
        step_sec: float,
        min_rest_sec: float = 0.15,
        refractory_sec: float = 0.25,
        conf_thresh: float = 0.55,
    ):
        self.gesture_names = gesture_names
        self.step_sec = float(step_sec)
        self.min_rest_steps = max(1, int(round(min_rest_sec / self.step_sec)))
        self.refractory_sec = float(refractory_sec)
        self.conf_thresh = float(conf_thresh)

        self.cur_label = 0
        self.rest_streak = 0
        self.armed = True

        self.last_end_t = -1e9
        self.last_end_label = 0
        # 修改 opposite 映射为 left(1) ↔ right(2)
        # self.opposite = {1: 2, 2: 1}  # 0=rest, 1=left, 2=right

    def update(self, t: float, label: int, conf: float) -> List[EventMsg]:
        label = int(label)
        conf = float(conf)

        # 统计稳定 rest
        if label == 0:
            self.rest_streak += 1
        else:
            self.rest_streak = 0

        # 稳定 rest -> 解锁下一次 start
        if self.rest_streak >= self.min_rest_steps:
            self.armed = True

        out: List[EventMsg] = []

        # 当前在 rest
        if self.cur_label == 0:
            if self.armed and label != 0 and conf >= self.conf_thresh:
                # end 后短时间屏蔽相反动作
                # if (t - self.last_end_t) < self.refractory_sec and label == self.opposite.get(self.last_end_label, -1):
                #   return out

                self.cur_label = label
                self.armed = False
                out.append(EventMsg("event", t, label, self.gesture_names[label], conf, "start"))
            return out

        # 当前在动作中：只允许 end（不允许 action->action 直接切换）
        if label == 0 and self.rest_streak >= self.min_rest_steps:
            out.append(EventMsg("event", t, self.cur_label, self.gesture_names[self.cur_label], conf, "end"))
            self.last_end_t = t
            self.last_end_label = self.cur_label
            self.cur_label = 0
            return out

        return out


# ----------------- 高效 ring buffer -----------------
class RingBuffer:
    def __init__(self, capacity: int, n_channels: int, dtype=np.float32):
        self.capacity = int(capacity)
        self.n_channels = int(n_channels)
        self.buf = np.zeros((self.capacity, self.n_channels), dtype=dtype)
        self.size = 0
        self.w = 0

    def append_block(self, block_tc: np.ndarray) -> None:
        """block_tc: [L,C]"""
        block_tc = np.asarray(block_tc)
        if block_tc.ndim != 2 or block_tc.shape[1] != self.n_channels:
            raise ValueError(f"block shape must be [L,{self.n_channels}] but got {block_tc.shape}")

        L = int(block_tc.shape[0])
        if L <= 0:
            return

        # 若 block 比 capacity 还大，只保留最后 capacity
        if L >= self.capacity:
            block_tc = block_tc[-self.capacity :]
            L = self.capacity

        end = self.w + L
        if end <= self.capacity:
            self.buf[self.w : end] = block_tc
        else:
            k = self.capacity - self.w
            self.buf[self.w :] = block_tc[:k]
            self.buf[: end - self.capacity] = block_tc[k:]

        self.w = end % self.capacity
        self.size = min(self.capacity, self.size + L)

    def get_last(self, n: int, offset: int = 0) -> np.ndarray:
        """取最近窗口。

        n: 窗口长度
        offset: 从“最新样本末尾”往前跳过多少个样本。
                offset=0 表示窗口以最新样本结尾；
                offset=step_len 表示窗口以倒数 step_len 个样本结尾。
        """
        n = int(n)
        offset = int(offset)
        if n <= 0:
            return np.zeros((0, self.n_channels), dtype=self.buf.dtype)
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if self.size < (n + offset):
            raise RuntimeError(f"ringbuffer has only {self.size} samples, need {n+offset}")

        end_excl = (self.w - offset) % self.capacity
        start = (end_excl - n) % self.capacity

        if start < end_excl:
            return self.buf[start:end_excl].copy()

        part1 = self.buf[start:]
        part2 = self.buf[:end_excl]
        return np.vstack([part1, part2]).copy()


class MaskRing:
    """存 0/1 mask，避免 bool ring buffer。"""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buf = np.zeros((self.capacity,), dtype=np.uint8)
        self.size = 0
        self.w = 0

    def append_block(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=np.uint8).reshape(-1)
        L = int(mask.shape[0])
        if L <= 0:
            return
        if L >= self.capacity:
            mask = mask[-self.capacity :]
            L = self.capacity

        end = self.w + L
        if end <= self.capacity:
            self.buf[self.w : end] = mask
        else:
            k = self.capacity - self.w
            self.buf[self.w :] = mask[:k]
            self.buf[: end - self.capacity] = mask[k:]

        self.w = end % self.capacity
        self.size = min(self.capacity, self.size + L)

    def mean_last(self, n: int, offset: int = 0) -> float:
        n = int(n)
        offset = int(offset)
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if self.size < (n + offset):
            raise RuntimeError(f"mask buffer has only {self.size} samples, need {n+offset}")

        end_excl = (self.w - offset) % self.capacity
        start = (end_excl - n) % self.capacity

        if start < end_excl:
            return float(self.buf[start:end_excl].mean())
        part1 = self.buf[start:]
        part2 = self.buf[:end_excl]
        return float(np.concatenate([part1, part2]).mean())


# ----------------- 在线 RMS + 迟滞门控 -----------------
class OnlineGate:
    def __init__(
        self,
        sfreq: float,
        rms_win_sec: float = 0.05,
        base_pct: float = 10.0,
        high_pct: float = 90.0,
        alpha_high: float = 0.8,
        alpha_low: float = 0.35,
        calib_sec: float = 5.0,
    ):
        self.sfreq = float(sfreq)
        self.rms_win = max(1, int(round(rms_win_sec * self.sfreq)))
        self.kernel = np.ones((self.rms_win,), dtype=np.float32) / float(self.rms_win)
        self.tail = np.zeros((self.rms_win - 1,), dtype=np.float32)

        self.base_pct = float(base_pct)
        self.high_pct = float(high_pct)
        self.alpha_high = float(alpha_high)
        self.alpha_low = float(alpha_low)

        self.calib_need = max(1, int(round(calib_sec * self.sfreq)))
        self.calib_vals: List[float] = []

        self.high_th: Optional[float] = None
        self.low_th: Optional[float] = None
        self.active = False

    def thresholds_ready(self) -> bool:
        return self.high_th is not None and self.low_th is not None

    def _maybe_finalize(self):
        if self.thresholds_ready():
            return
        if len(self.calib_vals) < self.calib_need:
            return
        arr = np.asarray(self.calib_vals, dtype=np.float32)
        base = float(np.percentile(arr, self.base_pct))
        high = float(np.percentile(arr, self.high_pct))
        self.high_th = (1.0 - self.alpha_high) * base + self.alpha_high * high
        self.low_th = (1.0 - self.alpha_low) * base + self.alpha_low * high
        print(f"[Gate] calibrated: high_th={self.high_th:.3e}, low_th={self.low_th:.3e}, calib_sec={len(arr)/self.sfreq:.2f}")

    def push_emg_block(self, block_tc: np.ndarray) -> np.ndarray:
        """输入 EMG block [L,C]，输出 mask [L] (0/1)"""
        block_tc = np.asarray(block_tc, dtype=np.float32)
        if block_tc.ndim != 2:
            raise ValueError("block must be 2D [L,C]")

        # power: mean over channels
        power = np.mean(block_tc * block_tc, axis=1).astype(np.float32)  # [L]

        # 平滑（与 offline compute_rms_envelope 等价的 moving average）
        if self.rms_win == 1:
            power_smooth = power
        else:
            ext = np.concatenate([self.tail, power])
            power_smooth = np.convolve(ext, self.kernel, mode="valid").astype(np.float32)
            self.tail = ext[-(self.rms_win - 1) :]

        rms = np.sqrt(np.maximum(power_smooth, 1e-12)).astype(np.float32)

        # 标定阶段：收集 rms
        if not self.thresholds_ready():
            # 不要无限增长
            need = self.calib_need - len(self.calib_vals)
            if need > 0:
                self.calib_vals.extend([float(x) for x in rms[:need]])
            self._maybe_finalize()

        # 门控
        out = np.zeros((rms.shape[0],), dtype=np.uint8)
        if not self.thresholds_ready():
            # 标定没完成前一律认为不活动
            return out

        hi = float(self.high_th)
        lo = float(self.low_th)

        # sample-by-sample hysteresis
        active = self.active
        for i in range(rms.shape[0]):
            v = float(rms[i])
            if not active:
                if v >= hi:
                    active = True
                    out[i] = 1
            else:
                out[i] = 1
                if v <= lo:
                    active = False
        self.active = active
        return out


# ----------------- WebSocket 事件：把每次收到的块对齐后塞到队列 -----------------
class BlockEvent(EventMessage):
    def __init__(self, device_mac: str, channels: List[int], block_q: queue.Queue):
        super().__init__()
        self.device_mac = device_mac
        self.channels = channels
        self.block_q = block_q

    def on_data(self, data: Union[str, dict, DataReceive]):
        if not (isinstance(data, DataReceive) and data.dataType == DataType.EP):
            return

        for device_data in data.data:
            if device_data.deviceId != self.device_mac:
                continue

            # 取出每个通道的 list，并保证长度一致
            vals = []
            min_len = None
            for ch in self.channels:
                key = str(ch)
                if key not in device_data.data:
                    return  # 缺通道就丢弃该包（更安全）
                arr = device_data.data[key]
                if min_len is None:
                    min_len = len(arr)
                else:
                    min_len = min(min_len, len(arr))
                vals.append(arr)

            if not vals or (min_len is None) or min_len <= 0:
                return

            block = np.stack([np.asarray(v[:min_len], dtype=np.float32) for v in vals], axis=1)  # [L,C]

            # 避免队列无限堆积：满了就丢包（实时优先）
            try:
                self.block_q.put_nowait(block)
            except queue.Full:
                pass


def main():
    ap = argparse.ArgumentParser()

    # 连接 / 配置（也支持从 realtime_waveform.py 的 config.ini 读取）
    ap.add_argument("--config", type=str, default=None, help="可选：config.ini 路径（同 realtime_waveform.py）")
    ap.add_argument("--host", type=str, default=None)
    ap.add_argument("--port", type=str, default=None)
    ap.add_argument("--websocket_host", type=str, default=None)
    ap.add_argument("--websocket_port", type=str, default=None)
    ap.add_argument("--login_id", type=str, default=None)
    ap.add_argument("--password", type=str, default=None)
    ap.add_argument("--guinea_pig_id", type=str, default=None)

    ap.add_argument("--device_mac", type=str, default="FB:58:F8:40:9F:1C")
    # 默认通道改为 1,2
    ap.add_argument("--channels", type=str, default="1,2,3,4,5", help="如 1,2,3,4,5；顺序要与训练一致")

    # 采样率（必须正确）
    ap.add_argument("--sfreq", type=float, required=True)

    # 模型
    ap.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "svm", "ensemble"])
    ap.add_argument("--ckpt", type=str, default=None, help="cnn_lru_3instr_0409_1.pth")
    ap.add_argument("--svm", type=str, default=None, help="svm_emg.pkl")
    ap.add_argument("--cnn_weight", type=float, default=0.6)
    ap.add_argument("--svm_weight", type=float, default=0.4)

    # 滑窗与门控
    ap.add_argument("--win_sec", type=float, default=0.20)
    ap.add_argument("--step_sec", type=float, default=0.05)
    ap.add_argument("--active_ratio_thresh", type=float, default=0.6)

    ap.add_argument("--rms_win_sec", type=float, default=0.05)
    ap.add_argument("--base_pct", type=float, default=10.0)
    ap.add_argument("--high_pct", type=float, default=90.0)
    ap.add_argument("--alpha_high", type=float, default=0.8)
    ap.add_argument("--alpha_low", type=float, default=0.3)
    ap.add_argument("--calib_sec", type=float, default=5.0, help="开始几秒用于估计 RMS 阈值")

    # 事件平滑
    ap.add_argument("--hist_len", type=int, default=5)
    ap.add_argument("--switch_ratio", type=float, default=0.7)

    # UDP 输出
    ap.add_argument("--ip", type=str, default="127.0.0.1")
    ap.add_argument("--port_udp", type=int, default=5005)

    # 其它
    ap.add_argument("--queue_max", type=int, default=50, help="WebSocket->分类线程的 block 队列长度")
    ap.add_argument("--ring_sec", type=float, default=10.0, help="ring buffer 保存的秒数")
    ap.add_argument("--scale", type=float, default=1.0, help="对输入幅值整体缩放（调单位用）")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    # ----------------- 读取 config.ini（可选） -----------------
    if args.config:
        cfg = configparser.ConfigParser()
        cfg.read(args.config, encoding="utf-8")

        def _get(section, key, cur):
            if cur is not None:
                return cur
            if cfg.has_option(section, key):
                return cfg.get(section, key)
            return cur

        args.host = _get("config-data", "host", args.host)
        args.port = _get("config-data", "port", args.port)
        args.websocket_host = _get("config-data", "websocket_host", args.websocket_host)
        args.websocket_port = _get("config-data", "websocket_port", args.websocket_port)
        args.login_id = _get("config-data", "login_id", args.login_id)
        args.password = _get("config-data", "password", args.password)
        args.guinea_pig_id = _get("config-data", "guinea_pig_id", args.guinea_pig_id)

    # 默认值
    host = (args.host or "http://localhost").split(";")[0]
    port = str(args.port or "8080")
    websocket_host = (args.websocket_host or "ws://localhost").split(";")[0]
    websocket_port = str(args.websocket_port or "9000")
    login_id = args.login_id or "admin"
    password = args.password or "admin"
    guinea_pig_id = args.guinea_pig_id or "20220526"

    channels = [int(x.strip()) for x in args.channels.split(",") if x.strip()]
    n_channels = len(channels)

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    svm_path = Path(args.svm) if args.svm else None
    if args.model_type in {"cnn", "ensemble"}:
        if ckpt_path is None or (not ckpt_path.exists()):
            raise FileNotFoundError(f"CNN ckpt not found: {ckpt_path}")
    if args.model_type in {"svm", "ensemble"}:
        if svm_path is None or (not svm_path.exists()):
            raise FileNotFoundError(f"SVM file not found: {svm_path}")

    # ----------------- 模型加载 -----------------
    model = None
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = std = None
    per_window_norm = True
    gesture_names = None
    if args.model_type in {"cnn", "ensemble"}:
        model, torch_device, mean, std, per_window_norm, gesture_names = load_ckpt(ckpt_path, n_channels=n_channels)
    svm_model = None
    if args.model_type in {"svm", "ensemble"}:
        svm_model, svm_class_names = load_svm(svm_path)
        if gesture_names is None:
            gesture_names = ["rest"] + list(svm_class_names)
    smoother = TapEventSmoother(
        gesture_names,
        step_sec=args.step_sec,
        min_rest_sec=0.03,
        refractory_sec=0.12,
        conf_thresh=0.55,
    )

    print(f"[Model] type={args.model_type}, device={torch_device}, per_window_norm={per_window_norm}")
    print(f"[Classes] 0=rest, 1..={gesture_names[1:]}")

    # ----------------- UDP socket -----------------
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.ip, int(args.port_udp))
    print(f"[UDP] send to {target[0]}:{target[1]}")

    # ----------------- 在线门控 + ring buffer -----------------
    sfreq = float(args.sfreq)
    win_len = int(round(args.win_sec * sfreq))
    step_len = int(round(args.step_sec * sfreq))
    if win_len <= 0 or step_len <= 0:
        raise ValueError("win_len/step_len <= 0，检查 --sfreq/--win_sec/--step_sec")

    ring_cap = int(round(args.ring_sec * sfreq))
    ring_cap = max(ring_cap, win_len * 2)

    emg_ring = RingBuffer(capacity=ring_cap, n_channels=n_channels, dtype=np.float32)
    mask_ring = MaskRing(capacity=ring_cap)

    gate = OnlineGate(
        sfreq=sfreq,
        rms_win_sec=args.rms_win_sec,
        base_pct=args.base_pct,
        high_pct=args.high_pct,
        alpha_high=args.alpha_high,
        alpha_low=args.alpha_low,
        calib_sec=args.calib_sec,
    )

    # ----------------- WebSocket / collection 启动 -----------------
    block_q: queue.Queue = queue.Queue(maxsize=int(args.queue_max))
    event_handler = BlockEvent(device_mac=args.device_mac, channels=channels, block_q=block_q)

    client = EpClient(init_user_status=True)
    login_res = client.do_action_json(UserLoginRequest(login_id, password))
    print("[EPStudio] Login:", login_res)

    set_patient_res = client.do_action_json(GuineaPigSetCurrentRequest(guinea_pig_id))
    print("[EPStudio] Set patient:", set_patient_res)

    websocket_client = EpWebSocketClient(event_msg=event_handler, enable_trace=False)
    websocket_client.start()

    collection_device = CollectionDeviceBean(args.device_mac, channelStatus=channels)
    collection = Collection(client=client, device_list=[collection_device], record_status=False)

    print("[EPStudio] Starting collection...")
    result = collection.start_collection()
    print("[EPStudio] start_collection:", result)

    # ----------------- 主循环：消费 block -> 更新 ring -> 每 step_len 做一次推理 -----------------
    samples_since_pred = 0
    first_pred_done = False
    total_samples = 0  # 从开始累计的样本点数（用于构造 t）

    try:
        while True:
            try:
                block = block_q.get(timeout=0.5)  # [L,C]
            except queue.Empty:
                continue

            if args.scale != 1.0:
                block = (np.asarray(block, dtype=np.float32) * float(args.scale)).astype(np.float32)
            else:
                block = np.asarray(block, dtype=np.float32)

            L = int(block.shape[0])
            if L <= 0:
                continue

            # 1) 更新门控 mask
            m = gate.push_emg_block(block)  # [L]

            # 2) 更新 ring buffer
            emg_ring.append_block(block)
            mask_ring.append_block(m)

            total_samples += L
            samples_since_pred += L

            # ring 不够窗口长度就先不推理
            if emg_ring.size < win_len or mask_ring.size < win_len:
                continue
            if not first_pred_done:
                samples_since_pred = step_len  # 让下面 while 恰好跑一次，并且 offset=0
                first_pred_done = True
            # 3) 可能需要做多次推理（如果一下来了很多样本）
            #    用 offset 把“该预测对应的窗口末端”对齐到正确的样本位置，避免重复用同一个 last window。
            while samples_since_pred >= step_len:
                offset = samples_since_pred - step_len

                emg_win = emg_ring.get_last(win_len, offset=offset)  # [win_len,C]
                active_ratio = mask_ring.mean_last(win_len, offset=offset)

                # 以样本计数构造时间（更稳定）
                end_sample_excl = total_samples - offset
                t_center = (end_sample_excl - win_len / 2.0) / sfreq

                if active_ratio < float(args.active_ratio_thresh) or (not gate.thresholds_ready()):
                    pred_idx = 0
                    conf = 1.0
                else:
                    if args.model_type == "cnn":
                        prob = infer_cnn_prob(model, torch_device, emg_win, per_window_norm, mean, std)
                    elif args.model_type == "svm":
                        prob = infer_svm_prob(svm_model, emg_win)
                    else:
                        p_cnn = infer_cnn_prob(model, torch_device, emg_win, per_window_norm, mean, std)
                        p_svm = infer_svm_prob(svm_model, emg_win)
                        prob = args.cnn_weight * p_cnn + args.svm_weight * p_svm
                        prob = prob / (np.sum(prob) + 1e-12)
                    pred4 = int(np.argmax(prob))
                    conf = float(np.max(prob))
                    pred_idx = pred4 + 1  # 模型输出 0/1 → 全局标签 1/2

                # frame
                frame = FrameMsg("frame", float(t_center), int(pred_idx), gesture_names[int(pred_idx)], float(conf), float(active_ratio))
                sock.sendto(json.dumps(frame.__dict__, ensure_ascii=False).encode("utf-8"), target)

                # event
                evs = smoother.update(t=float(t_center), label=int(pred_idx), conf=float(conf))
                for ev in evs:
                    sock.sendto(json.dumps(ev.__dict__, ensure_ascii=False).encode("utf-8"), target)

                if args.debug:
                    print(f"[frame] t={t_center:.2f} label={pred_idx} name={gesture_names[pred_idx]} conf={conf:.2f} active_ratio={active_ratio:.2f}")
                    print(f"[event] state={ev.state} label={ev.label} name={ev.name} conf={ev.conf:.2f}")
                samples_since_pred -= step_len

    except KeyboardInterrupt:
        print("\n[Exit] KeyboardInterrupt")
    finally:
        try:
            collection.stop_collection()
        except Exception:
            pass
        try:
            websocket_client.stop()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
        print("[Done]")


if __name__ == "__main__":
    main()