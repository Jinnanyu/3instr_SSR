# -*- coding: utf-8 -*-
"""
Quick visual QA: show per-sample RMS and window start markers for random trials.

Usage:
python inspect_trial_windows.py --session_dir .\records\session_YYYYmmdd_HHMMSS --n 5
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def smooth_ma(x, k):
    if k <= 1:
        return x
    w = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, w, mode="same").astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_dir", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--win_sec", type=float, default=0.20)
    ap.add_argument("--step_sec", type=float, default=0.05)
    ap.add_argument("--trim_head_sec", type=float, default=0.20)
    ap.add_argument("--trim_tail_sec", type=float, default=0.20)
    ap.add_argument("--rms_smooth_sec", type=float, default=0.02)
    ap.add_argument("--thr_q", type=float, default=0.55)
    args = ap.parse_args()

    session = Path(args.session_dir)
    files = sorted(session.glob("trial_*.npz"))
    if not files:
        raise FileNotFoundError("no trial_*.npz")

    rng = np.random.default_rng(0)
    pick = rng.choice(files, size=min(args.n, len(files)), replace=False)

    for fp in pick:
        obj = np.load(fp, allow_pickle=True)
        emg = obj["emg"].astype(np.float32)
        sfreq = float(obj["sfreq"])
        lab = int(obj.get("label_id", -1))
        name = str(obj.get("label_name", ""))

        T = emg.shape[0]
        t = np.arange(T) / sfreq

        head = int(round(args.trim_head_sec * sfreq))
        tail = int(round(args.trim_tail_sec * sfreq))
        s = head
        e = max(s, T - tail)

        win_len = int(round(args.win_sec * sfreq))
        step_len = int(round(args.step_sec * sfreq))

        rms = np.sqrt(np.mean(emg**2, axis=1)).astype(np.float32)
        k = int(round(args.rms_smooth_sec * sfreq))
        rms_s = smooth_ma(rms, max(k, 1))
        thr = float(np.quantile(rms_s[s:e], np.clip(args.thr_q, 0.0, 1.0)))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(t, rms_s, linewidth=1.0)
        plt.axvline(t[s], linestyle="--")
        plt.axvline(t[e-1], linestyle="--")
        plt.axhline(thr, linestyle=":")

        if e - s >= win_len:
            for st in range(s, e - win_len + 1, step_len):
                plt.axvline(t[st], alpha=0.12)

        plt.title(f"{fp.name}  label={lab}({name})  dur={T/sfreq:.2f}s")
        plt.xlabel("time (s)")
        plt.ylabel("RMS (smoothed)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
