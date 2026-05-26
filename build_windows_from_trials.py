# -*- coding: utf-8 -*-
"""
Build fixed-length training windows from per-trial NPZ recordings (with activity filtering).

This is a stricter version of build_windows_from_trials.py:
- trims head/tail
- computes per-sample RMS (across channels) + smoothing
- builds a boolean "active" mask using a per-trial RMS threshold (quantile)
- keeps only windows whose active_ratio >= active_ratio_thresh

Why:
Your original build_windows_from_trials.py labels *every* window in a trial as the trial label,
so transition/rest segments inside the trial become noisy labels.

Input:
- session folder containing trial_*.npz from realtime_record_trials.py
  Each npz:
    emg: [T,C], sfreq, label_id, label_name(optional)

Output:
- X_single_windows.npy : [N, win_len, C]
- y_single_labels.npy  : [N]
- gesture_to_id_single.csv
- windows_index.csv (optional but recommended): per-window traceability & QA

Recommended start:
python build_windows_from_trials_filtered.py --in_dir .\records\session_YYYYmmdd_HHMMSS --win_sec 0.20 --step_sec 0.05 ^
  --trim_head_sec 0.30 --trim_tail_sec 0.30 ^
  --rms_smooth_sec 0.02 --thr_q 0.55 --active_ratio_thresh 0.65 --write_index
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features_emg import extract_features_window


LABELS = {
    0: "left",
    1: "right",
    2: "up",
}


def smooth_ma(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    w = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, w, mode="same").astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="session folder containing trial_*.npz")
    ap.add_argument("--out_dir", type=str, default=None, help="output folder; default=in_dir")
    ap.add_argument("--win_sec", type=float, default=0.30)
    ap.add_argument("--step_sec", type=float, default=0.05)
    ap.add_argument("--trim_head_sec", type=float, default=0.20)
    ap.add_argument("--trim_tail_sec", type=float, default=0.20)
    ap.add_argument("--min_trial_sec", type=float, default=0.60, help="drop trials shorter than this after trimming")

    # filtering knobs
    ap.add_argument("--rms_smooth_sec", type=float, default=0.02, help="RMS smoothing (seconds)")
    ap.add_argument("--thr_q", type=float, default=0.55,
                    help="per-trial RMS threshold quantile in [0,1]. mask = rms >= quantile(rms, thr_q)")
    ap.add_argument("--active_ratio_thresh", type=float, default=0.65,
                    help="keep a window only if mean(active_mask) >= this")
    ap.add_argument("--min_window_rms_q", type=float, default=0.00,
                    help="optional extra filter: keep window only if window_mean_rms >= quantile(window_mean_rms, q). "
                         "Set 0 to disable, e.g. 0.2 keeps top 80%%.")
    ap.add_argument("--write_index", action="store_true", help="write windows_index.csv for QA/traceability")
    ap.add_argument("--save_features", action="store_true", help="also save handcrafted features for SVM/RF training")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("trial_*.npz"))
    if not files:
        raise FileNotFoundError(f"no trial_*.npz found in {in_dir}")

    X_list = []
    y_list = []
    X_feat_list = []
    feature_names = None
    idx_rows = []

    drop_stats = {k: {"kept": 0, "dropped": 0} for k in LABELS.keys()}

    for fp in files:
        obj = np.load(fp, allow_pickle=True)
        emg = obj["emg"].astype(np.float32)  # [T,C]
        sfreq = float(obj["sfreq"])
        lab = int(obj["label_id"])

        if lab not in LABELS:
            print(f"[skip] {fp.name}: unknown label_id={lab}")
            continue

        T = emg.shape[0]
        head = int(round(args.trim_head_sec * sfreq))
        tail = int(round(args.trim_tail_sec * sfreq))
        s = head
        e = max(s, T - tail)
        emg2 = emg[s:e, :]
        if emg2.shape[0] < int(round(args.min_trial_sec * sfreq)):
            print(f"[skip] {fp.name}: too short after trim: {emg2.shape[0]/sfreq:.2f}s")
            continue

        win_len = int(round(args.win_sec * sfreq))
        step_len = int(round(args.step_sec * sfreq))
        if win_len <= 0 or step_len <= 0:
            raise ValueError("bad win/step")

        # --- activity mask (per-sample) ---
        rms = np.sqrt(np.mean(emg2 ** 2, axis=1)).astype(np.float32)  # [T2]
        k = int(round(args.rms_smooth_sec * sfreq))
        rms_s = smooth_ma(rms, max(k, 1))

        thr = float(np.quantile(rms_s, np.clip(args.thr_q, 0.0, 1.0)))
        active = (rms_s >= thr).astype(np.float32)  # [T2]

        # precompute window stats
        win_starts = list(range(0, emg2.shape[0] - win_len + 1, step_len))
        if not win_starts:
            print(f"[skip] {fp.name}: too short for windowing")
            continue

        win_mean_rms = np.array([float(rms_s[st:st + win_len].mean()) for st in win_starts], dtype=np.float32)
        if args.min_window_rms_q > 0:
            q = float(np.clip(args.min_window_rms_q, 0.0, 1.0))
            win_rms_thr = float(np.quantile(win_mean_rms, q))
        else:
            win_rms_thr = -np.inf

        for i, st in enumerate(win_starts):
            act_ratio = float(active[st:st + win_len].mean())
            mean_r = float(win_mean_rms[i])

            keep = (act_ratio >= args.active_ratio_thresh) and (mean_r >= win_rms_thr)
            if keep:
                w = emg2[st:st + win_len, :]  # [win_len, C]
                X_list.append(w)
                y_list.append(lab)
                if args.save_features:
                    feat, cur_names = extract_features_window(w)
                    X_feat_list.append(feat)
                    if feature_names is None:
                        feature_names = cur_names
                drop_stats[lab]["kept"] += 1
            else:
                drop_stats[lab]["dropped"] += 1

            if args.write_index:
                idx_rows.append({
                    "trial_file": fp.name,
                    "label_id": lab,
                    "label_name": LABELS[lab],
                    "sfreq": sfreq,
                    "trim_head_sec": args.trim_head_sec,
                    "trim_tail_sec": args.trim_tail_sec,
                    "win_sec": args.win_sec,
                    "step_sec": args.step_sec,
                    "win_start_sample_in_trim": int(st),
                    "win_end_sample_in_trim": int(st + win_len),
                    "active_ratio": act_ratio,
                    "mean_rms": mean_r,
                    "thr_q": args.thr_q,
                    "thr_value": thr,
                    "keep": int(keep),
                })

    if not X_list:
        raise RuntimeError("no windows built (try lowering active_ratio_thresh or thr_q)")

    X = np.stack(X_list, axis=0).astype(np.float32)  # [N,T,C]
    y = np.asarray(y_list, dtype=np.int64)

    np.save(out_dir / "X_single_windows.npy", X)
    np.save(out_dir / "y_single_labels.npy", y)

    if args.save_features:
        X_feat = np.stack(X_feat_list, axis=0).astype(np.float32)
        np.save(out_dir / "X_features.npy", X_feat)
        pd.DataFrame({"feature_name": feature_names}).to_csv(out_dir / "feature_names.csv", index=False, encoding="utf-8-sig")

    s = pd.Series({LABELS[k]: k for k in sorted(LABELS.keys())})
    s.to_csv(out_dir / "gesture_to_id_single.csv", header=False, encoding="utf-8-sig")

    if args.write_index and idx_rows:
        pd.DataFrame(idx_rows).to_csv(out_dir / "windows_index.csv", index=False, encoding="utf-8-sig")

    print("[OK] saved:", out_dir / "X_single_windows.npy", X.shape)
    print("[OK] saved:", out_dir / "y_single_labels.npy", y.shape)
    if args.save_features:
        print("[OK] saved:", out_dir / "X_features.npy", X_feat.shape)
        print("[OK] saved:", out_dir / "feature_names.csv")
    print("[OK] saved:", out_dir / "gesture_to_id_single.csv")
    if args.write_index:
        print("[OK] saved:", out_dir / "windows_index.csv")

    print("\n=== Window keep/drop stats ===")
    for k in sorted(LABELS.keys()):
        kept = drop_stats[k]["kept"]
        dropped = drop_stats[k]["dropped"]
        total = kept + dropped
        rate = kept / total if total > 0 else 0.0
        print(f"  label {k}({LABELS[k]}): kept={kept} dropped={dropped} keep_rate={rate:.2f}")


if __name__ == "__main__":
    main()
