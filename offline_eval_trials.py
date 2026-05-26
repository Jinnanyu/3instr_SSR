# -*- coding: utf-8 -*-
"""Offline evaluation for EMG trials supporting CNN / SVM / ensemble."""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from features_emg import extract_features_window

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise RuntimeError("This script requires PyTorch.") from e


DEFAULT_CLASS_NAMES = ["left", "right"]


class EMGCNN(nn.Module):
    def __init__(self, n_channels: int, n_classes: int = 2):
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
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def prf_from_cm(cm: np.ndarray):
    out = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        support = cm[k, :].sum()
        out.append((prec, rec, f1, support))
    return out


def load_cnn(ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    class_names = list(ckpt.get("class_names", DEFAULT_CLASS_NAMES))
    n_classes = len(class_names)
    w0 = None
    for k, v in state.items():
        if k.endswith("feature_extractor.0.weight"):
            w0 = v
            break
    if w0 is None:
        raise RuntimeError("Cannot infer CNN input channels.")
    n_channels = int(w0.shape[1])
    model = EMGCNN(n_channels=n_channels, n_classes=n_classes)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    mean = ckpt.get("mean", None)
    std = ckpt.get("std", None)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1) if mean is not None else None
    std = np.asarray(std, dtype=np.float32).reshape(1, -1) if std is not None else None
    return {
        "type": "cnn",
        "model": model,
        "device": device,
        "class_names": class_names,
        "n_channels": n_channels,
        "per_window_norm": bool(ckpt.get("per_window_norm", True)),
        "mean": mean,
        "std": std,
    }


def load_svm(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        class_names = list(obj.get("class_names", DEFAULT_CLASS_NAMES))
    else:
        model = obj
        class_names = DEFAULT_CLASS_NAMES
    return {"type": "svm", "model": model, "class_names": class_names}


def infer_cnn_prob(cnn_obj, w_tc: np.ndarray) -> np.ndarray:
    w = w_tc.astype(np.float32)
    if cnn_obj["per_window_norm"]:
        mu = w.mean(axis=0, keepdims=True)
        sigma = w.std(axis=0, keepdims=True) + 1e-8
        w = (w - mu) / sigma
    else:
        mean = cnn_obj["mean"]
        std = cnn_obj["std"]
        if mean is None or std is None:
            mu = w.mean(axis=0, keepdims=True)
            sigma = w.std(axis=0, keepdims=True) + 1e-8
            w = (w - mu) / sigma
        else:
            w = (w - mean) / (std + 1e-8)
    x = torch.from_numpy(w.T[None, :, :]).to(cnn_obj["device"])
    with torch.no_grad():
        logits = cnn_obj["model"](x).detach().cpu().numpy()[0]
    return softmax_np(logits, axis=0).astype(np.float32)


def infer_svm_prob(svm_obj, w_tc: np.ndarray) -> np.ndarray:
    feat, _ = extract_features_window(w_tc)
    return svm_obj["model"].predict_proba(feat[None, :])[0].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_dir", type=str, required=True)
    ap.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "svm", "ensemble"])
    ap.add_argument("--ckpt", type=str, default=None, help="cnn checkpoint .pth")
    ap.add_argument("--svm", type=str, default=None, help="svm pickle .pkl")
    ap.add_argument("--cnn_weight", type=float, default=0.6)
    ap.add_argument("--svm_weight", type=float, default=0.4)
    ap.add_argument("--win_sec", type=float, default=0.20)
    ap.add_argument("--step_sec", type=float, default=0.05)
    ap.add_argument("--trim_head_sec", type=float, default=0.20)
    ap.add_argument("--trim_tail_sec", type=float, default=0.20)
    ap.add_argument("--vote", type=str, default="majority", choices=["majority", "meanprob"])
    ap.add_argument("--use_file_sfreq", action="store_true", default=True)
    ap.add_argument("--sfreq", type=float, default=1000.0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--print_limit", type=int, default=200)
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    files = sorted(session_dir.glob("trial_*.npz"))
    if not files:
        raise FileNotFoundError(f"no trial_*.npz in {session_dir}")

    dev = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if dev == "auto":
        dev = "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"

    cnn_obj = load_cnn(Path(args.ckpt), dev) if args.model_type in {"cnn", "ensemble"} else None
    svm_obj = load_svm(Path(args.svm)) if args.model_type in {"svm", "ensemble"} else None
    if args.model_type in {"cnn", "ensemble"} and cnn_obj is None:
        raise ValueError("--ckpt is required for cnn/ensemble")
    if args.model_type in {"svm", "ensemble"} and svm_obj is None:
        raise ValueError("--svm is required for svm/ensemble")

    class_names = cnn_obj["class_names"] if cnn_obj is not None else svm_obj["class_names"]
    n_classes = len(class_names)

    y_true, y_pred, rows = [], [], []
    printed = 0
    for fp in files:
        obj = np.load(fp, allow_pickle=True)
        emg = obj["emg"].astype(np.float32)
        sfreq = float(obj["sfreq"]) if (args.use_file_sfreq and "sfreq" in obj) else float(args.sfreq)
        lab = int(obj["label_id"]) if "label_id" in obj else -1
        lab_name = str(obj["label_name"]) if "label_name" in obj else (class_names[lab] if 0 <= lab < n_classes else "unknown")
        if emg.ndim != 2:
            continue
        if cnn_obj is not None and emg.shape[1] != cnn_obj["n_channels"]:
            print(f"[skip] {fp.name}: channels mismatch file_C={emg.shape[1]} ckpt_C={cnn_obj['n_channels']}")
            continue

        T = emg.shape[0]
        s = min(max(int(round(args.trim_head_sec * sfreq)), 0), T)
        e = max(s, T - max(int(round(args.trim_tail_sec * sfreq)), 0))
        emg2 = emg[s:e, :]
        dur = emg2.shape[0] / sfreq if sfreq > 0 else 0.0
        win_len = int(round(args.win_sec * sfreq))
        step_len = int(round(args.step_sec * sfreq))
        if win_len <= 4 or step_len <= 0 or emg2.shape[0] < win_len:
            print(f"[skip] {fp.name}: too short for windowing (dur={dur:.2f}s)")
            continue

        probs, preds = [], []
        for start in range(0, emg2.shape[0] - win_len + 1, step_len):
            w = emg2[start:start + win_len, :]
            if args.model_type == "cnn":
                p = infer_cnn_prob(cnn_obj, w)
            elif args.model_type == "svm":
                p = infer_svm_prob(svm_obj, w)
            else:
                p_cnn = infer_cnn_prob(cnn_obj, w)
                p_svm = infer_svm_prob(svm_obj, w)
                p = args.cnn_weight * p_cnn + args.svm_weight * p_svm
                p = p / (np.sum(p) + 1e-12)
            probs.append(p)
            preds.append(int(np.argmax(p)))

        probs = np.asarray(probs, dtype=np.float32)
        preds = np.asarray(preds, dtype=np.int64)
        if args.vote == "meanprob":
            mean_p = probs.mean(axis=0)
            pred = int(np.argmax(mean_p))
            conf = float(np.max(mean_p))
        else:
            binc = np.bincount(preds, minlength=n_classes)
            pred = int(np.argmax(binc))
            conf = float(binc[pred] / (preds.size + 1e-12))

        y_true.append(lab)
        y_pred.append(pred)
        rows.append({
            "file": fp.name, "true_id": lab, "true_name": lab_name,
            "pred_id": pred, "pred_name": class_names[pred], "confidence": conf,
            "sfreq": sfreq, "dur_sec": dur, "n_windows": int(preds.size), "model_type": args.model_type,
        })
        if printed < args.print_limit:
            ok = "OK" if pred == lab else "X"
            print(f"[{ok}] {fp.name} true={lab}({lab_name}) pred={pred}({class_names[pred]}) conf={conf:.3f} dur={dur:.2f}s windows={preds.size}")
            printed += 1

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    print("\n=== Summary ===")
    print("Trials:", y_true.size, "Accuracy:", f"{acc:.3f}")
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    print("Confusion (rows=true, cols=pred):\n", cm)
    for k, (p, r, f1, sup) in enumerate(prf_from_cm(cm)):
        print(f"  {k:>2d} {class_names[k]:<8s}  P={p:.3f} R={r:.3f} F1={f1:.3f}  n={int(sup)}")

    out_csv = Path(args.out_csv) if args.out_csv else session_dir / f"eval_results_{args.model_type}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n[OK] wrote:", out_csv)


if __name__ == "__main__":
    main()
