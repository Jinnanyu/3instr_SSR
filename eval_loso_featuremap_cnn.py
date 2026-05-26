# -*- coding: utf-8 -*-
"""
eval_loso_featuremap_cnn.py

Feature-map CNN for cross-subject / LOSO facial EMG command recognition.

This script reads existing:
    X_features.npy
    y_single_labels.npy
    feature_names.csv

Then converts each window's feature vector into a small feature map:
    [num_features, num_channels]

Example:
              ch1   ch2   ch3   ch4   ch5
    RMS       ...   ...   ...   ...   ...
    MAV       ...   ...   ...   ...   ...
    WL        ...   ...   ...   ...   ...
    ZC        ...   ...   ...   ...   ...
    SSC       ...   ...   ...   ...   ...
    ...

The feature map is treated as a 1-channel image:
    [N, 1, num_features, num_channels]

Then a lightweight 2D-CNN is trained and evaluated by LOSO:
    train: two subjects
    test : one held-out subject

Author: generated for cross-subject facial EMG graduation project.
"""

import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


CLASS_NAMES = ["left", "right", "up"]


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Data loading
# -----------------------------
def read_feature_names(path: Path):
    if not path.exists():
        return None

    # Handles both:
    #   one feature per row
    #   one comma-separated row
    names = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 1 and len(rows[0]) > 1:
        names = [x.strip() for x in rows[0] if x.strip()]
    else:
        for row in rows:
            if not row:
                continue
            # If csv has a header like "feature_name", skip it.
            item = row[0].strip()
            if item.lower() in {"feature", "feature_name", "name", "features"}:
                continue
            if item:
                names.append(item)
    return names if names else None


def load_subject(subject_name: str, subject_dir: Path):
    x_path = subject_dir / "X_features.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing {x_path}. Please run build_windows_from_trials.py with --save_features first.")

    y_candidates = [
        subject_dir / "y_single_labels.npy",
        subject_dir / "y_labels.npy",
        subject_dir / "y.npy",
    ]
    y_path = None
    for p in y_candidates:
        if p.exists():
            y_path = p
            break
    if y_path is None:
        raise FileNotFoundError(f"Missing labels in {subject_dir}. Expected y_single_labels.npy / y_labels.npy / y.npy.")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    names = read_feature_names(subject_dir / "feature_names.csv")

    if X.ndim != 2:
        raise ValueError(f"{x_path} should be [N, D], got {X.shape}.")
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch for {subject_name}: X={len(X)}, y={len(y)}")

    return X, y, names


# -----------------------------
# Feature-name parsing
# -----------------------------
def parse_channel_from_name(name: str):
    """
    Supports common formats:
        rms_ch0, rms_ch1
        ch0_rms, ch1_rms
        rms_c0, c0_rms
        channel0_rms, rms_channel0
        rms-0 / rms_0 as weak fallback
    Returns channel integer or None.
    """
    s = name.lower()

    patterns = [
        r"(?:^|[_\-\s])ch(?:annel)?[_\-\s]?(\d+)(?:$|[_\-\s])",
        r"(?:^|[_\-\s])c[_\-\s]?(\d+)(?:$|[_\-\s])",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return int(m.group(1))

    # Weak fallback: final number, e.g. rms_0
    m = re.search(r"(?:_|-)(\d+)$", s)
    if m:
        return int(m.group(1))

    return None


def strip_channel_token(name: str):
    s = name.lower()
    # remove common channel tokens
    s = re.sub(r"(?:^|[_\-\s])ch(?:annel)?[_\-\s]?\d+(?:$|[_\-\s])", "_", s)
    s = re.sub(r"(?:^|[_\-\s])c[_\-\s]?\d+(?:$|[_\-\s])", "_", s)
    s = re.sub(r"(?:_|-)\d+$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s


FEATURE_GROUPS = {
    "rms": ["rms"],
    "amp": ["rms", "iemg", "mav", "var", "log"],
    "shape": ["wl", "zc", "ssc", "wamp", "dasdv", "mavs"],
    "freq": ["mnf", "mdf", "psr"],
    "shape_freq": ["wl", "zc", "ssc", "wamp", "dasdv", "mavs", "mnf", "mdf", "psr"],
    "all": None,
}


def selected_by_group(base_name: str, group: str):
    if group == "all":
        return True
    keys = FEATURE_GROUPS[group]
    b = base_name.lower()
    return any(k in b for k in keys)


def desired_feature_order(feature_names):
    priority = [
        "rms", "iemg", "mav", "var", "log",
        "wl", "zc", "ssc", "wamp", "dasdv", "mavs",
        "mnf", "mdf", "psr",
    ]

    def key_fn(x):
        x_low = x.lower()
        for i, k in enumerate(priority):
            if k == x_low or k in x_low:
                return (i, x_low)
        return (len(priority), x_low)

    return sorted(feature_names, key=key_fn)


def build_feature_maps_from_names(X: np.ndarray, names, group: str, num_channels_hint: int = 5):
    """
    Build [N, F, C] maps using feature_names.csv.
    """
    if names is None or len(names) != X.shape[1]:
        return None, None, None

    entries = []
    for idx, name in enumerate(names):
        ch = parse_channel_from_name(name)
        base = strip_channel_token(name)
        if ch is None or not base:
            continue
        if not selected_by_group(base, group):
            continue
        entries.append((idx, base, ch, name))

    if not entries:
        return None, None, None

    channels = sorted({e[2] for e in entries})

    # Handle 1-based channels if names are ch1..ch5.
    # We preserve original labels for plotting but map to 0..C-1 internally.
    if channels and min(channels) == 1 and 0 not in channels:
        ch_to_col = {ch: i for i, ch in enumerate(channels)}
    else:
        ch_to_col = {ch: i for i, ch in enumerate(channels)}

    features = desired_feature_order(sorted({e[1] for e in entries}))

    # If parsing produced too many channels from weak suffixes, restrict to hint if possible.
    if len(channels) > num_channels_hint:
        channels = channels[:num_channels_hint]
        ch_to_col = {ch: i for i, ch in enumerate(channels)}

    feat_to_row = {f: i for i, f in enumerate(features)}
    fmap = np.zeros((X.shape[0], len(features), len(channels)), dtype=np.float32)
    filled = np.zeros((len(features), len(channels)), dtype=np.int32)

    for idx, base, ch, name in entries:
        if base not in feat_to_row or ch not in ch_to_col:
            continue
        r = feat_to_row[base]
        c = ch_to_col[ch]
        fmap[:, r, c] = X[:, idx]
        filled[r, c] += 1

    # Remove feature rows that are entirely empty.
    keep_rows = np.where(filled.sum(axis=1) > 0)[0]
    fmap = fmap[:, keep_rows, :]
    features = [features[i] for i in keep_rows]

    channel_labels = [f"ch{ch}" for ch in channels]
    return fmap, features, channel_labels


def build_feature_maps_fallback(X: np.ndarray, group: str, layout: str, num_channels: int):
    """
    Fallback when feature_names.csv cannot be parsed.

    layout:
        feature_major: [f1_ch1, f1_ch2, ... f1_chC, f2_ch1, ...]
        channel_major: [ch1_f1, ch1_f2, ... ch2_f1, ...]
        auto: defaults to feature_major
    """
    D = X.shape[1]
    if D % num_channels != 0:
        raise ValueError(
            f"Cannot reshape feature vector length D={D} into num_channels={num_channels}. "
            f"Please provide feature_names.csv or set --num_channels correctly."
        )

    F_count = D // num_channels
    if layout == "auto":
        layout = "feature_major"

    if layout == "feature_major":
        fmap = X.reshape(X.shape[0], F_count, num_channels)
    elif layout == "channel_major":
        fmap = X.reshape(X.shape[0], num_channels, F_count).transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown fallback layout: {layout}")

    feature_names = [f"feat_{i:02d}" for i in range(F_count)]
    channel_labels = [f"ch{i}" for i in range(num_channels)]

    if group != "all":
        print("[WARN] feature_names.csv was unavailable/unparseable; feature_group cannot be applied accurately.")
        print("[WARN] Fallback uses all reshaped features.")

    return fmap.astype(np.float32), feature_names, channel_labels


def make_subject_feature_map(subject_name, X, names, group, num_channels, fallback_layout):
    fmap, feature_names, channel_labels = build_feature_maps_from_names(
        X, names, group=group, num_channels_hint=num_channels
    )
    if fmap is None:
        print(f"[WARN] {subject_name}: feature_names.csv could not be parsed; using fallback reshape.")
        fmap, feature_names, channel_labels = build_feature_maps_fallback(
            X, group=group, layout=fallback_layout, num_channels=num_channels
        )

    if fmap.ndim != 3:
        raise ValueError(f"Feature map must be [N, F, C], got {fmap.shape}")

    return fmap.astype(np.float32), feature_names, channel_labels


# -----------------------------
# Normalization
# -----------------------------
def normalize_train_zscore(X_train, X_val, X_test, eps=1e-6):
    """
    X_*: [N, F, C]
    Fit only on training samples for each feature-map cell.
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + eps
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, {
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
    }


def normalize_sample_zscore(X, eps=1e-6):
    """
    Per-sample feature map z-score.
    Use carefully; it removes overall energy information.
    """
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True) + eps
    return (X - mean) / std


# -----------------------------
# Dataset/model
# -----------------------------
class FeatureMapDataset(Dataset):
    def __init__(self, X, y):
        # [N, F, C] -> [N, 1, F, C]
        self.X = torch.tensor(X[:, None, :, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FeatureMapCNN(nn.Module):
    def __init__(self, num_classes=3, dropout=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        return self.fc(x)


def make_loaders(X_train, y_train, X_val, y_val, batch_size, num_workers=0):
    train_ds = FeatureMapDataset(X_train, y_train)
    val_ds = FeatureMapDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, val_loader


def run_one_epoch(model, loader, optimizer, device, criterion, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(yb)
            pred = logits.argmax(dim=1)
            total_correct += (pred == yb).sum().item()
            total_count += len(yb)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def predict(model, X, batch_size, device):
    ds = FeatureMapDataset(X, np.zeros(len(X), dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


# -----------------------------
# Plotting
# -----------------------------
def plot_confusion_matrix(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_curve(values1, values2, out_path, ylabel, title, label1="train", label2="val"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values1, label=label1)
    ax.plot(values2, label=label2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mean_feature_map(X, y, feature_names, channel_labels, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        idx = np.where(y == cls_id)[0]
        if len(idx) == 0:
            continue
        mean_map = X[idx].mean(axis=0)

        fig_h = max(4, 0.35 * len(feature_names))
        fig, ax = plt.subplots(figsize=(6, fig_h))
        im = ax.imshow(mean_map, aspect="auto")
        ax.set_title(f"Mean feature map: {cls_name}")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Feature")
        ax.set_xticks(range(len(channel_labels)))
        ax.set_xticklabels(channel_labels, rotation=0)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"mean_feature_map_{cls_name}.png", dpi=180)
        plt.close(fig)


# -----------------------------
# Main LOSO
# -----------------------------
def split_calibration_data(X, y, calib_ratio, seed):
    if calib_ratio <= 0:
        return None, None, X, y
    X_calib, X_test, y_calib, y_test = train_test_split(
        X, y, train_size=calib_ratio, random_state=seed, stratify=y
    )
    return X_calib, y_calib, X_test, y_test


def run_loso(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Device] {device}")

    subject_dirs = {
        "jny": Path(args.jny_dir),
        "wjw": Path(args.wjw_dir),
        "zjh": Path(args.zjh_dir),
        "zly": Path(args.zly_dir),
    }

    raw_subjects = {}
    for s, d in subject_dirs.items():
        X, y, names = load_subject(s, d)
        fmap, feature_names, channel_labels = make_subject_feature_map(
            s, X, names, args.feature_group, args.num_channels, args.fallback_layout
        )
        raw_subjects[s] = {"X": fmap, "y": y}
        print(f"[Load] {s}: X_features={X.shape} -> fmap={fmap.shape}, y={y.shape}")

    # Basic consistency check.
    shapes = {tuple(v["X"].shape[1:]) for v in raw_subjects.values()}
    if len(shapes) != 1:
        raise ValueError(
            f"Feature map shapes differ across subjects: {shapes}. "
            f"Check feature_names.csv and feature extraction consistency."
        )

    map_shape = list(next(iter(shapes)))
    exp_name = (
        f"featmap_group-{args.feature_group}_norm-{args.norm}"
        f"_calib-{args.calib_ratio:.2f}_shape-{map_shape[0]}x{map_shape[1]}"
    )
    out_root = Path(args.out_dir) / exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_true = []
    all_pred = []

    for test_subj in raw_subjects.keys():
        train_subjs = [s for s in raw_subjects.keys() if s != test_subj]
        print("\n" + "=" * 80)
        print(f"LOSO Feature-map CNN test={test_subj}; train={'+'.join(train_subjs)}; group={args.feature_group}; norm={args.norm}; calib={args.calib_ratio}")
        print("=" * 80)

        X_train_base = np.concatenate([raw_subjects[s]["X"] for s in train_subjs], axis=0)
        y_train_base = np.concatenate([raw_subjects[s]["y"] for s in train_subjs], axis=0)

        X_test_subject = raw_subjects[test_subj]["X"]
        y_test_subject = raw_subjects[test_subj]["y"]

        X_calib, y_calib, X_test_final, y_test_final = split_calibration_data(
            X_test_subject, y_test_subject, args.calib_ratio, args.seed
        )
        if args.calib_ratio > 0:
            X_train_base = np.concatenate([X_train_base, X_calib], axis=0)
            y_train_base = np.concatenate([y_train_base, y_calib], axis=0)
            print(f"[Calibration] added {len(y_calib)} samples from {test_subj}; final test={len(y_test_final)}")

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_base, y_train_base,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=y_train_base,
        )

        if args.norm == "none":
            norm_info = {"mode": "none"}
        elif args.norm == "train_zscore":
            X_train, X_val, X_test_final, norm_info = normalize_train_zscore(X_train, X_val, X_test_final)
            norm_info["mode"] = "train_zscore"
        elif args.norm == "sample_zscore":
            X_train = normalize_sample_zscore(X_train)
            X_val = normalize_sample_zscore(X_val)
            X_test_final = normalize_sample_zscore(X_test_final)
            norm_info = {"mode": "sample_zscore"}
        else:
            raise ValueError(f"Unknown norm: {args.norm}")

        run_dir = out_root / f"test-{test_subj}_train-{'+'.join(train_subjs)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        plot_mean_feature_map(X_train, y_train, feature_names, channel_labels, run_dir)

        train_loader, val_loader = make_loaders(
            X_train, y_train, X_val, y_val, batch_size=args.batch_size, num_workers=args.num_workers
        )

        model = FeatureMapCNN(num_classes=len(CLASS_NAMES), dropout=args.dropout).to(device)

        # Class weights may help if a new dataset is imbalanced.
        counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_acc = -1.0
        best_state = None
        no_improve = 0
        hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_one_epoch(model, train_loader, optimizer, device, criterion, train=True)
            val_loss, val_acc = run_one_epoch(model, val_loader, optimizer, device, criterion, train=False)

            hist["train_loss"].append(train_loss)
            hist["train_acc"].append(train_acc)
            hist["val_loss"].append(val_loss)
            hist["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:03d}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        y_pred = predict(model, X_test_final, args.batch_size, device)
        acc = accuracy_score(y_test_final, y_pred)
        macro_f1 = f1_score(y_test_final, y_pred, average="macro")
        cm = confusion_matrix(y_test_final, y_pred, labels=list(range(len(CLASS_NAMES))))
        report = classification_report(
            y_test_final, y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )

        print(f"test_acc={acc:.4f}, macro_f1={macro_f1:.4f}")
        print("cm:\n", cm)
        print(report)

        with open(run_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n")
            f.write(f"accuracy={acc:.6f}\nmacro_f1={macro_f1:.6f}\n")
            f.write(f"cm=\n{cm}\n")

        plot_confusion_matrix(cm, run_dir / "confusion_matrix.png", f"Feature-map CNN test={test_subj}")
        plot_curve(hist["train_loss"], hist["val_loss"], run_dir / "loss_curve.png", "Loss", f"Loss test={test_subj}")
        plot_curve(hist["train_acc"], hist["val_acc"], run_dir / "accuracy_curve.png", "Accuracy", f"Accuracy test={test_subj}")

        torch.save(model.state_dict(), run_dir / "best_model.pth")

        info = {
            "test_subject": test_subj,
            "train_subjects": train_subjs,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "map_shape": map_shape,
            "feature_names": feature_names,
            "channel_labels": channel_labels,
            "norm_info": norm_info,
            "args": vars(args),
        }
        with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        row = {
            "test_subject": test_subj,
            "train_subjects": "+".join(train_subjs),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
        }
        for i, cname in enumerate(CLASS_NAMES):
            support = cm[i].sum()
            recall = cm[i, i] / support if support > 0 else 0.0
            pred_count = cm[:, i].sum()
            precision = cm[i, i] / pred_count if pred_count > 0 else 0.0
            row[f"{cname}_recall"] = float(recall)
            row[f"{cname}_precision"] = float(precision)
            row[f"{cname}_support"] = int(support)
            row[f"pred_{cname}_count"] = int(pred_count)

        all_results.append(row)
        all_true.append(y_test_final)
        all_pred.append(y_pred)

    # Save summary csv/json.
    csv_path = out_root / "loso_featuremap_cnn_results.csv"
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(CLASS_NAMES))))
    plot_confusion_matrix(cm_all, out_root / "confusion_matrix_overall.png", "Feature-map CNN overall")

    summary = {
        "avg_accuracy": float(np.mean([r["accuracy"] for r in all_results])),
        "avg_macro_f1": float(np.mean([r["macro_f1"] for r in all_results])),
        "results": all_results,
        "map_shape": map_shape,
        "feature_names": feature_names,
        "channel_labels": channel_labels,
        "args": vars(args),
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] saved results to: {out_root}")
    print(f"[SUMMARY] avg_accuracy={summary['avg_accuracy']:.4f}, avg_macro_f1={summary['avg_macro_f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Feature-map CNN LOSO for facial EMG features.")

    parser.add_argument("--jny_dir", type=str, default=r".\records\session_merge_jny")
    parser.add_argument("--wjw_dir", type=str, default=r".\records\session_merge_wjw")
    parser.add_argument("--zjh_dir", type=str, default=r".\records\session_merge_zjh")
    parser.add_argument("--zly_dir", type=str, default=r".\records\session_merge_zly")

    parser.add_argument("--feature_group", type=str, default="all",
                        choices=["rms", "amp", "shape", "freq", "shape_freq", "all"],
                        help="Which feature rows to keep in the feature map.")

    parser.add_argument("--num_channels", type=int, default=5,
                        help="Fallback channel count if feature_names.csv cannot be parsed.")

    parser.add_argument("--fallback_layout", type=str, default="feature_major",
                        choices=["auto", "feature_major", "channel_major"],
                        help="Only used when feature_names.csv cannot be parsed.")

    parser.add_argument("--norm", type=str, default="train_zscore",
                        choices=["none", "train_zscore", "sample_zscore"],
                        help="Feature-map normalization.")

    parser.add_argument("--calib_ratio", type=float, default=0.0,
                        help="Optional target-subject calibration ratio. 0.1 means use 10%% target samples for training.")

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_dir", type=str, default=r".\results_loso_featuremap_cnn")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_loso(args)
