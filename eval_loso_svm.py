# -*- coding: utf-8 -*-
"""Leave-One-Subject-Out cross-subject SVM evaluation for 3-command EMG.

Default subjects match your current records folder:
- jny: records/session_20260507_merge_jny
- wjw: records/session_20260421_merge_wjw
- zjh: records/session_20260424_merge_zjh

The script supports three normalization settings:
1) --norm none
   Use raw windows/features directly.
2) --norm train_zscore
   Fit mean/std on TRAIN subjects' raw windows only, then apply to train/test.
   This is strict LOSO and avoids leakage.
3) --norm subject_p95
   Approximate MVC normalization: divide each subject/channel by that subject's
   95th percentile absolute amplitude. If you later collect true MVC trials,
   replace this with true MVC values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features_emg import extract_features_batch, feature_names_per_channel

CLASS_NAMES = ["left", "right", "up"]
DEFAULT_SUBJECTS = {
    "jny": r".\records\session_merge_jny",
    "wjw": r".\records\session_merge_wjw",
    "zjh": r".\records\session_merge_zjh",
    "zly": r".\records\session_merge_zly",
}
FEATURE_GROUPS = {
    "rms": ["rms"],
    "amp": ["rms", "iemg", "mav", "var", "logdetector"],
    "shape": ["wl", "zc", "ssc", "wamp", "dasdv", "mavs"],
    "freq": ["mnf", "mdf"],
    "all": feature_names_per_channel(),
}


def parse_subjects(items: List[str]) -> Dict[str, str]:
    if not items:
        return DEFAULT_SUBJECTS.copy()
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--subject format must be name=path, e.g. --subject jny=records/session_xxx")
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def load_subject(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x_win = path / "X_single_windows.npy"
    y_file = path / "y_single_labels.npy"
    if not x_win.exists():
        raise FileNotFoundError(f"{x_win} not found. Run build_windows_from_trials.py first.")
    if not y_file.exists():
        raise FileNotFoundError(f"{y_file} not found. Run build_windows_from_trials.py first.")
    X = np.load(x_win).astype(np.float32)  # [N,T,C]
    y = np.load(y_file).astype(np.int64)
    return X, y


def subject_p95_normalize(X: np.ndarray) -> np.ndarray:
    scale = np.percentile(np.abs(X.reshape(-1, X.shape[-1])), 95, axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-8)
    return X / scale.reshape(1, 1, -1)


def train_zscore_fit(X_train_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    merged = np.concatenate([x.reshape(-1, x.shape[-1]) for x in X_train_list], axis=0)
    mean = merged.mean(axis=0).astype(np.float32)
    std = merged.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-8)
    return mean, std


def apply_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)


def select_feature_group(X_feat: np.ndarray, group: str, n_channels: int) -> Tuple[np.ndarray, List[str]]:
    all_per_ch = feature_names_per_channel()
    wanted = FEATURE_GROUPS[group]
    idx = []
    names = []
    for c in range(n_channels):
        base = c * len(all_per_ch)
        for j, name in enumerate(all_per_ch):
            if name in wanted:
                idx.append(base + j)
                names.append(f"ch{c+1}_{name}")
    return X_feat[:, idx], names


def make_clf(kind: str, C: float, gamma: str, kernel: str, seed: int):
    if kind == "svm":
        return Pipeline([
            ("feature_scaler", StandardScaler()),
            ("svc", SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced")),
        ])
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced_subsample", n_jobs=-1)
    raise ValueError(f"unknown clf={kind}")


def plot_cm(cm: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", action="append", default=[], help="name=path; can repeat. Default uses jny/wjw/zjh/zly records.")
    ap.add_argument("--norm", default="train_zscore", choices=["none", "train_zscore", "subject_p95"])
    ap.add_argument("--feature_group", default="all", choices=list(FEATURE_GROUPS.keys()))
    ap.add_argument("--clf", default="svm", choices=["svm", "rf"])
    ap.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "poly", "sigmoid"])
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--sfreq", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results_loso")
    args = ap.parse_args()

    subjects = {k: Path(v) for k, v in parse_subjects(args.subject).items()}
    out_dir = Path(args.out_dir) / f"norm-{args.norm}_feat-{args.feature_group}_clf-{args.clf}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = {name: load_subject(path) for name, path in subjects.items()}
    rows = []
    all_true = []
    all_pred = []

    for test_name in subjects.keys():
        train_names = [s for s in subjects.keys() if s != test_name]
        X_train_raw_list = [raw[s][0] for s in train_names]
        y_train_list = [raw[s][1] for s in train_names]
        X_test_raw, y_test = raw[test_name]

        if args.norm == "none":
            X_train_norm_list = X_train_raw_list
            X_test_norm = X_test_raw
        elif args.norm == "train_zscore":
            mean, std = train_zscore_fit(X_train_raw_list)
            X_train_norm_list = [apply_zscore(x, mean, std) for x in X_train_raw_list]
            X_test_norm = apply_zscore(X_test_raw, mean, std)
        elif args.norm == "subject_p95":
            X_train_norm_list = [subject_p95_normalize(x) for x in X_train_raw_list]
            X_test_norm = subject_p95_normalize(X_test_raw)
        else:
            raise ValueError(args.norm)

        X_train_feat_list = []
        for x in X_train_norm_list:
            feat, _ = extract_features_batch(x, sfreq=args.sfreq)
            feat, feature_names = select_feature_group(feat, args.feature_group, n_channels=x.shape[-1])
            X_train_feat_list.append(feat)
        X_train = np.concatenate(X_train_feat_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        X_test_feat, _ = extract_features_batch(X_test_norm, sfreq=args.sfreq)
        X_test, feature_names = select_feature_group(X_test_feat, args.feature_group, n_channels=X_test_raw.shape[-1])

        clf = make_clf(args.clf, args.C, args.gamma, args.kernel, args.seed)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])

        print(f"\n=== LOSO test={test_name}; train={'+'.join(train_names)} ===")
        print("accuracy:", f"{acc:.4f}")
        print("cm:\n", cm)
        print(classification_report(y_test, pred, labels=[0,1,2], target_names=CLASS_NAMES, digits=4, zero_division=0))

        plot_cm(cm, f"LOSO test={test_name} acc={acc:.3f}", out_dir / f"cm_test_{test_name}.png")
        rows.append({
            "test_subject": test_name,
            "train_subjects": "+".join(train_names),
            "accuracy": acc,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "norm": args.norm,
            "feature_group": args.feature_group,
            "clf": args.clf,
        })
        all_true.append(y_test)
        all_pred.append(pred)

    y_all = np.concatenate(all_true)
    p_all = np.concatenate(all_pred)
    cm_all = confusion_matrix(y_all, p_all, labels=[0, 1, 2])
    acc_all = accuracy_score(y_all, p_all)
    plot_cm(cm_all, f"LOSO overall acc={acc_all:.3f}", out_dir / "cm_overall.png")

    rows.append({
        "test_subject": "MEAN_OF_ROUNDS",
        "train_subjects": "-",
        "accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "n_train": "-",
        "n_test": "-",
        "norm": args.norm,
        "feature_group": args.feature_group,
        "clf": args.clf,
    })
    rows.append({
        "test_subject": "POOLED_ALL_WINDOWS",
        "train_subjects": "-",
        "accuracy": float(acc_all),
        "n_train": "-",
        "n_test": int(len(y_all)),
        "norm": args.norm,
        "feature_group": args.feature_group,
        "clf": args.clf,
    })

    pd.DataFrame(rows).to_csv(out_dir / "loso_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature_name": feature_names}).to_csv(out_dir / "used_feature_names.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print("\n[OK] results saved to:", out_dir)


if __name__ == "__main__":
    main()
