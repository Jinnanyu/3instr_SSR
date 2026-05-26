# -*- coding: utf-8 -*-
"""Feature visualization for cross-subject EMG analysis.

Outputs:
- feature_boxplots/: boxplots for selected feature types
- pca_by_label.png: PCA colored by command
- pca_by_subject.png: PCA colored by subject
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from features_emg import extract_features_batch, feature_names_per_channel
from eval_loso_svm import DEFAULT_SUBJECTS, CLASS_NAMES, parse_subjects, load_subject, subject_p95_normalize


def plot_pca(df_xy: pd.DataFrame, color_col: str, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for key, sub in df_xy.groupby(color_col):
        ax.scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.65, label=str(key))
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", action="append", default=[], help="name=path; can repeat. Default uses jny/wjw/zjh/zly records.")
    ap.add_argument("--norm", default="subject_p95", choices=["none", "subject_p95"])
    ap.add_argument("--sfreq", type=float, default=1000.0)
    ap.add_argument("--max_points", type=int, default=4000, help="subsample for PCA/plots to keep images readable")
    ap.add_argument("--out_dir", default="results_feature_vis")
    args = ap.parse_args()

    subjects = {k: Path(v) for k, v in parse_subjects(args.subject).items()}
    out_dir = Path(args.out_dir) / f"norm-{args.norm}"
    box_dir = out_dir / "feature_boxplots"
    box_dir.mkdir(parents=True, exist_ok=True)

    Xs, ys, ss = [], [], []
    for name, path in subjects.items():
        X_raw, y = load_subject(path)
        X_norm = subject_p95_normalize(X_raw) if args.norm == "subject_p95" else X_raw
        X_feat, feature_names = extract_features_batch(X_norm, sfreq=args.sfreq)
        Xs.append(X_feat)
        ys.append(y)
        ss.extend([name] * len(y))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subjects_arr = np.asarray(ss)
    labels_arr = np.asarray([CLASS_NAMES[int(i)] if int(i) < len(CLASS_NAMES) else str(i) for i in y])

    rng = np.random.default_rng(42)
    idx = np.arange(len(y))
    if len(idx) > args.max_points:
        idx = rng.choice(idx, size=args.max_points, replace=False)

    # PCA
    X_scaled = StandardScaler().fit_transform(X[idx])
    xy = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    df_xy = pd.DataFrame({
        "PC1": xy[:, 0],
        "PC2": xy[:, 1],
        "subject": subjects_arr[idx],
        "label": labels_arr[idx],
    })
    plot_pca(df_xy, "label", out_dir / "pca_by_label.png", "PCA of EMG features - colored by command")
    plot_pca(df_xy, "subject", out_dir / "pca_by_subject.png", "PCA of EMG features - colored by subject")

    # boxplots by feature type: average over channels to make figures readable
    per_ch = feature_names_per_channel()
    n_channels = len(feature_names) // len(per_ch)
    for feat_type in ["rms", "mav", "wl", "zc", "ssc", "wamp", "mnf", "mdf"]:
        col_idx = []
        for c in range(n_channels):
            j = per_ch.index(feat_type)
            col_idx.append(c * len(per_ch) + j)
        vals = X[:, col_idx].mean(axis=1)
        df = pd.DataFrame({"value": vals, "subject": subjects_arr, "label": labels_arr})
        fig, ax = plt.subplots(figsize=(8, 4))
        groups = []
        ticklabels = []
        for subj in sorted(df["subject"].unique()):
            for lab in CLASS_NAMES:
                groups.append(df[(df["subject"] == subj) & (df["label"] == lab)]["value"].values)
                ticklabels.append(f"{subj}\n{lab}")
        ax.boxplot(groups, showfliers=False)
        ax.set_title(f"{feat_type.upper()} distribution across subjects and commands")
        ax.set_ylabel(feat_type)
        ax.set_xticklabels(ticklabels, rotation=0, fontsize=8)
        fig.tight_layout()
        fig.savefig(box_dir / f"box_{feat_type}.png", dpi=200)
        plt.close(fig)

    pd.DataFrame({"feature_name": feature_names}).to_csv(out_dir / "feature_names.csv", index=False, encoding="utf-8-sig")
    print("[OK] visualization saved to:", out_dir)


if __name__ == "__main__":
    main()
