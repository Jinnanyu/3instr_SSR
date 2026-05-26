# -*- coding: utf-8 -*-
"""
eval_loso_svm_calib.py

SVM LOSO with target-subject calibration.

Default:
    python eval_loso_svm_calib.py --feature_group all --calib_ratio 0.1 --out_dir .\results_loso_svm_calib

Meaning:
    train = other subjects + 10% target subject data
    test  = remaining 90% target subject data
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASS_NAMES = ["left", "right", "up"]

FEATURE_GROUPS = {
    "rms": ["rms"],
    "amp": ["rms", "iemg", "mav", "var", "log"],
    "shape": ["wl", "zc", "ssc", "wamp", "dasdv", "mavs"],
    "freq": ["mnf", "mdf", "psr", "meanfreq", "medianfreq"],
    "shape_freq": ["wl", "zc", "ssc", "wamp", "dasdv", "mavs", "mnf", "mdf", "psr", "meanfreq", "medianfreq"],
    "all": None,
}


def read_feature_names(path: Path, dim: int):
    if not path.exists():
        return [f"feat_{i}" for i in range(dim)]

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    names = []
    if len(rows) == 1 and len(rows[0]) > 1:
        names = [x.strip() for x in rows[0] if x.strip()]
    else:
        for row in rows:
            if not row:
                continue
            item = row[0].strip()
            if item.lower() in {"feature", "feature_name", "name", "features"}:
                continue
            if item:
                names.append(item)

    if len(names) != dim:
        print(f"[WARN] feature_names length {len(names)} != dim {dim}, fallback names used.")
        return [f"feat_{i}" for i in range(dim)]
    return names


def load_subject(name: str, folder: Path):
    x_path = folder / "X_features.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing {x_path}. Please run build_windows_from_trials.py --save_features first.")

    y_path = None
    for p in [folder / "y_single_labels.npy", folder / "y_labels.npy", folder / "y.npy"]:
        if p.exists():
            y_path = p
            break
    if y_path is None:
        raise FileNotFoundError(f"Missing labels in {folder}.")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    if X.ndim != 2:
        raise ValueError(f"{x_path} should be [N,D], got {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"{name}: X/y length mismatch: X={len(X)}, y={len(y)}")

    names = read_feature_names(folder / "feature_names.csv", X.shape[1])
    print(f"[{name}] X_features={X.shape}, y={y.shape}, counts={np.bincount(y, minlength=3).tolist()}")
    return X, y, names


def strip_channel_token(name: str):
    s = name.lower()
    s = re.sub(r"(?:^|[_\-\s])ch(?:annel)?[_\-\s]?\d+(?:$|[_\-\s])", "_", s)
    s = re.sub(r"(?:^|[_\-\s])c[_\-\s]?\d+(?:$|[_\-\s])", "_", s)
    s = re.sub(r"(?:_|-)\d+$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def get_feature_group_indices(feature_names, group):
    if group == "all":
        return np.arange(len(feature_names), dtype=int)

    keys = FEATURE_GROUPS[group]
    idx = []
    for i, name in enumerate(feature_names):
        base = strip_channel_token(name)
        if any(k in base for k in keys):
            idx.append(i)

    if not idx:
        raise ValueError(f"No features matched feature_group={group}. Check feature_names.csv.")
    return np.array(idx, dtype=int)


def split_target_calibration(X, y, calib_ratio, seed):
    if calib_ratio <= 0:
        return None, None, X, y

    return train_test_split(
        X,
        y,
        train_size=calib_ratio,
        random_state=seed,
        stratify=y,
    )


def build_svm(args):
    class_weight = None if args.class_weight == "none" else args.class_weight
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=args.C, gamma=args.gamma, class_weight=class_weight)),
    ])


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


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args):
    subject_dirs = {
        "jny": Path(args.jny_dir),
        "wjw": Path(args.wjw_dir),
        "zjh": Path(args.zjh_dir),
        "zly": Path(args.zly_dir),
    }

    raw = {}
    common_names = None

    for name, folder in subject_dirs.items():
        X, y, names = load_subject(name, folder)
        if common_names is None:
            common_names = names
        elif len(names) != len(common_names):
            raise ValueError(f"Feature dimension mismatch for {name}.")
        raw[name] = {"X_raw": X, "y": y}

    group_idx = get_feature_group_indices(common_names, args.feature_group)
    selected_names = [common_names[i] for i in group_idx]

    data = {}
    for name, d in raw.items():
        data[name] = {"X": d["X_raw"][:, group_idx], "y": d["y"]}

    exp_name = (
        f"svm_calib_feat-{args.feature_group}_calib-{args.calib_ratio:.2f}"
        f"_C{args.C}_gamma{args.gamma}_cw{args.class_weight}_seed{args.seed}"
    )
    out_root = Path(args.out_dir) / exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "feature_names_used.txt", "w", encoding="utf-8") as f:
        for n in selected_names:
            f.write(n + "\n")

    print(f"[Feature group] {args.feature_group}: D={len(selected_names)}")
    print(f"[Calibration ratio] {args.calib_ratio}")
    print(f"[Output] {out_root}")

    rows = []
    all_true = []
    all_pred = []

    for test_subj in subject_dirs.keys():
        train_subjs = [s for s in subject_dirs.keys() if s != test_subj]

        print("\n" + "=" * 80)
        print(f"LOSO + calibration test={test_subj}; train={'+'.join(train_subjs)} + {test_subj}({args.calib_ratio*100:.1f}%)")
        print("=" * 80)

        X_train_others = np.concatenate([data[s]["X"] for s in train_subjs], axis=0)
        y_train_others = np.concatenate([data[s]["y"] for s in train_subjs], axis=0)

        X_target = data[test_subj]["X"]
        y_target = data[test_subj]["y"]

        split = split_target_calibration(X_target, y_target, args.calib_ratio, args.seed)
        if args.calib_ratio > 0:
            X_calib, X_test, y_calib, y_test = split
            X_train = np.concatenate([X_train_others, X_calib], axis=0)
            y_train = np.concatenate([y_train_others, y_calib], axis=0)
            calib_counts = np.bincount(y_calib, minlength=len(CLASS_NAMES)).tolist()
        else:
            X_calib, y_calib, X_test, y_test = None, None, X_target, y_target
            X_train, y_train = X_train_others, y_train_others
            calib_counts = [0] * len(CLASS_NAMES)

        train_counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).tolist()
        test_counts = np.bincount(y_test, minlength=len(CLASS_NAMES)).tolist()

        print(f"[Calibration] n={0 if y_calib is None else len(y_calib)}, counts={calib_counts}")
        print(f"[Train] X={X_train.shape}, counts={train_counts}")
        print(f"[Test]  X={X_test.shape}, counts={test_counts}")

        model = build_svm(args)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
        report = classification_report(
            y_test, y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )

        print(f"accuracy: {acc:.4f}, macro_f1: {macro_f1:.4f}")
        print("cm:\n", cm)
        print(report)

        run_dir = out_root / f"test-{test_subj}_train-{'+'.join(train_subjs)}_calib-{test_subj}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n")
            f.write(f"accuracy={acc:.6f}\nmacro_f1={macro_f1:.6f}\ncm=\n{cm}\n")
            f.write(f"calib_counts={calib_counts}\n")
            f.write(f"test_counts={test_counts}\n")
            f.write(f"train_counts={train_counts}\n")

        plot_confusion_matrix(cm, run_dir / "confusion_matrix.png", f"SVM calib test={test_subj}")

        row = {
            "test_subject": test_subj,
            "train_subjects": "+".join(train_subjs),
            "calib_ratio": args.calib_ratio,
            "calib_n": 0 if y_calib is None else int(len(y_calib)),
            "test_n": int(len(y_test)),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "feature_group": args.feature_group,
            "C": args.C,
            "gamma": args.gamma,
            "class_weight": args.class_weight,
        }

        for i, cname in enumerate(CLASS_NAMES):
            support = cm[i].sum()
            recall = cm[i, i] / support if support > 0 else 0.0
            pred_count = cm[:, i].sum()
            precision = cm[i, i] / pred_count if pred_count > 0 else 0.0
            row[f"{cname}_recall"] = float(recall)
            row[f"{cname}_precision"] = float(precision)
            row[f"{cname}_support"] = int(support)
            row[f"calib_{cname}_count"] = int(calib_counts[i])
            row[f"test_{cname}_count"] = int(test_counts[i])
            row[f"pred_{cname}_count"] = int(pred_count)

        rows.append(row)
        all_true.append(y_test)
        all_pred.append(y_pred)

    save_csv(out_root / "loso_svm_calib_results.csv", rows)

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(CLASS_NAMES))))
    plot_confusion_matrix(cm_all, out_root / "confusion_matrix_overall.png", "SVM calibration overall")

    summary = {
        "feature_group": args.feature_group,
        "calib_ratio": args.calib_ratio,
        "C": args.C,
        "gamma": args.gamma,
        "class_weight": args.class_weight,
        "avg_accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "avg_macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
        "results": rows,
        "args": vars(args),
    }

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] saved results to: {out_root}")
    print(f"[SUMMARY] avg_accuracy={summary['avg_accuracy']:.4f}, avg_macro_f1={summary['avg_macro_f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="SVM LOSO with target subject calibration.")

    parser.add_argument("--jny_dir", type=str, default=r".\records\session_merge_jny")
    parser.add_argument("--wjw_dir", type=str, default=r".\records\session_merge_wjw")
    parser.add_argument("--zjh_dir", type=str, default=r".\records\session_merge_zjh")
    parser.add_argument("--zly_dir", type=str, default=r".\records\session_merge_zly")

    parser.add_argument("--feature_group", type=str, default="all",
                        choices=["rms", "amp", "shape", "freq", "shape_freq", "all"])
    parser.add_argument("--calib_ratio", type=float, default=0.1)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--class_weight", type=str, default="none", choices=["none", "balanced"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=r".\results_loso_svm_calib")

    args = parser.parse_args()

    if args.gamma not in {"scale", "auto"}:
        args.gamma = float(args.gamma)

    if not (0 <= args.calib_ratio < 1):
        raise ValueError("--calib_ratio must be in [0, 1).")

    return args


if __name__ == "__main__":
    run(parse_args())
