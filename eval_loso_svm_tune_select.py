# -*- coding: utf-8 -*-
"""
eval_loso_svm_tune_select.py

Nested LOSO SVM with hyperparameter search + feature selection.

目的：
1. 在规范动作数据上进一步优化当前最强的 SVM-all / SVM-shape。
2. 避免数据泄露：外层测试被试永远不参与调参。
3. 外层 LOSO:
      test = 1 subject
      train = other subjects
4. 内层 LOSO:
      在外层训练被试内部，再留 1 个训练被试做验证，用于选择：
        - SVM C
        - SVM gamma
        - class_weight
        - feature selection method / k

读取：
    X_features.npy
    y_single_labels.npy
    feature_names.csv

默认被试：
    records/session_merge_jny
    records/session_merge_wjw
    records/session_merge_zjh
    records/session_merge_zly

推荐先跑：
    python eval_loso_svm_tune_select.py --feature_group all --select_mode kbest --score macro_f1 --out_dir .\results_loso_svm_tuned

再跑：
    python eval_loso_svm_tune_select.py --feature_group shape --select_mode kbest --score macro_f1 --out_dir .\results_loso_svm_tuned
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASS_NAMES = ["left", "right", "up"]


# -----------------------------
# Loading
# -----------------------------
def read_feature_names(path: Path, dim: int):
    if not path.exists():
        return [f"feat_{i}" for i in range(dim)]

    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

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
        print(f"[WARN] feature_names length {len(names)} != dim {dim}, using fallback names.")
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
        raise FileNotFoundError(f"Missing labels in {folder}")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    if X.ndim != 2:
        raise ValueError(f"{x_path} should be [N,D], got {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"{name}: X/y length mismatch: {len(X)} vs {len(y)}")

    feature_names = read_feature_names(folder / "feature_names.csv", X.shape[1])
    print(f"[{name}] X_features={X.shape}, y={y.shape}, counts={np.bincount(y, minlength=3).tolist()}")

    return X, y, feature_names


# -----------------------------
# Feature groups
# -----------------------------
FEATURE_GROUPS = {
    "rms": ["rms"],
    "amp": ["rms", "iemg", "mav", "var", "log"],
    "shape": ["wl", "zc", "ssc", "wamp", "dasdv", "mavs"],
    "freq": ["mnf", "mdf", "psr", "meanfreq", "medianfreq"],
    "shape_freq": ["wl", "zc", "ssc", "wamp", "dasdv", "mavs", "mnf", "mdf", "psr", "meanfreq", "medianfreq"],
    "all": None,
}


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


# -----------------------------
# Optional feature selector that supports passthrough and top-k
# -----------------------------
class IdentitySelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X

    def get_support(self):
        return np.ones(self.n_features_in_, dtype=bool)


class SafeSelectKBest(SelectKBest):
    """
    SelectKBest with k clipped to feature dimension.
    """
    def fit(self, X, y=None):
        if isinstance(self.k, int):
            self.k = min(self.k, X.shape[1])
        return super().fit(X, y)


def build_selector(method, k, random_state=42):
    if method == "none":
        return IdentitySelector()
    if method == "f_classif":
        return SafeSelectKBest(score_func=f_classif, k=k)
    if method == "mutual_info":
        def mi(X, y):
            return mutual_info_classif(X, y, random_state=random_state)
        return SafeSelectKBest(score_func=mi, k=k)
    raise ValueError(f"Unknown selector method: {method}")


def build_pipeline(selector_method, k, C, gamma, class_weight, random_state=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("selector", build_selector(selector_method, k, random_state=random_state)),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=False,
        )),
    ])


def score_predictions(y_true, y_pred, score_name):
    if score_name == "accuracy":
        return accuracy_score(y_true, y_pred)
    if score_name == "macro_f1":
        return f1_score(y_true, y_pred, average="macro")
    raise ValueError(f"Unknown score: {score_name}")


# -----------------------------
# Nested LOSO tuning
# -----------------------------
def inner_loso_score(subject_data, train_subjs, params, score_name, random_state):
    """
    train_subjs are the subjects available inside outer training set.
    For each inner validation subject:
        inner_train = train_subjs - val_subj
        inner_val   = val_subj
    """
    scores = []

    for val_subj in train_subjs:
        inner_train_subjs = [s for s in train_subjs if s != val_subj]

        X_inner_train = np.concatenate([subject_data[s]["X"] for s in inner_train_subjs], axis=0)
        y_inner_train = np.concatenate([subject_data[s]["y"] for s in inner_train_subjs], axis=0)

        X_inner_val = subject_data[val_subj]["X"]
        y_inner_val = subject_data[val_subj]["y"]

        pipe = build_pipeline(
            selector_method=params["selector_method"],
            k=params["k"],
            C=params["C"],
            gamma=params["gamma"],
            class_weight=params["class_weight"],
            random_state=random_state,
        )
        pipe.fit(X_inner_train, y_inner_train)
        pred = pipe.predict(X_inner_val)
        scores.append(score_predictions(y_inner_val, pred, score_name))

    return float(np.mean(scores)), scores


def make_param_grid(args, n_features):
    if args.select_mode == "none":
        selector_methods = ["none"]
        k_values = ["all"]
    elif args.select_mode == "kbest":
        selector_methods = args.selector_methods.split(",")
        raw_k = [int(x) for x in args.k_list.split(",") if x.strip()]
        k_values = sorted(set([min(k, n_features) for k in raw_k if k > 0]))
    else:
        raise ValueError(f"Unknown select_mode: {args.select_mode}")

    C_values = [float(x) for x in args.C_list.split(",") if x.strip()]
    gamma_values = []
    for x in args.gamma_list.split(","):
        x = x.strip()
        if not x:
            continue
        if x in {"scale", "auto"}:
            gamma_values.append(x)
        else:
            gamma_values.append(float(x))

    class_weights = []
    for x in args.class_weight_list.split(","):
        x = x.strip().lower()
        if x in {"none", "null", "no"}:
            class_weights.append(None)
        elif x in {"balanced", "balance"}:
            class_weights.append("balanced")
        else:
            raise ValueError(f"Unknown class_weight option: {x}")

    grid = []
    for selector_method in selector_methods:
        for k in k_values:
            for C in C_values:
                for gamma in gamma_values:
                    for cw in class_weights:
                        grid.append({
                            "selector_method": selector_method,
                            "k": k,
                            "C": C,
                            "gamma": gamma,
                            "class_weight": cw,
                        })
    return grid


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


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -----------------------------
# Main
# -----------------------------
def run(args):
    subject_dirs = {
        "jny": Path(args.jny_dir),
        "wjw": Path(args.wjw_dir),
        "zjh": Path(args.zjh_dir),
        "zly": Path(args.zly_dir),
    }

    raw = {}
    common_feature_names = None
    for name, folder in subject_dirs.items():
        X, y, names = load_subject(name, folder)
        if common_feature_names is None:
            common_feature_names = names
        elif len(names) != len(common_feature_names):
            raise ValueError(f"Feature dimension mismatch for {name}.")

        raw[name] = {"X_raw": X, "y": y, "feature_names": names}

    group_idx = get_feature_group_indices(common_feature_names, args.feature_group)
    selected_names = [common_feature_names[i] for i in group_idx]

    subject_data = {}
    for name, d in raw.items():
        subject_data[name] = {
            "X": d["X_raw"][:, group_idx],
            "y": d["y"],
        }

    n_features = len(selected_names)

    exp_name = f"nested_svm_feat-{args.feature_group}_select-{args.select_mode}_score-{args.score}_D{n_features}"
    out_root = Path(args.out_dir) / exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[Feature group] {args.feature_group}: D={n_features}")
    print(f"[Output] {out_root}")

    param_grid = make_param_grid(args, n_features=n_features)
    print(f"[Grid] {len(param_grid)} candidates")

    outer_rows = []
    all_true = []
    all_pred = []

    for test_subj in subject_dirs.keys():
        train_subjs = [s for s in subject_dirs.keys() if s != test_subj]

        print("\n" + "=" * 80)
        print(f"OUTER LOSO test={test_subj}; train={'+'.join(train_subjs)}")
        print("=" * 80)

        tune_rows = []
        best_params = None
        best_score = -1.0
        best_inner_scores = None

        for idx, params in enumerate(param_grid, start=1):
            mean_score, inner_scores = inner_loso_score(
                subject_data,
                train_subjs=train_subjs,
                params=params,
                score_name=args.score,
                random_state=args.seed,
            )

            row = {
                "candidate": idx,
                "test_subject": test_subj,
                "selector_method": params["selector_method"],
                "k": params["k"],
                "C": params["C"],
                "gamma": params["gamma"],
                "class_weight": str(params["class_weight"]),
                "inner_mean_score": mean_score,
            }
            for s, sc in zip(train_subjs, inner_scores):
                row[f"inner_val_{s}"] = sc
            tune_rows.append(row)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_inner_scores = inner_scores

        run_dir = out_root / f"test-{test_subj}_train-{'+'.join(train_subjs)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_csv(run_dir / "inner_tuning_results.csv", tune_rows)

        print(f"[Best inner] score={best_score:.4f}, params={best_params}, inner_scores={best_inner_scores}")

        X_train = np.concatenate([subject_data[s]["X"] for s in train_subjs], axis=0)
        y_train = np.concatenate([subject_data[s]["y"] for s in train_subjs], axis=0)
        X_test = subject_data[test_subj]["X"]
        y_test = subject_data[test_subj]["y"]

        final_pipe = build_pipeline(
            selector_method=best_params["selector_method"],
            k=best_params["k"],
            C=best_params["C"],
            gamma=best_params["gamma"],
            class_weight=best_params["class_weight"],
            random_state=args.seed,
        )

        final_pipe.fit(X_train, y_train)
        y_pred = final_pipe.predict(X_test)

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

        print(f"test_acc={acc:.4f}, macro_f1={macro_f1:.4f}")
        print("cm:\n", cm)
        print(report)

        with open(run_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n")
            f.write(f"accuracy={acc:.6f}\nmacro_f1={macro_f1:.6f}\n")
            f.write(f"best_inner_score={best_score:.6f}\n")
            f.write(f"best_params={best_params}\n")
            f.write(f"cm=\n{cm}\n")

        plot_confusion_matrix(cm, run_dir / "confusion_matrix.png", f"Tuned SVM test={test_subj}")

        # Save selected feature names if possible.
        support = final_pipe.named_steps["selector"].get_support()
        selected_final_names = [name for name, keep in zip(selected_names, support) if keep]
        with open(run_dir / "selected_features.txt", "w", encoding="utf-8") as f:
            for name in selected_final_names:
                f.write(name + "\n")

        row = {
            "test_subject": test_subj,
            "train_subjects": "+".join(train_subjs),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "best_inner_score": float(best_score),
            "selector_method": best_params["selector_method"],
            "k": best_params["k"],
            "C": best_params["C"],
            "gamma": best_params["gamma"],
            "class_weight": str(best_params["class_weight"]),
            "num_selected_features": int(len(selected_final_names)),
        }

        for i, cname in enumerate(CLASS_NAMES):
            support_count = cm[i].sum()
            recall = cm[i, i] / support_count if support_count > 0 else 0.0
            pred_count = cm[:, i].sum()
            precision = cm[i, i] / pred_count if pred_count > 0 else 0.0
            row[f"{cname}_recall"] = float(recall)
            row[f"{cname}_precision"] = float(precision)
            row[f"{cname}_support"] = int(support_count)
            row[f"pred_{cname}_count"] = int(pred_count)

        outer_rows.append(row)
        all_true.append(y_test)
        all_pred.append(y_pred)

    save_csv(out_root / "loso_tuned_svm_results.csv", outer_rows)

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(CLASS_NAMES))))
    plot_confusion_matrix(cm_all, out_root / "confusion_matrix_overall.png", "Tuned SVM overall")

    summary = {
        "feature_group": args.feature_group,
        "select_mode": args.select_mode,
        "score": args.score,
        "feature_dim": int(n_features),
        "avg_accuracy": float(np.mean([r["accuracy"] for r in outer_rows])),
        "avg_macro_f1": float(np.mean([r["macro_f1"] for r in outer_rows])),
        "results": outer_rows,
        "args": vars(args),
    }

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] saved results to: {out_root}")
    print(f"[SUMMARY] avg_accuracy={summary['avg_accuracy']:.4f}, avg_macro_f1={summary['avg_macro_f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Nested LOSO SVM tuning + feature selection.")

    parser.add_argument("--jny_dir", type=str, default=r".\records\session_merge_jny")
    parser.add_argument("--wjw_dir", type=str, default=r".\records\session_merge_wjw")
    parser.add_argument("--zjh_dir", type=str, default=r".\records\session_merge_zjh")
    parser.add_argument("--zly_dir", type=str, default=r".\records\session_merge_zly")

    parser.add_argument("--feature_group", type=str, default="all",
                        choices=["rms", "amp", "shape", "freq", "shape_freq", "all"])

    parser.add_argument("--select_mode", type=str, default="kbest", choices=["none", "kbest"])
    parser.add_argument("--selector_methods", type=str, default="f_classif",
                        help="For kbest: f_classif or mutual_info, or f_classif,mutual_info")
    parser.add_argument("--k_list", type=str, default="5,10,15,20,30,40,60,80,120",
                        help="Candidate number of selected features. Clipped to feature dim.")

    parser.add_argument("--C_list", type=str, default="0.1,1,10,100")
    parser.add_argument("--gamma_list", type=str, default="scale,0.001,0.01,0.1,1")
    parser.add_argument("--class_weight_list", type=str, default="none,balanced")

    parser.add_argument("--score", type=str, default="macro_f1", choices=["accuracy", "macro_f1"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=r".\results_loso_svm_tuned")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
