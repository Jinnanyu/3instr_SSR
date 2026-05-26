# -*- coding: utf-8 -*-
"""Train SVM on handcrafted EMG features for 3-class single-subject baseline."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASS_NAMES = ["left", "right", "up"]  # y labels 0..2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="folder containing X_features.npy and y_single_labels.npy")
    ap.add_argument("--out", type=str, default=None, help="output .pkl path")
    ap.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "linear", "poly", "sigmoid"])
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--gamma", type=str, default="scale")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    X = np.load(data_dir / "X_features.npy").astype(np.float32)
    y = np.load(data_dir / "y_single_labels.npy").astype(np.int64)

    labels = sorted(np.unique(y).tolist())
    target_names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class{i}" for i in labels]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=args.test_size + args.val_size, random_state=args.seed, stratify=y
    )
    rel_val = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1.0 - rel_val, random_state=args.seed, stratify=y_tmp
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=args.kernel, C=args.C, gamma=args.gamma, probability=True, class_weight="balanced")),
    ])
    clf.fit(X_train, y_train)

    rows = []
    for split_name, XX, yy in [("val", X_val, y_val), ("test", X_test, y_test)]:
        pred = clf.predict(XX)
        acc = accuracy_score(yy, pred)
        print(f"\n=== {split_name} ===")
        print("acc:", f"{acc:.4f}")
        print("cm:\n", confusion_matrix(yy, pred, labels=labels))
        print(classification_report(yy, pred, labels=labels, target_names=target_names, digits=4, zero_division=0))
        rows.append({"split": split_name, "accuracy": acc})

    out_path = Path(args.out) if args.out else data_dir / "svm_emg_3class.pkl"
    feature_names = []
    fn = data_dir / "feature_names.csv"
    if fn.exists():
        feature_names = pd.read_csv(fn)["feature_name"].astype(str).tolist()

    payload = {
        "model": clf,
        "class_names": CLASS_NAMES,
        "feature_source": "handcrafted_emg",
        "feature_names": feature_names if feature_names else [str(x) for x in range(X.shape[1])],
        "config": vars(args),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    pd.DataFrame(rows).to_csv(data_dir / "single_subject_svm_results.csv", index=False, encoding="utf-8-sig")
    print("\n[OK] saved:", out_path)


if __name__ == "__main__":
    main()
