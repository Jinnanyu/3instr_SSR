# -*- coding: utf-8 -*-
"""
eval_loso_cnn_lstm.py

CNN-LSTM LOSO for facial EMG / silent speech command recognition.

Input files in each subject folder:
    X_single_windows.npy
    y_single_labels.npy

Default subjects:
    records/session_merge_jny
    records/session_merge_wjw
    records/session_merge_zjh
    records/session_merge_zly

Recommended:
    python eval_loso_cnn_lstm.py --norm per_window --epochs 40 --batch_size 128 --out_dir .\results_loso_cnn_lstm_new
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


CLASS_NAMES = ["left", "right", "up"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def ensure_channels_first(X):
    if X.ndim != 3:
        raise ValueError(f"X must be [N,T,C] or [N,C,T], got {X.shape}")
    if X.shape[1] <= 16 and X.shape[2] > X.shape[1]:
        return X.astype(np.float32)
    if X.shape[2] <= 16 and X.shape[1] > X.shape[2]:
        return np.transpose(X, (0, 2, 1)).astype(np.float32)
    return X.astype(np.float32)


def load_subject(name, folder):
    folder = Path(folder)
    x_path = folder / "X_single_windows.npy"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing {x_path}")

    y_path = None
    for p in [folder / "y_single_labels.npy", folder / "y_labels.npy", folder / "y.npy"]:
        if p.exists():
            y_path = p
            break
    if y_path is None:
        raise FileNotFoundError(f"Missing labels in {folder}")

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    if len(X) != len(y):
        raise ValueError(f"{name}: X/y length mismatch: {len(X)} vs {len(y)}")

    X = ensure_channels_first(X)
    print(f"[{name}] X={X.shape} [N,C,T], y={y.shape}, counts={np.bincount(y, minlength=3).tolist()}")
    return X, y


def normalize_per_window(X, eps=1e-6):
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + eps
    return (X - mean) / std


def normalize_train_zscore(X_train, X_val, X_test, eps=1e-6):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + eps
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def normalize_subject_p95(X, eps=1e-6):
    scale = np.percentile(np.abs(X), 95, axis=(0, 2), keepdims=True) + eps
    return X / scale


def apply_norm(norm, X_train, X_val, X_test):
    if norm == "none":
        return X_train, X_val, X_test
    if norm == "per_window":
        return normalize_per_window(X_train), normalize_per_window(X_val), normalize_per_window(X_test)
    if norm == "train_zscore":
        return normalize_train_zscore(X_train, X_val, X_test)
    if norm == "subject_p95":
        return normalize_subject_p95(X_train), normalize_subject_p95(X_val), normalize_subject_p95(X_test)
    raise ValueError(f"Unknown norm: {norm}")


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNNLSTM(nn.Module):
    """
    CNN extracts local EMG waveform features; LSTM models temporal changes.

    Input:  [B, C, T]
    CNN:    [B, F, T']
    LSTM:   [B, T', F] -> [B, T', H]
    Output: [B, num_classes]
    """
    def __init__(
        self,
        in_ch=5,
        num_classes=3,
        conv_ch=64,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.35,
        bidirectional=False,
        temporal_pool="last",
    ):
        super().__init__()
        self.temporal_pool = temporal_pool

        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, conv_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(conv_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(conv_ch, conv_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(conv_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(conv_ch, conv_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(conv_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=conv_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        z = self.cnn(x)              # [B,F,T']
        z = z.transpose(1, 2)        # [B,T',F]
        out, _ = self.lstm(z)        # [B,T',H]

        if self.temporal_pool == "mean":
            feat = out.mean(dim=1)
        elif self.temporal_pool == "max":
            feat = out.max(dim=1).values
        else:
            feat = out[:, -1, :]

        return self.classifier(feat)


def make_loaders(X_train, y_train, X_val, y_val, batch_size, num_workers=0):
    train_ds = WindowDataset(X_train, y_train)
    val_ds = WindowDataset(X_val, y_val)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def run_one_epoch(model, loader, optimizer, device, criterion, train=True):
    model.train(train)
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_count += len(yb)
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def predict(model, X, batch_size, device):
    ds = WindowDataset(X, np.zeros(len(X), dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def split_calibration_data(X, y, calib_ratio, seed):
    if calib_ratio <= 0:
        return None, None, X, y
    X_calib, X_test, y_calib, y_test = train_test_split(
        X, y, train_size=calib_ratio, random_state=seed, stratify=y
    )
    return X_calib, y_calib, X_test, y_test


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


def plot_curve(values1, values2, out_path, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values1, label="train")
    ax.plot(values2, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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

    subjects = {}
    for name, folder in subject_dirs.items():
        X, y = load_subject(name, folder)
        subjects[name] = {"X": X, "y": y}

    in_ch = next(iter(subjects.values()))["X"].shape[1]
    time_len = next(iter(subjects.values()))["X"].shape[2]

    exp_name = (
        f"cnn_lstm_norm-{args.norm}_calib-{args.calib_ratio:.2f}_T{time_len}"
        f"_conv{args.conv_ch}_h{args.lstm_hidden}_layers{args.lstm_layers}_{args.temporal_pool}_drop{args.dropout}"
    )
    if args.bidirectional:
        exp_name += "_bidir"
    if args.class_weight:
        exp_name += "_classweight"

    out_root = Path(args.out_dir) / exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    all_results, all_true, all_pred = [], [], []

    for test_subj in subject_dirs.keys():
        train_subjs = [s for s in subject_dirs.keys() if s != test_subj]
        print("\n" + "=" * 80)
        print(f"LOSO CNN-LSTM test={test_subj}; train={'+'.join(train_subjs)}; norm={args.norm}; calib={args.calib_ratio}")
        print("=" * 80)

        X_train_base = np.concatenate([subjects[s]["X"] for s in train_subjs], axis=0)
        y_train_base = np.concatenate([subjects[s]["y"] for s in train_subjs], axis=0)

        X_test_subject = subjects[test_subj]["X"]
        y_test_subject = subjects[test_subj]["y"]

        X_calib, y_calib, X_test_final, y_test_final = split_calibration_data(
            X_test_subject, y_test_subject, args.calib_ratio, args.seed
        )

        if args.calib_ratio > 0:
            X_train_base = np.concatenate([X_train_base, X_calib], axis=0)
            y_train_base = np.concatenate([y_train_base, y_calib], axis=0)
            print(f"[Calibration] added {len(y_calib)} samples from {test_subj}; final test={len(y_test_final)}")

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_base, y_train_base, test_size=args.val_ratio, random_state=args.seed, stratify=y_train_base
        )

        X_train, X_val, X_test_final = apply_norm(args.norm, X_train, X_val, X_test_final)
        print(f"train X={X_train.shape}, val X={X_val.shape}, test X={X_test_final.shape}")

        run_dir = out_root / f"test-{test_subj}_train-{'+'.join(train_subjs)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader = make_loaders(
            X_train, y_train, X_val, y_val, args.batch_size, args.num_workers
        )

        model = CNNLSTM(
            in_ch=in_ch,
            num_classes=len(CLASS_NAMES),
            conv_ch=args.conv_ch,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            temporal_pool=args.temporal_pool,
        ).to(device)

        if args.class_weight:
            counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float32)
            weights = counts.sum() / np.maximum(counts, 1.0)
            weights = weights / weights.mean()
            print(f"[Class weights] {weights.tolist()}")
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_monitor, best_state, no_improve = None, None, 0
        hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_one_epoch(model, train_loader, optimizer, device, criterion, train=True)
            val_loss, val_acc = run_one_epoch(model, val_loader, None, device, criterion, train=False)

            hist["train_loss"].append(train_loss)
            hist["train_acc"].append(train_acc)
            hist["val_loss"].append(val_loss)
            hist["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:03d}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            monitor_value = val_acc if args.monitor == "val_acc" else -val_loss
            if best_monitor is None or monitor_value > best_monitor:
                best_monitor = monitor_value
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
            y_test_final, y_pred, labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES, digits=4, zero_division=0
        )

        print(f"test_acc={acc:.4f}, macro_f1={macro_f1:.4f}")
        print("cm:\n", cm)
        print(report)

        with open(run_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n")
            f.write(f"accuracy={acc:.6f}\nmacro_f1={macro_f1:.6f}\n")
            f.write(f"cm=\n{cm}\n")

        plot_confusion_matrix(cm, run_dir / "confusion_matrix.png", f"CNN-LSTM test={test_subj}")
        plot_curve(hist["train_loss"], hist["val_loss"], run_dir / "loss_curve.png", "Loss", f"Loss CNN-LSTM test={test_subj}")
        plot_curve(hist["train_acc"], hist["val_acc"], run_dir / "accuracy_curve.png", "Accuracy", f"Accuracy CNN-LSTM test={test_subj}")

        torch.save(model.state_dict(), run_dir / "best_model.pth")

        row = {
            "model": "cnn_lstm",
            "norm": args.norm,
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

        with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump({"row": row, "input_shape": [int(in_ch), int(time_len)], "args": vars(args)}, f, ensure_ascii=False, indent=2)

    csv_path = out_root / "loso_cnn_lstm_results.csv"
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    cm_all = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(CLASS_NAMES))))
    plot_confusion_matrix(cm_all, out_root / "confusion_matrix_overall.png", "CNN-LSTM overall")

    summary = {
        "model": "cnn_lstm",
        "avg_accuracy": float(np.mean([r["accuracy"] for r in all_results])),
        "avg_macro_f1": float(np.mean([r["macro_f1"] for r in all_results])),
        "results": all_results,
        "input_shape": [int(in_ch), int(time_len)],
        "args": vars(args),
    }

    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] saved results to: {out_root}")
    print(f"[SUMMARY] avg_accuracy={summary['avg_accuracy']:.4f}, avg_macro_f1={summary['avg_macro_f1']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="CNN-LSTM LOSO for facial EMG windows.")

    parser.add_argument("--jny_dir", type=str, default=r".\records\session_merge_jny")
    parser.add_argument("--wjw_dir", type=str, default=r".\records\session_merge_wjw")
    parser.add_argument("--zjh_dir", type=str, default=r".\records\session_merge_zjh")
    parser.add_argument("--zly_dir", type=str, default=r".\records\session_merge_zly")

    parser.add_argument("--norm", type=str, default="per_window",
                        choices=["none", "per_window", "train_zscore", "subject_p95"])

    parser.add_argument("--calib_ratio", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)

    parser.add_argument("--conv_ch", type=int, default=64)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--temporal_pool", type=str, default="last", choices=["last", "mean", "max"])

    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--monitor", type=str, default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_dir", type=str, default=r".\results_loso_cnn_lstm_new")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_loso(args)
