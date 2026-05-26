# -*- coding: utf-8 -*-
"""
eval_loso_stft_cnn.py

STFT-CNN 跨个体留一被试法 LOSO 测试脚本。
适用于工程中已有的 X_single_windows.npy / y_single_labels.npy。

默认输入：X_single_windows.npy 形状为 [N, T, C]，例如 [样本数, 200, 5]。
处理流程：
  原始窗口 [N,T,C]
    -> 可选原始波形归一化 raw_norm
    -> STFT，得到 [N,C,F,TimeFrames]
    -> 可选频谱图标准化 spec_norm
    -> 2D-CNN 分类 left/right/up

示例：
  python eval_loso_stft_cnn.py --raw_norm per_window --n_fft 64 --hop_length 16 --epochs 40 --out_dir .\results_loso_stft_cnn
  python eval_loso_stft_cnn.py --raw_norm train_zscore --n_fft 128 --hop_length 32 --epochs 40 --out_dir .\results_loso_stft_cnn

说明：
  1) 脚本不会自动改变 0.2s/0.3s/0.4s 窗长。窗长由 X_single_windows.npy 决定。
  2) 若要测试更长窗，请先用 build_windows_from_trials.py 重新生成不同 win_sec 的数据目录。
  3) 推荐先跑 raw_norm=per_window，因为这最接近你原 CNN 的 per_window_norm=True。
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


CLASS_NAMES = ["left", "right", "up"]
N_CLASSES = len(CLASS_NAMES)
EPS = 1e-8


# -----------------------------
# Dataset / Model
# -----------------------------
class STFTMapDataset(Dataset):
    """X: [N, C, F, S]，C 作为 2D-CNN 的输入通道数。"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class STFTCNN(nn.Module):
    def __init__(self, n_channels: int = 5, n_classes: int = N_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# -----------------------------
# Loading / raw normalization
# -----------------------------
def load_subject(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    X_path = data_dir / "X_single_windows.npy"
    y_path = data_dir / "y_single_labels.npy"
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing X/y in {data_dir}: {X_path.name}, {y_path.name}")
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    if X.ndim != 3:
        raise ValueError(f"X should be [N,T,C], got {X.shape} from {X_path}")
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch in {data_dir}: {len(X)} vs {len(y)}")
    return X, y


def per_window_norm(X: np.ndarray) -> np.ndarray:
    """每个窗口、每个通道独立 z-score，适合原始肌电波形。X: [N,T,C]"""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + EPS
    return (X - mu) / sd


def train_zscore_norm(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """只用训练被试统计每个通道 mean/std，再应用到训练集和测试集。"""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + EPS
    return (X_train - mean) / std, (X_test - mean) / std, {
        "raw_train_mean": mean.squeeze().tolist(),
        "raw_train_std": std.squeeze().tolist(),
    }


def subject_p95_norm(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """每个被试按每通道 abs 95% 分位数归一化。注意：这是近似幅值归一化，不是真正 MVC。"""
    scale = np.percentile(np.abs(X), 95, axis=(0, 1), keepdims=True) + EPS
    return X / scale, scale.squeeze()


def apply_raw_norm_for_loso(
    subject_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],
    train_names: List[str],
    test_name: str,
    raw_norm: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """返回 raw waveform X_train/y_train/X_test/y_test，暂不做 STFT。"""
    info = {"raw_norm": raw_norm}

    if raw_norm == "subject_p95":
        X_train_list, y_train_list = [], []
        scales = {}
        for name in train_names:
            X, y = subject_arrays[name]
            Xn, sc = subject_p95_norm(X)
            X_train_list.append(Xn)
            y_train_list.append(y)
            scales[name] = sc.tolist()
        X_test, y_test = subject_arrays[test_name]
        X_test, sc = subject_p95_norm(X_test)
        scales[test_name] = sc.tolist()
        info["subject_p95_scale"] = scales
        return np.concatenate(X_train_list, axis=0), np.concatenate(y_train_list, axis=0), X_test, y_test, info

    X_train = np.concatenate([subject_arrays[n][0] for n in train_names], axis=0)
    y_train = np.concatenate([subject_arrays[n][1] for n in train_names], axis=0)
    X_test, y_test = subject_arrays[test_name]

    if raw_norm == "none":
        return X_train, y_train, X_test, y_test, info
    if raw_norm == "per_window":
        return per_window_norm(X_train), y_train, per_window_norm(X_test), y_test, info
    if raw_norm == "train_zscore":
        X_train, X_test, stat = train_zscore_norm(X_train, X_test)
        info.update(stat)
        return X_train, y_train, X_test, y_test, info

    raise ValueError(f"Unknown raw_norm: {raw_norm}")


# -----------------------------
# STFT feature map
# -----------------------------
def compute_stft_maps(
    X: np.ndarray,
    fs: int,
    n_fft: int,
    hop_length: int,
    win_length: Optional[int],
    freq_max: Optional[float],
    mode: str,
    stft_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    X: [N,T,C] -> maps: [N,C,F,S]
    mode:
      magnitude: abs(STFT)
      log_magnitude: log1p(abs(STFT))
      log_power: log1p(abs(STFT)^2)
    """
    if win_length is None:
        win_length = n_fft
    if X.ndim != 3:
        raise ValueError(f"X should be [N,T,C], got {X.shape}")
    N, T, C = X.shape
    if T < win_length:
        raise ValueError(f"Window length T={T} is smaller than STFT win_length={win_length}. 请减小 n_fft/win_length 或重新生成更长窗口。")

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    if freq_max is not None and freq_max > 0:
        keep = np.where(freqs <= freq_max)[0]
    else:
        keep = np.arange(len(freqs))

    out_chunks = []
    window = torch.hann_window(win_length, device=device)

    for start in range(0, N, stft_batch_size):
        end = min(N, start + stft_batch_size)
        xb = X[start:end]  # [B,T,C]
        B = xb.shape[0]
        # [B,C,T] -> [B*C,T]
        xb_t = torch.from_numpy(xb.transpose(0, 2, 1).reshape(B * C, T)).float().to(device)
        spec = torch.stft(
            xb_t,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )  # [B*C,F,S]
        mag = spec.abs()
        if mode == "magnitude":
            feat = mag
        elif mode == "log_magnitude":
            feat = torch.log1p(mag)
        elif mode == "log_power":
            feat = torch.log1p(mag.pow(2))
        else:
            raise ValueError(f"Unknown stft_mode: {mode}")
        feat = feat[:, keep, :]
        F, S = feat.shape[1], feat.shape[2]
        feat = feat.reshape(B, C, F, S).detach().cpu().numpy().astype(np.float32)
        out_chunks.append(feat)

    return np.concatenate(out_chunks, axis=0)


def fit_spec_norm(X_train_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对 STFT map 做标准化。按每个输入通道 C 统计 mean/std，统计维度为 N,F,S。"""
    mean = X_train_map.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train_map.std(axis=(0, 2, 3), keepdims=True) + EPS
    return mean, std


def apply_spec_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def per_sample_spec_norm(X: np.ndarray) -> np.ndarray:
    """每个样本每个通道独立标准化频谱图，X: [N,C,F,S]。"""
    mean = X.mean(axis=(2, 3), keepdims=True)
    std = X.std(axis=(2, 3), keepdims=True) + EPS
    return (X - mean) / std


# -----------------------------
# Training / evaluation utilities
# -----------------------------
def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int):
    train_loader = DataLoader(STFTMapDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(STFTMapDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(STFTMapDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        if criterion is not None:
            loss_sum += criterion(logits, yb).item() * yb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(yb.cpu().numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    loss = loss_sum / max(total, 1) if criterion is not None else None
    return loss, correct / max(total, 1), y_true, y_pred


def save_curve(values_train, values_val, ylabel: str, title: str, out_path: Path):
    plt.figure(figsize=(8, 5))
    xs = np.arange(1, len(values_train) + 1)
    plt.plot(xs, values_train, label="train")
    plt.plot(xs, values_val, label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(N_CLASSES), CLASS_NAMES)
    plt.yticks(range(N_CLASSES), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_mean_spectrogram(X_map: np.ndarray, y: np.ndarray, out_dir: Path, prefix: str):
    """保存每个类别的平均频谱图，便于论文/答辩展示。对 C 个通道取平均后画图。"""
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        idx = np.where(y == cls_idx)[0]
        if len(idx) == 0:
            continue
        avg = X_map[idx].mean(axis=0).mean(axis=0)  # [F,S]
        plt.figure(figsize=(6, 4))
        plt.imshow(avg, aspect="auto", origin="lower")
        plt.title(f"{prefix} mean STFT - {cls_name}")
        plt.xlabel("STFT frame")
        plt.ylabel("Frequency bin")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_mean_stft_{cls_name}.png", dpi=300)
        plt.close()


def train_one_round(
    X_train_raw: np.ndarray,
    y_train_all: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    n_channels: int,
    out_dir: Path,
    args,
):
    # 训练被试内部划分 validation；测试被试绝不参与 validation/early stopping
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train_all,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y_train_all,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    stft_device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu and args.stft_on_gpu else "cpu")

    print("[STFT] computing train maps...")
    X_train = compute_stft_maps(
        X_train_raw, args.fs, args.n_fft, args.hop_length, args.win_length,
        args.freq_max, args.stft_mode, args.stft_batch_size, stft_device
    )
    print("[STFT] computing val maps...")
    X_val = compute_stft_maps(
        X_val_raw, args.fs, args.n_fft, args.hop_length, args.win_length,
        args.freq_max, args.stft_mode, args.stft_batch_size, stft_device
    )
    print("[STFT] computing test maps...")
    X_test = compute_stft_maps(
        X_test_raw, args.fs, args.n_fft, args.hop_length, args.win_length,
        args.freq_max, args.stft_mode, args.stft_batch_size, stft_device
    )

    spec_info = {"raw_shape_train": list(X_train_raw.shape), "stft_shape_train": list(X_train.shape)}
    if args.spec_norm == "train_zscore":
        mean, std = fit_spec_norm(X_train)
        X_train = apply_spec_norm(X_train, mean, std)
        X_val = apply_spec_norm(X_val, mean, std)
        X_test = apply_spec_norm(X_test, mean, std)
        spec_info["spec_norm_mean_shape"] = list(mean.shape)
        spec_info["spec_norm_std_shape"] = list(std.shape)
    elif args.spec_norm == "per_sample":
        X_train = per_sample_spec_norm(X_train)
        X_val = per_sample_spec_norm(X_val)
        X_test = per_sample_spec_norm(X_test)
    elif args.spec_norm == "none":
        pass
    else:
        raise ValueError(f"Unknown spec_norm: {args.spec_norm}")

    (out_dir / "stft_shape_info.json").write_text(json.dumps(spec_info, indent=2), encoding="utf-8")
    save_mean_spectrogram(X_train, y_train, out_dir, prefix="train")

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=args.batch_size
    )

    model = STFTCNN(n_channels=n_channels, n_classes=N_CLASSES, dropout=args.dropout).to(device)

    counts = np.bincount(y_train, minlength=N_CLASSES)
    weights = len(y_train) / (N_CLASSES * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_state = None
    no_imp = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_acc, y_true, y_pred = evaluate(model, test_loader, device, criterion=None)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "n_channels": n_channels,
        "raw_norm": args.raw_norm,
        "spec_norm": args.spec_norm,
        "fs": args.fs,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "freq_max": args.freq_max,
        "stft_mode": args.stft_mode,
    }, out_dir / "best_model.pth")

    save_curve(train_losses, val_losses, "Loss", "STFT-CNN LOSO Loss", out_dir / "loss_curve.png")
    save_curve(train_accs, val_accs, "Accuracy", "STFT-CNN LOSO Accuracy", out_dir / "accuracy_curve.png")
    save_confusion_matrix(cm, out_dir / "confusion_matrix.png", title="STFT-CNN LOSO Confusion Matrix")

    (out_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    np.save(out_dir / "y_true.npy", y_true)
    np.save(out_dir / "y_pred.npy", y_pred)

    metrics = {
        "accuracy": float(test_acc),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "left_recall": float(recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "right_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "up_recall": float(recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)),
        "best_val_acc": float(best_val_acc),
        "support_left": int((y_true == 0).sum()),
        "support_right": int((y_true == 1).sum()),
        "support_up": int((y_true == 2).sum()),
        "pred_left": int((y_pred == 0).sum()),
        "pred_right": int((y_pred == 1).sum()),
        "pred_up": int((y_pred == 2).sum()),
        "report": report_dict,
    }
    return metrics, cm


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jny_dir", type=str, default="records/session_merge_jny")
    ap.add_argument("--wjw_dir", type=str, default="records/session_merge_wjw")
    ap.add_argument("--zjh_dir", type=str, default="records/session_merge_zjh")
    ap.add_argument("--zly_dir", type=str, default="records/session_merge_zly")
    ap.add_argument("--out_dir", type=str, default="results_loso_stft_cnn")

    ap.add_argument("--raw_norm", type=str, default="per_window", choices=["none", "per_window", "train_zscore", "subject_p95"])
    ap.add_argument("--spec_norm", type=str, default="train_zscore", choices=["none", "train_zscore", "per_sample"])

    ap.add_argument("--fs", type=int, default=1000)
    ap.add_argument("--n_fft", type=int, default=64)
    ap.add_argument("--hop_length", type=int, default=16)
    ap.add_argument("--win_length", type=int, default=None)
    ap.add_argument("--freq_max", type=float, default=250.0, help="只保留 <=freq_max Hz 的频率；<=0 表示不截断")
    ap.add_argument("--stft_mode", type=str, default="log_power", choices=["magnitude", "log_magnitude", "log_power"])
    ap.add_argument("--stft_batch_size", type=int, default=512)
    ap.add_argument("--stft_on_gpu", action="store_true", help="用 GPU 计算 STFT；显存不够时不要开")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.freq_max is not None and args.freq_max <= 0:
        args.freq_max = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    subject_dirs = {
        "jny": Path(args.jny_dir),
        "wjw": Path(args.wjw_dir),
        "zjh": Path(args.zjh_dir),
        "zly": Path(args.zly_dir),
    }
    subject_arrays = {}
    for name, d in subject_dirs.items():
        X, y = load_subject(d)
        subject_arrays[name] = (X, y)
        print(f"[{name}] X={X.shape}, y={y.shape}, counts={np.bincount(y, minlength=N_CLASSES).tolist()}")

    n_channels_set = {subject_arrays[n][0].shape[2] for n in subject_arrays}
    if len(n_channels_set) != 1:
        raise ValueError(f"Different channel numbers across subjects: {n_channels_set}")
    n_channels = list(n_channels_set)[0]

    T_set = {subject_arrays[n][0].shape[1] for n in subject_arrays}
    if len(T_set) != 1:
        raise ValueError(f"Different window lengths across subjects: {T_set}. STFT-CNN 对比时建议三个人窗口长度一致。")
    T = list(T_set)[0]
    win_sec_est = T / float(args.fs)

    tag = (
        f"raw-{args.raw_norm}_spec-{args.spec_norm}_"
        f"T{T}_win{win_sec_est:.2f}s_nfft{args.n_fft}_hop{args.hop_length}_"
        f"mode-{args.stft_mode}_fmax-{args.freq_max if args.freq_max is not None else 'all'}"
    )
    root = Path(args.out_dir) / tag
    root.mkdir(parents=True, exist_ok=True)

    params = vars(args).copy()
    params["T_samples"] = T
    params["estimated_win_sec"] = win_sec_est
    (root / "run_params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")

    results = []
    cms = []
    subjects = list(subject_arrays.keys())
    for test_name in subjects:
        train_names = [n for n in subjects if n != test_name]
        print("\n" + "=" * 80)
        print(f"LOSO STFT-CNN test={test_name}; train={'+'.join(train_names)}; raw_norm={args.raw_norm}; spec_norm={args.spec_norm}")
        print("=" * 80)

        X_train, y_train, X_test, y_test, norm_info = apply_raw_norm_for_loso(
            subject_arrays, train_names, test_name, args.raw_norm
        )
        print(f"raw train X={X_train.shape}, raw test X={X_test.shape}")

        round_dir = root / f"test-{test_name}_train-{'+'.join(train_names)}"
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "norm_info.json").write_text(json.dumps(norm_info, indent=2, ensure_ascii=False), encoding="utf-8")

        metrics, cm = train_one_round(X_train, y_train, X_test, y_test, n_channels, round_dir, args)
        row = {
            "test_subject": test_name,
            "train_subjects": "+".join(train_names),
            "raw_norm": args.raw_norm,
            "spec_norm": args.spec_norm,
            "T_samples": T,
            "win_sec_est": win_sec_est,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "stft_mode": args.stft_mode,
            "freq_max": args.freq_max if args.freq_max is not None else "all",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "left_recall": metrics["left_recall"],
            "right_recall": metrics["right_recall"],
            "up_recall": metrics["up_recall"],
            "best_val_acc": metrics["best_val_acc"],
            "pred_left": metrics["pred_left"],
            "pred_right": metrics["pred_right"],
            "pred_up": metrics["pred_up"],
        }
        results.append(row)
        cms.append(cm)

        print(f"test_acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
        print("cm:\n", cm)
        print((round_dir / "classification_report.txt").read_text(encoding="utf-8"))

    csv_path = root / "loso_stft_cnn_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    overall_cm = np.sum(np.stack(cms, axis=0), axis=0)
    save_confusion_matrix(overall_cm, root / "confusion_matrix_overall.png", title="STFT-CNN LOSO Overall Confusion Matrix")
    np.savetxt(root / "confusion_matrix_overall.csv", overall_cm, delimiter=",", fmt="%d")

    avg_acc = np.mean([r["accuracy"] for r in results])
    avg_macro_f1 = np.mean([r["macro_f1"] for r in results])
    summary = {
        "avg_accuracy": float(avg_acc),
        "avg_macro_f1": float(avg_macro_f1),
        "results": results,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[OK] saved results to:", root)
    print(f"[SUMMARY] avg_accuracy={avg_acc:.4f}, avg_macro_f1={avg_macro_f1:.4f}")


if __name__ == "__main__":
    main()
