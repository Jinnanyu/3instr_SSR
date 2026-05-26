# -*- coding: utf-8 -*-
"""Train EMGCNN for 2 actions (left/right) from X_single_windows.npy."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = ["left", "right", "up"]
N_CLASSES = len(CLASS_NAMES)


class EMGDataset(Dataset):
    def __init__(self, X, y, per_window_norm=True, mean=None, std=None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.per_window_norm = bool(per_window_norm)
        self.mean = mean.astype(np.float32) if mean is not None else None
        self.std = std.astype(np.float32) if std is not None else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # [T,C]
        if self.per_window_norm:
            mu = x.mean(axis=0, keepdims=True)
            sigma = x.std(axis=0, keepdims=True) + 1e-8
            x = (x - mu) / sigma
        else:
            assert self.mean is not None and self.std is not None
            x = (x - self.mean.squeeze(0)) / self.std.squeeze(0)
        x = x.transpose(1, 0)  # [C,T]
        return torch.from_numpy(x), torch.tensor(self.y[idx])


class EMGCNN(nn.Module):
    def __init__(self, n_channels=5, n_classes=N_CLASSES):
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
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))


def compute_global_mean_std(X):
    mean = X.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = (X.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
    return mean, std


@torch.no_grad()
def eval_model(model, loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss_sum += loss.item() * yb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total


def save_training_curves(train_losses, val_losses, train_accs, val_accs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=300)
    plt.close()

    print("[OK] saved curves to:", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="cnn_2actions_leftright.pth")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--per_window_norm", action="store_true", default=True)
    ap.add_argument("--plot_dir", type=str, default="training_plots")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    X = np.load(data_dir / "X_single_windows.npy")
    y = np.load(data_dir / "y_single_labels.npy")
    print("X:", X.shape, "y:", y.shape)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    mean, std = compute_global_mean_std(X_train)

    train_ds = EMGDataset(X_train, y_train, per_window_norm=args.per_window_norm, mean=mean, std=std)
    val_ds = EMGDataset(X_val, y_val, per_window_norm=args.per_window_norm, mean=mean, std=std)
    test_ds = EMGDataset(X_test, y_test, per_window_norm=args.per_window_norm, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EMGCNN(n_channels=X.shape[2], n_classes=N_CLASSES).to(device)

    counts = np.bincount(y_train, minlength=N_CLASSES)
    weights = len(y_train) / (N_CLASSES * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    best_state = None
    no_imp = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        corr = 0
        tot = 0
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
            corr += (pred == yb).sum().item()
            tot += yb.size(0)

        train_loss = loss_sum / tot
        train_acc = corr / tot
        val_loss, val_acc = eval_model(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_p = []
    all_t = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            p = logits.argmax(dim=1).cpu().numpy()
            all_p.append(p)
            all_t.append(yb.numpy())

    y_pred = np.concatenate(all_p)
    y_true = np.concatenate(all_t)
    print("test_acc:", (y_pred == y_true).mean())
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))
    print("confusion_matrix:\n", confusion_matrix(y_true, y_pred))

    ckpt_path = Path(args.ckpt)
    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "per_window_norm": bool(args.per_window_norm),
        "class_names": CLASS_NAMES,
    }, ckpt_path)
    print("[OK] saved ckpt:", ckpt_path)

    save_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        out_dir=args.plot_dir
    )


if __name__ == "__main__":
    main()