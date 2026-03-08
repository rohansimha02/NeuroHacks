"""
train.py
--------
Trains a 1D CNN on processed EMG windows to classify hand/wrist movements.

Architecture:
    Input (batch, 200, 1)
    -> Conv1D(64, k=3) -> BN -> ReLU
    -> Conv1D(128, k=3) -> BN -> ReLU
    -> MaxPool1D(2)
    -> Conv1D(128, k=3) -> BN -> ReLU
    -> GlobalAveragePool
    -> Dropout(0.3)
    -> Linear(64) -> ReLU
    -> Linear(num_classes)
    -> Softmax

Outputs:
    models/emg_classifier.pt   — best model weights
    models/label_map.json      — {int_str: movement_name} mapping

Run:
    python train.py
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# =============================================================================
# SETTINGS
# =============================================================================
EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
PATIENCE      = 15          # Early stopping patience (epochs without val improvement)
NUM_CLASSES   = 3
WINDOW_SIZE   = 200
N_CHANNELS    = 1
MOVEMENTS     = ['strong_grip', 'wrist_extension', 'finger_spread']

DATA_PROC_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH    = os.path.join(MODELS_DIR, "emg_classifier.pt")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")
# =============================================================================


class EMGClassifier(nn.Module):
    """
    1D Convolutional Neural Network for EMG movement classification.

    Expects input tensors of shape (batch, window_size, n_channels).
    PyTorch Conv1d expects (batch, channels, length), so the forward pass
    permutes the input accordingly.
    """

    def __init__(self, n_channels: int, window_size: int, num_classes: int):
        """
        Args:
            n_channels:  Number of EMG input channels (1 — channel_1 only).
            window_size: Number of time-steps per window (200).
            num_classes: Number of output movement classes (3).
        """
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # GlobalAveragePool collapses the time dimension
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout  = nn.Dropout(0.3)
        self.fc1      = nn.Linear(128, 64)
        self.relu_fc  = nn.ReLU()
        self.fc2      = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, window_size, n_channels).

        Returns:
            Raw logits of shape (batch, num_classes).
            Use CrossEntropyLoss (applies softmax internally) for training.
            Apply softmax manually at inference time for probabilities.
        """
        # Permute to (batch, n_channels, window_size) for Conv1d
        x = x.permute(0, 2, 1)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.pool(x)
        x = self.conv_block3(x)

        x = self.global_avg_pool(x)   # (batch, 128, 1)
        x = x.squeeze(-1)             # (batch, 128)

        x = self.dropout(x)
        x = self.relu_fc(self.fc1(x))
        return self.fc2(x)


def load_data() -> tuple:
    """
    Load preprocessed numpy arrays from data/processed/.

    Returns:
        Tuple (X_train, y_train, X_val, y_val, X_test, y_test) as PyTorch tensors.

    Raises:
        SystemExit: If expected .npy files are missing.
    """
    required = ["X_train.npy", "X_val.npy", "X_test.npy",
                "y_train.npy", "y_val.npy", "y_test.npy"]
    for fname in required:
        path = os.path.join(DATA_PROC_DIR, fname)
        if not os.path.exists(path):
            print(f"[ERROR] Missing processed file: {path}")
            print("  -> Run preprocess.py first.")
            import sys; sys.exit(1)

    X_train = torch.tensor(np.load(os.path.join(DATA_PROC_DIR, "X_train.npy")), dtype=torch.float32)
    X_val   = torch.tensor(np.load(os.path.join(DATA_PROC_DIR, "X_val.npy")),   dtype=torch.float32)
    X_test  = torch.tensor(np.load(os.path.join(DATA_PROC_DIR, "X_test.npy")),  dtype=torch.float32)
    y_train = torch.tensor(np.load(os.path.join(DATA_PROC_DIR, "y_train.npy")), dtype=torch.long)
    y_val   = torch.tensor(np.load(os.path.join(DATA_PROC_DIR, "y_val.npy")),   dtype=torch.long)
    y_test  = torch.tensor(np.load(os.path.join(DATA_PROC_DIR, "y_test.npy")),  dtype=torch.long)

    print(f"  Loaded data:")
    print(f"    Train : {X_train.shape}  labels: {y_train.shape}")
    print(f"    Val   : {X_val.shape}    labels: {y_val.shape}")
    print(f"    Test  : {X_test.shape}   labels: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Run one full training epoch.

    Args:
        model:     The CNN model.
        loader:    DataLoader for training data.
        optimizer: Adam optimizer.
        criterion: CrossEntropyLoss.
        device:    CPU or CUDA device.

    Returns:
        Tuple (avg_loss, accuracy) for this epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (preds.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluate model on a DataLoader without updating weights.

    Args:
        model:     The CNN model.
        loader:    DataLoader for eval data.
        criterion: CrossEntropyLoss.
        device:    CPU or CUDA device.

    Returns:
        Tuple (avg_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds      = model(X_batch)
            loss       = criterion(preds, y_batch)
            total_loss += loss.item() * len(y_batch)
            correct    += (preds.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Print a formatted confusion matrix and per-class accuracy to the terminal.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    header = "  " + " " * 14 + "  ".join(f"{m[:8]:>8}" for m in MOVEMENTS)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8}" for v in row)
        print(f"  {MOVEMENTS[i]:<14}  {row_str}")

    print("\n  Per-class accuracy:")
    for i, name in enumerate(MOVEMENTS):
        row_sum = cm[i].sum()
        acc = cm[i, i] / row_sum * 100 if row_sum > 0 else 0
        print(f"    {name:<12} : {acc:.1f}%  ({cm[i,i]}/{row_sum} correct)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    print("\n  Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)
    test_ds  = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model     = EMGClassifier(N_CHANNELS, WINDOW_SIZE, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\n  Model architecture:")
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    print(f"\n  Training for up to {EPOCHS} epochs (early stop patience={PATIENCE})...")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    best_val_acc    = -1.0
    best_val_loss   = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"  {epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.1%} | {val_loss:>9.4f} | {val_acc:>7.1%}  ({elapsed:.1f}s)")

        # Save best model on validation accuracy
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc  = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    print(f"\n  Best validation accuracy: {best_val_acc:.1%}")
    print(f"  Model saved to: {MODEL_PATH}")

    # ---- Final evaluation on test set ----
    print("\n  Loading best model for test evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.1%}")

    # Collect all predictions for confusion matrix
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    print_confusion_matrix(np.array(all_true), np.array(all_preds))

    # ---- Save label map ----
    label_map = {str(i): name for i, name in enumerate(MOVEMENTS)}
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n  Label map saved to: {LABEL_MAP_PATH}")
    print(f"  Contents: {label_map}")
    print("\n  Training complete!\n")


if __name__ == "__main__":
    main()
