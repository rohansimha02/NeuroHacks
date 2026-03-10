"""
inference.py
------------------
ML EMG classification using the trained 1D CNN model.

Run:
    python inference.py
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from preprocess import (
    bandpass_filter, notch_filter, normalize_window,
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ, NOTCH_Q, SAMPLE_RATE,
    WINDOW_SIZE,
)
from hub_integration import send_movement_event

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# =============================================================================
# SETTINGS
# =============================================================================
SERIAL_PORT        = "COM4"
INFERENCE_INTERVAL = 0.8
CH_1_IDX           = 1

MODELS_DIR      = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH      = os.path.join(MODELS_DIR, "emg_classifier.pt")
LABEL_MAP_PATH  = os.path.join(MODELS_DIR, "label_map.json")

N_CHANNELS  = 1
NUM_CLASSES = 3
# =============================================================================


class EMGClassifier(nn.Module):
    def __init__(self, n_channels: int, num_classes: int):
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
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout  = nn.Dropout(0.3)
        self.fc1      = nn.Linear(128, 64)
        self.relu_fc  = nn.ReLU()
        self.fc2      = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.pool(x)
        x = self.conv_block3(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.relu_fc(self.fc1(x))
        return self.fc2(x)


def load_model() -> tuple:
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  -> Run train.py first.")
        sys.exit(1)
    if not os.path.exists(LABEL_MAP_PATH):
        print(f"[ERROR] Label map not found: {LABEL_MAP_PATH}")
        print("  -> Run train.py first.")
        sys.exit(1)

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)  # {"0": "strong_grip", ...}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EMGClassifier(N_CHANNELS, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, label_map, device


def connect_board() -> BoardShim:
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    try:
        board.prepare_session()
        board.start_stream()
        print(f"  Connected to Cyton board on {SERIAL_PORT}")
        time.sleep(2)
        return board
    except Exception as e:
        print(f"\n[ERROR] Could not connect to Cyton board: {e}")
        sys.exit(1)


def get_latest_window(board: BoardShim) -> np.ndarray | None:
    if board.get_board_data_count() < WINDOW_SIZE:
        return None
    data = board.get_board_data()
    ch1 = data[CH_1_IDX, -WINDOW_SIZE:]
    return ch1.reshape(-1, 1).astype(np.float32)


def filter_window(window: np.ndarray) -> np.ndarray:
    filtered = np.zeros_like(window)
    for ch in range(window.shape[1]):
        sig = window[:, ch]
        sig = bandpass_filter(sig, BANDPASS_LOW, BANDPASS_HIGH, SAMPLE_RATE)
        sig = notch_filter(sig, NOTCH_FREQ, NOTCH_Q, SAMPLE_RATE)
        filtered[:, ch] = sig
    return normalize_window(filtered)


def classify_window(model, window: np.ndarray, label_map: dict, device) -> tuple:
    """Run the CNN on a filtered window. Returns (label, confidence)."""
    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    idx = int(probs.argmax().item())
    return label_map[str(idx)], float(probs[idx].item())


def main():
    print("\n" + "="*50)
    print("  EMG REAL-TIME INFERENCE (ML MODEL)")
    print(f"  Board             : Cyton ({SERIAL_PORT})")
    print(f"  Inference interval: {INFERENCE_INTERVAL}s")
    print(f"  Model             : {MODEL_PATH}")
    print("="*50 + "\n")

    print("  Loading model...")
    model, label_map, device = load_model()
    print(f"  Model loaded. Classes: {list(label_map.values())}")

    print("\n  Connecting to board...")
    board = connect_board()

    print("\n  Starting inference loop. Press Ctrl+C to stop.\n")
    print("-" * 50)

    detection_counts = {name: 0 for name in label_map.values()}

    try:
        while True:
            loop_start = time.time()

            window = get_latest_window(board)
            if window is None:
                time.sleep(INFERENCE_INTERVAL)
                continue

            filtered = filter_window(window)
            movement, confidence = classify_window(model, filtered, label_map, device)
            detection_counts[movement] += 1

            print(f"  >>> {movement.upper():<20} (conf={confidence:.2f})")
            if movement != "idle":
                send_movement_event(movement, confidence)

            elapsed = time.time() - loop_start
            sleep_time = INFERENCE_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n" + "="*50)
        print("  Session stopped by user.")
        print("\n  Detection summary:")
        for movement, count in detection_counts.items():
            print(f"    {movement:<20} : {count} times")
        print("="*50)

    finally:
        try:
            board.stop_stream()
            board.release_session()
            print("\n  Board disconnected cleanly.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
