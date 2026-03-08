"""
inference_knn.py
----------------
Real-time EMG movement classification using nearest-neighbor matching against
the preprocessed training windows. No neural network — just finds the most
similar window in the dataset and returns its label.

Uses RMS, MAV, waveform length, and zero crossing rate as features for
distance comparison (much more meaningful than raw sample-by-sample distance).

Run:
    python inference_knn.py
"""

import os
import sys
import time
import json
import argparse

import numpy as np

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
K_NEIGHBORS        = 5   # number of nearest neighbors to vote

DATA_PROC_DIR  = os.path.join(os.path.dirname(__file__), "data", "processed")
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "models", "label_map.json")

MOVEMENTS_ALL = ['strong_grip', 'wrist_extension', 'finger_spread']  # must match training labels
MOVEMENTS = ['wrist_extension', 'finger_spread']  # only classify these
ACTIVE_LABELS = [MOVEMENTS_ALL.index(m) for m in MOVEMENTS]
# =============================================================================


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract simple time-domain features from a (WINDOW_SIZE, 1) window.
    Returns a 1-D feature vector: [rms, mav, zcr, wl]
    """
    sig = window[:, 0]
    rms = np.sqrt(np.mean(sig ** 2))
    mav = np.mean(np.abs(sig))
    zcr = np.sum(np.diff(np.sign(sig)) != 0) / len(sig)
    wl  = np.sum(np.abs(np.diff(sig)))
    return np.array([rms, mav, zcr, wl], dtype=np.float32)


def load_training_data():
    """Load preprocessed X_train and y_train, extract features from each window."""
    X_train = np.load(os.path.join(DATA_PROC_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_PROC_DIR, "y_train.npy"))

    # Filter to only active movement classes
    mask = np.isin(y_train, ACTIVE_LABELS)
    X_train = X_train[mask]
    y_train = y_train[mask]
    print(f"  Loaded {len(X_train)} training windows (filtered to {MOVEMENTS})")

    # Extract features from every training window
    features = np.array([extract_features(X_train[i]) for i in range(len(X_train))])

    # Normalize features for fair distance comparison
    feat_mean = features.mean(axis=0)
    feat_std  = features.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    features_normed = (features - feat_mean) / feat_std

    print(f"  Feature shape per window: {features.shape[1]}")

    # Show per-class feature averages so you know what the training data looks like
    print(f"\n  Training data summary (avg signal per class):")
    for label in ACTIVE_LABELS:
        name = MOVEMENTS_ALL[label]
        mask = y_train == label
        if mask.sum() == 0:
            continue
        avg = features[mask].mean(axis=0)
        print(f"    {name:<18}: strength={avg[0]:.0f}  activity={avg[1]:.0f}  variability={avg[3]:.0f}  ({mask.sum()} windows)")

    # Compute typical within-class distances to calibrate similarity score
    # Use median distance between same-class pairs as the baseline
    all_dists = []
    for i in range(min(200, len(features_normed))):
        for j in range(i+1, min(200, len(features_normed))):
            all_dists.append(np.linalg.norm(features_normed[i] - features_normed[j]))
    median_dist = float(np.median(all_dists)) if all_dists else 1.0
    print(f"  Typical distance between training windows: {median_dist:.2f}")

    return features_normed, y_train, feat_mean, feat_std, features, median_dist


def classify_knn(window_features: np.ndarray, train_features: np.ndarray,
                 train_labels: np.ndarray, train_features_raw: np.ndarray,
                 k: int) -> tuple:
    """
    Find the k nearest training windows and vote on the label.

    Returns (movement_name, confidence, avg_dist, closest_raw_feats, closest_idx)
    """
    # Euclidean distance to all training windows
    dists = np.linalg.norm(train_features - window_features, axis=1)

    # Get k nearest
    nearest_idx = np.argpartition(dists, k)[:k]
    nearest_labels = train_labels[nearest_idx]
    nearest_dists = dists[nearest_idx]

    # Majority vote
    unique, counts = np.unique(nearest_labels, return_counts=True)
    winner_idx = np.argmax(counts)
    winner_label = int(unique[winner_idx])
    confidence = float(counts[winner_idx]) / k

    movement = MOVEMENTS_ALL[winner_label]
    avg_dist = float(nearest_dists.mean())

    # Find the single closest match
    closest_idx = int(nearest_idx[np.argmin(nearest_dists)])
    closest_raw_feats = train_features_raw[closest_idx]
    closest_dist = float(dists[closest_idx])

    return movement, confidence, avg_dist, closest_raw_feats, closest_idx, closest_dist


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


def main():
    print("\n" + "="*50)
    print("  EMG REAL-TIME INFERENCE (KNN MODE)")
    print(f"  Board             : Cyton ({SERIAL_PORT})")
    print(f"  K neighbors       : {K_NEIGHBORS}")
    print(f"  Inference interval: {INFERENCE_INTERVAL}s")
    print("="*50 + "\n")

    print("  Loading training data...")
    train_features, train_labels, feat_mean, feat_std, train_features_raw, median_dist = load_training_data()

    print("\n  Connecting to board...")
    board = connect_board()

    print("\n  Starting inference loop. Press Ctrl+C to stop.\n")
    print("-" * 50)

    detection_counts = {m: 0 for m in MOVEMENTS}

    try:
        while True:
            loop_start = time.time()

            window = get_latest_window(board)
            if window is None:
                time.sleep(INFERENCE_INTERVAL)
                continue

            filtered = filter_window(window)

            # Extract features and normalize with training stats
            feats = extract_features(filtered)
            feats_normed = (feats - feat_mean) / feat_std

            movement, confidence, _, _, _, closest_dist = classify_knn(
                feats_normed, train_features, train_labels, train_features_raw, K_NEIGHBORS
            )

            # Similarity: 100% = identical, 0% = very different
            # Scale relative to typical training distance so the number is meaningful
            similarity = max(0, 100 * (1 - closest_dist / (2 * median_dist)))
            similarity = min(similarity, 100)

            print(f"  ┌─── {movement.upper().replace('_', ' ')}")
            print(f"  │  Confidence : {confidence:.0%} ({int(confidence * K_NEIGHBORS)}/{K_NEIGHBORS} neighbors agree)")
            print(f"  │  Similarity : {similarity:.0f}% match to training data")
            print(f"  │  Your signal: strength={feats[0]:.0f}  activity={feats[1]:.0f}  variability={feats[3]:.0f}")
            print(f"  └───")
            print()

            if confidence >= 0.6:  # at least 3/5 neighbors agree
                send_movement_event(movement, confidence)
                detection_counts[movement] += 1

            elapsed = time.time() - loop_start
            sleep_time = INFERENCE_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n" + "="*50)
        print("  Session stopped by user.")
        print("\n  Detection summary:")
        for movement, count in detection_counts.items():
            print(f"    {movement:<18} : {count} times")
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
