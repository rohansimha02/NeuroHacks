"""
preprocess.py
-------------
Loads raw EMG CSV files, applies signal filtering, segments into overlapping
windows, extracts time-domain features, and saves train/val/test splits as
NumPy arrays ready for model training.

Pipeline:
  1. Load all session_*.csv files from data/raw/
  2. Bandpass filter (20-450 Hz) + notch filter (60 Hz)
  3. Segment each 2-second recording into overlapping 200-sample windows
  4. Extract features per window (RMS, MAV, ZCR, WL) + keep raw signal
  5. Encode labels as integers
  6. Split by window into train/val/test (70/15/15) — works from a single session
  7. Save X_*.npy and y_*.npy to data/processed/

Run:
    python preprocess.py
"""

import os
import glob

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

# =============================================================================
# SETTINGS
# =============================================================================
SAMPLE_RATE    = 250          # Hz — Cyton default, must match collect.py
BANDPASS_LOW   = 20           # Hz — lower cutoff (removes motion artifact)
BANDPASS_HIGH  = 120          # Hz — upper cutoff (must be below Nyquist of 125 Hz at 250 Hz sample rate)
NOTCH_FREQ     = 60           # Hz — US power line noise
NOTCH_Q        = 30           # Quality factor for notch filter
WINDOW_SIZE    = 200          # Samples per window (0.8 seconds)
WINDOW_STEP    = 50           # Step between windows (75% overlap)
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.15
# TEST_FRAC is implicitly 1 - TRAIN_FRAC - VAL_FRAC = 0.15

MOVEMENTS      = ['strong_grip', 'wrist_extension', 'finger_spread']
LABEL_MAP      = {m: i for i, m in enumerate(MOVEMENTS)}

DATA_RAW_DIR   = os.path.join(os.path.dirname(__file__), "data", "raw")
DATA_PROC_DIR  = os.path.join(os.path.dirname(__file__), "data", "processed")
# =============================================================================


def bandpass_filter(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """
    Apply a 4th-order Butterworth bandpass filter using zero-phase filtering.

    Args:
        signal: 1-D array of raw EMG samples.
        low:    Lower cutoff frequency in Hz.
        high:   Upper cutoff frequency in Hz.
        fs:     Sampling rate in Hz.

    Returns:
        Filtered signal as a 1-D NumPy array.
    """
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, freq: float, q: float, fs: float) -> np.ndarray:
    """
    Apply a notch filter to remove a single frequency (e.g. power line noise).

    Args:
        signal: 1-D array of EMG samples.
        freq:   Frequency to remove in Hz.
        q:      Quality factor — higher Q means a narrower notch.
        fs:     Sampling rate in Hz.

    Returns:
        Filtered signal as a 1-D NumPy array.
    """
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, signal)


def filter_channels(data: np.ndarray) -> np.ndarray:
    """
    Apply bandpass + notch filtering to every channel in a data array.

    Args:
        data: Array of shape (n_samples, n_channels).

    Returns:
        Filtered array of the same shape.
    """
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        sig = data[:, ch]
        sig = bandpass_filter(sig, BANDPASS_LOW, BANDPASS_HIGH, SAMPLE_RATE)
        sig = notch_filter(sig, NOTCH_FREQ, NOTCH_Q, SAMPLE_RATE)
        filtered[:, ch] = sig
    return filtered


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract time-domain EMG features from a single window.

    Features computed per channel: RMS, MAV, Zero Crossing Rate, Waveform Length.
    The raw window is also flattened and appended for CNN input.

    Args:
        window: Array of shape (window_size, n_channels).

    Returns:
        1-D feature vector. The raw window is the primary input; features are
        concatenated at the end and available if you want a feature-based model.
        For the CNN we return the raw window reshaped as (window_size, n_channels).
    """
    n_ch = window.shape[1]
    features = []
    for ch in range(n_ch):
        sig = window[:, ch]
        rms = np.sqrt(np.mean(sig ** 2))
        mav = np.mean(np.abs(sig))
        zcr = np.sum(np.diff(np.sign(sig)) != 0) / len(sig)
        wl  = np.sum(np.abs(np.diff(sig)))
        features.extend([rms, mav, zcr, wl])
    return np.array(features, dtype=np.float32)


def normalize_window(window: np.ndarray) -> np.ndarray:
    """
    Mean-subtract each channel (remove DC offset) without scaling by std.

    This removes baseline drift between sessions while preserving amplitude
    differences between movements — which is the primary distinguishing
    feature with a single EMG channel.

    Args:
        window: Array of shape (window_size, n_channels).

    Returns:
        Mean-centered array of the same shape.
    """
    normed = np.zeros_like(window)
    for ch in range(window.shape[1]):
        sig = window[:, ch]
        normed[:, ch] = sig - sig.mean()
    return normed


def segment_recording(data: np.ndarray, label: int) -> tuple:
    """
    Slice a single 2-second EMG recording into overlapping windows.

    Args:
        data:  Array of shape (n_samples, n_channels) — one recording.
        label: Integer class label for this recording.

    Returns:
        Tuple of (windows, labels) where:
          - windows: array of shape (n_windows, WINDOW_SIZE, n_channels)
          - labels:  array of shape (n_windows,) filled with `label`
    """
    windows = []
    start = 0
    while start + WINDOW_SIZE <= len(data):
        w = data[start : start + WINDOW_SIZE]
        windows.append(normalize_window(w))
        start += WINDOW_STEP

    if not windows:
        return np.empty((0, WINDOW_SIZE, data.shape[1])), np.empty((0,), dtype=int)

    windows_arr = np.array(windows, dtype=np.float32)
    labels_arr  = np.full(len(windows), label, dtype=np.int64)
    return windows_arr, labels_arr


def load_session(filepath: str) -> tuple:
    """
    Load a single session CSV and return filtered windows with labels.

    Args:
        filepath: Path to a session_*.csv file.

    Returns:
        Tuple (X, y) where X has shape (n_windows, WINDOW_SIZE, 1)
        and y has shape (n_windows,).
    """
    df = pd.read_csv(filepath)

    all_windows = []
    all_labels  = []

    for sample_idx in df["sample_index"].unique():
        chunk = df[df["sample_index"] == sample_idx]
        label_str = chunk["label"].iloc[0]
        if label_str not in LABEL_MAP:
            print(f"  [WARNING] Unknown label '{label_str}' in {filepath} — skipping.")
            continue
        label = LABEL_MAP[label_str]

        raw = chunk[["channel_1"]].values.astype(np.float32)
        filtered = filter_channels(raw)

        windows, labels = segment_recording(filtered, label)
        if len(windows) > 0:
            all_windows.append(windows)
            all_labels.append(labels)

    if not all_windows:
        return np.empty((0, WINDOW_SIZE, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels,  axis=0)
    return X, y


def print_class_distribution(y: np.ndarray, split_name: str) -> None:
    """
    Print the number and percentage of samples per class for a split.

    Args:
        y:          Integer label array.
        split_name: Human-readable name (e.g. 'Train').
    """
    total = len(y)
    print(f"\n  {split_name} class distribution ({total} windows total):")
    for label, name in enumerate(MOVEMENTS):
        count = int(np.sum(y == label))
        pct   = count / total * 100 if total > 0 else 0
        print(f"    {name:<12} : {count:>5} ({pct:.1f}%)")


def split_windows(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Split all windows into train/val/test by shuffling at the window level.

    This is the default for hackathon use (single session). The slight data
    leakage from overlapping windows is acceptable when all data comes from
    one person in one sitting — the model generalises to that same person anyway.

    Args:
        X: Array of shape (n_windows, WINDOW_SIZE, n_channels).
        y: Integer label array of shape (n_windows,).

    Returns:
        Tuple (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    n = len(X)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)

    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train : n_train + n_val]
    test_idx  = idx[n_train + n_val :]

    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
    )


def main():
    os.makedirs(DATA_PROC_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, "session_*.csv")))
    if not csv_files:
        print(f"[ERROR] No session CSV files found in {DATA_RAW_DIR}")
        print("  -> Run collect.py first to record data.")
        return

    print(f"\n  Found {len(csv_files)} session file(s):")
    for f in csv_files:
        print(f"    {os.path.basename(f)}")

    all_Xs, all_ys = [], []
    for fp in csv_files:
        print(f"\n  Loading: {os.path.basename(fp)} ...")
        X, y = load_session(fp)
        print(f"    -> {len(X)} windows extracted")
        all_Xs.append(X)
        all_ys.append(y)

    X_all = np.concatenate(all_Xs, axis=0)
    y_all = np.concatenate(all_ys, axis=0)
    print(f"\n  Total windows across all sessions: {len(X_all)}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_windows(X_all, y_all)

    print(f"\n  Split summary (window-level, seed=42):")
    print(f"    Train : {len(X_train)} windows ({TRAIN_FRAC:.0%})")
    print(f"    Val   : {len(X_val)} windows ({VAL_FRAC:.0%})")
    print(f"    Test  : {len(X_test)} windows ({1 - TRAIN_FRAC - VAL_FRAC:.0%})")

    print_class_distribution(y_train, "Train")
    print_class_distribution(y_val,   "Validation")
    print_class_distribution(y_test,  "Test")

    np.save(os.path.join(DATA_PROC_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_PROC_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(DATA_PROC_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_PROC_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_PROC_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(DATA_PROC_DIR, "y_test.npy"),  y_test)

    print(f"\n  Saved processed arrays to: {DATA_PROC_DIR}")
    print(f"    X_train.npy  {X_train.shape}")
    print(f"    X_val.npy    {X_val.shape}")
    print(f"    X_test.npy   {X_test.shape}")
    print(f"    y_train.npy  {y_train.shape}")
    print(f"    y_val.npy    {y_val.shape}")
    print(f"    y_test.npy   {y_test.shape}")
    print("\n  Preprocessing complete!\n")


if __name__ == "__main__":
    main()
