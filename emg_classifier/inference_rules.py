"""
inference_rules.py
------------------
Rule-based EMG classification using simple signal strength thresholds.
No model, no training data — just hardcoded ranges.

Thresholds (RMS signal strength):
    0  - 35   : IDLE
    36 - 75   : CLENCH
    76 - 94   : IDLE (dead zone)
    95+       : WRIST EXTENSION

Run:
    python inference_rules.py
"""

import sys
import time

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

# Thresholds on RMS signal strength
CLENCH_LOW         = 36
CLENCH_HIGH        = 75
EXTENSION_LOW      = 95
# =============================================================================


def classify_signal(rms: float) -> str:
    """Classify based on RMS signal strength thresholds."""
    if CLENCH_LOW <= rms <= CLENCH_HIGH:
        return "clench"
    elif rms >= EXTENSION_LOW:
        return "wrist_extension"
    else:
        return "idle"


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
    print("  EMG REAL-TIME INFERENCE (RULE-BASED)")
    print(f"  Board             : Cyton ({SERIAL_PORT})")
    print(f"  Inference interval: {INFERENCE_INTERVAL}s")
    print(f"  Thresholds:")
    print(f"    IDLE            : strength < {CLENCH_LOW} or {CLENCH_HIGH+1}-{EXTENSION_LOW-1}")
    print(f"    CLENCH          : {CLENCH_LOW} - {CLENCH_HIGH}")
    print(f"    WRIST EXTENSION : {EXTENSION_LOW}+")
    print("="*50 + "\n")

    print("  Connecting to board...")
    board = connect_board()

    print("\n  Starting inference loop. Press Ctrl+C to stop.\n")
    print("-" * 50)

    detection_counts = {"clench": 0, "wrist_extension": 0, "idle": 0}

    try:
        while True:
            loop_start = time.time()

            window = get_latest_window(board)
            if window is None:
                time.sleep(INFERENCE_INTERVAL)
                continue

            filtered = filter_window(window)

            # RMS = signal strength
            sig = filtered[:, 0]
            rms = float(np.sqrt(np.mean(sig ** 2)))

            movement = classify_signal(rms)
            detection_counts[movement] += 1

            if movement == "idle":
                print(f"  --- IDLE (strength={rms:.0f})")
            else:
                print(f"  >>> {movement.upper():<18} (strength={rms:.0f})")
                send_movement_event(movement, 1.0)

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
