"""
collect.py
----------
Guided EMG data collection session using an OpenBCI Cyton board via BrainFlow.

Usage:
    python collect.py           # full session (N_SAMPLES per movement)
    python collect.py --test    # quick test (3 samples per movement)

The session records 2 seconds of EMG data per sample, labels it with the
current movement, and saves everything to data/raw/session_{timestamp}.csv.
"""

import os
import sys
import time
import argparse
import csv
from datetime import datetime

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

# =============================================================================
# SETTINGS — edit these to match your hardware setup
# =============================================================================
# Connects directly to the Cyton board via its USB dongle serial port.
# Close the OpenBCI GUI before running — it holds the serial port.
SERIAL_PORT    = "COM4"  # <-- set your Cyton USB dongle port
SAMPLE_RATE    = 250              # Hz — Cyton default
RECORD_SECONDS = 2                # Duration per sample in seconds
N_SAMPLES      = 50               # Number of samples to collect per movement
TEST_SAMPLES   = 3                # Samples per movement in --test mode
CH_1_IDX       = 1                # BrainFlow index for N1P
COUNTDOWN_SECS = 3                # Seconds to count down before recording

MOVEMENTS = ['strong_grip', 'wrist_extension', 'finger_spread']

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
# =============================================================================


def connect_board() -> BoardShim:
    """
    Connect directly to the OpenBCI Cyton board via its USB dongle serial port.

    The OpenBCI GUI must be closed so it doesn't hold the serial port.
    On macOS the port looks like: /dev/tty.usbserial-DM00D0GN
    On Windows it looks like: COM3

    Returns:
        An initialized and streaming BoardShim instance.

    Raises:
        SystemExit: If the board cannot be reached.
    """
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)

    try:
        board.prepare_session()
        board.start_stream()
        print(f"  Connected to Cyton board on {SERIAL_PORT}")
        time.sleep(2)  # Let the buffer fill before reading
        return board
    except Exception as e:
        print(f"\n[ERROR] Could not connect to Cyton board.")
        print("  -> Is the USB dongle plugged in?")
        print("  -> Is the OpenBCI GUI closed? (It holds the serial port)")
        print(f"  -> Set SERIAL_PORT in collect.py (currently: '{SERIAL_PORT}')")
        print("  -> macOS: check /dev/tty.usbserial-* | Windows: check Device Manager")
        print(f"  -> BrainFlow error: {e}")
        sys.exit(1)


def record_sample(board: BoardShim, n_samples: int) -> np.ndarray:
    """
    Record exactly n_samples of raw EMG data from channel 1.

    Clears the buffer before recording, then waits until enough new samples
    have accumulated.

    Args:
        board:     Active BoardShim instance.
        n_samples: Number of samples to record (e.g. 500 for 2 seconds at 250 Hz).

    Returns:
        NumPy array of shape (n_samples, 1) — single channel.
    """
    # Flush any stale data in the ring buffer
    board.get_board_data()

    # Wait until the buffer fills with fresh samples
    start = time.time()
    timeout = n_samples / SAMPLE_RATE * 3  # 3x expected duration as safety margin
    while True:
        ready = board.get_board_data_count()
        if ready >= n_samples:
            break
        if time.time() - start > timeout:
            print("[WARNING] Timed out waiting for samples — using what's available.")
            break
        time.sleep(0.01)

    data = board.get_board_data()          # shape: (n_board_channels, n_samples)
    ch1 = data[CH_1_IDX, -n_samples:]
    return ch1.reshape(-1, 1)  # shape: (n_samples, 1)


def wait_for_space(prompt: str) -> None:
    """
    Print a prompt and wait for the user to press Enter before continuing.

    Args:
        prompt: Text to display to the user.
    """
    input(prompt)


def countdown(seconds: int) -> None:
    """
    Print a human-readable countdown to the terminal.

    Args:
        seconds: Number of seconds to count down from.
    """
    for i in range(seconds, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print("  >>> HOLD THE MOVEMENT <<<")


def run_collection_session(n_samples_per_movement: int) -> list:
    """
    Run the full guided data collection session.

    Iterates over each movement in MOVEMENTS, collects n_samples_per_movement
    labeled windows, and returns all collected rows.

    Args:
        n_samples_per_movement: How many 2-second recordings to capture per class.

    Returns:
        List of dicts with keys: sample_index, channel_1, label.
        Each row is one sample (one time step within a recording window).
    """
    board = connect_board()
    all_rows = []
    global_sample_idx = 0

    try:
        for movement in MOVEMENTS:
            print(f"\n{'='*50}")
            print(f"  MOVEMENT: {movement.upper()}")
            print(f"  You will record {n_samples_per_movement} samples.")
            print(f"{'='*50}")

            for i in range(n_samples_per_movement):
                print(f"\n  Sample {i + 1}/{n_samples_per_movement}")
                wait_for_space("  Press ENTER when ready to record a sample...")

                countdown(COUNTDOWN_SECS)

                n_raw = int(SAMPLE_RATE * RECORD_SECONDS)
                window = record_sample(board, n_raw)  # shape: (n_raw, 1)

                print("  RELAX")

                # Append each time-step as a row
                for step in range(len(window)):
                    all_rows.append({
                        "sample_index": global_sample_idx,
                        "channel_1":    window[step, 0],
                        "label":        movement,
                    })
                global_sample_idx += 1

                print(f"  Saved {len(window)} data points (label='{movement}')")

    except KeyboardInterrupt:
        print("\n\n[INFO] Session interrupted by user.")
    finally:
        try:
            board.stop_stream()
            board.release_session()
            print("\n  Board disconnected cleanly.")
        except Exception:
            pass

    return all_rows


def save_session(rows: list) -> str:
    """
    Save collected rows to a timestamped CSV file in data/raw/.

    Args:
        rows: List of dicts from run_collection_session().

    Returns:
        Absolute path to the saved CSV file.
    """
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.csv"
    filepath = os.path.join(DATA_RAW_DIR, filename)

    fieldnames = ["sample_index", "channel_1", "label"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return filepath


def main():
    parser = argparse.ArgumentParser(description="OpenBCI Cyton EMG data collector")
    parser.add_argument(
        "--test",
        action="store_true",
        help=f"Quick test mode: collect only {TEST_SAMPLES} samples per movement",
    )
    args = parser.parse_args()

    n_samples = TEST_SAMPLES if args.test else N_SAMPLES

    print("\n" + "="*50)
    print("  EMG DATA COLLECTION SESSION")
    print(f"  Movements : {MOVEMENTS}")
    print(f"  Samples   : {n_samples} per movement")
    print(f"  Duration  : {RECORD_SECONDS}s per sample")
    print(f"  Port      : {SERIAL_PORT}")
    if args.test:
        print("  [TEST MODE ACTIVE]")
    print("="*50 + "\n")

    rows = run_collection_session(n_samples)

    if not rows:
        print("[WARNING] No data was collected. Nothing saved.")
        return

    filepath = save_session(rows)
    n_windows = len(set(r["sample_index"] for r in rows))
    print(f"\n{'='*50}")
    print(f"  Session complete!")
    print(f"  Total windows collected : {n_windows}")
    print(f"  Total data rows         : {len(rows)}")
    print(f"  Saved to                : {filepath}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
