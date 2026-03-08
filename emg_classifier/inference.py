"""
inference.py
------------
Real-time EMG movement classification using a trained CNN model and a live
OpenBCI Cyton board stream.

How it works:
  1. Load the trained model and label map from models/
  2. Connect to the Cyton board via BrainFlow
  3. Every INFERENCE_INTERVAL seconds, grab the latest WINDOW_SIZE samples
  4. Filter (bandpass + notch) and run through the model
  5. Always fire handle_prediction() with the top class regardless of confidence

Press Ctrl+C to stop. A session summary is printed on exit.

Run:
    python inference.py
"""

import os
import sys
import time
import json

import numpy as np
import torch

from preprocess import bandpass_filter, notch_filter, normalize_window, BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ, NOTCH_Q, SAMPLE_RATE
from train import EMGClassifier, N_CHANNELS, WINDOW_SIZE, NUM_CLASSES
from hub_integration import send_movement_event

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# =============================================================================
# SETTINGS
# =============================================================================
# Connects directly to the Cyton board via its USB dongle — must match collect.py.
# macOS: /dev/tty.usbserial-* | Windows: COM3 (check Device Manager)
SERIAL_PORT           = "COM4"  # <-- set your Cyton USB dongle port
CONFIDENCE_THRESHOLD  = 0.70            # Minimum softmax probability to fire an event
INFERENCE_INTERVAL    = 0.8             # Seconds between classification attempts (match window duration)
CH_1_IDX              = 1              # BrainFlow channel index for channel 1

MODELS_DIR     = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH     = os.path.join(MODELS_DIR, "emg_classifier.pt")
LABEL_MAP_PATH = os.path.join(MODELS_DIR, "label_map.json")
# =============================================================================


def load_model(model_path: str, label_map_path: str, device: torch.device) -> tuple:
    """
    Load the trained CNN and label map from disk.

    Args:
        model_path:     Path to the saved .pt model weights file.
        label_map_path: Path to label_map.json.
        device:         PyTorch device to load the model onto.

    Returns:
        Tuple (model, label_map) where label_map is a dict {int: str}.

    Raises:
        SystemExit: If model or label map files are missing.
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("  -> Run train.py first to train a model.")
        sys.exit(1)

    if not os.path.exists(label_map_path):
        print(f"[ERROR] Label map not found: {label_map_path}")
        print("  -> Run train.py first — it saves label_map.json automatically.")
        sys.exit(1)

    with open(label_map_path, "r") as f:
        raw_map = json.load(f)
    label_map = {int(k): v for k, v in raw_map.items()}

    num_classes = len(label_map)
    model = EMGClassifier(N_CHANNELS, WINDOW_SIZE, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"  Model loaded from: {model_path}")
    print(f"  Classes: {label_map}")
    return model, label_map


def connect_board() -> BoardShim:
    """
    Connect to the OpenBCI Cyton board via its USB dongle serial port.

    Returns:
        Active BoardShim instance.

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
        time.sleep(2)  # Let the buffer fill
        return board
    except Exception as e:
        print(f"\n[ERROR] Could not connect to Cyton board.")
        print("  -> Is the USB dongle plugged in?")
        print("  -> Is the OpenBCI GUI closed? (It holds the serial port)")
        print(f"  -> Set SERIAL_PORT in inference.py (currently: '{SERIAL_PORT}')")
        print("  -> macOS: check /dev/tty.usbserial-* | Windows: check Device Manager")
        print(f"  -> BrainFlow error: {e}")
        sys.exit(1)


def get_latest_window(board: BoardShim) -> np.ndarray | None:
    """
    Wait for WINDOW_SIZE fresh samples, then consume them from the buffer.

    Uses get_board_data() (which flushes) so each inference window contains
    entirely new data — no overlap with the previous prediction.

    Args:
        board: Active BoardShim instance.

    Returns:
        Array of shape (WINDOW_SIZE, 1) or None if not enough data yet.
    """
    if board.get_board_data_count() < WINDOW_SIZE:
        return None

    data = board.get_board_data()  # flush buffer — fresh samples only
    ch1  = data[CH_1_IDX, -WINDOW_SIZE:]
    return ch1.reshape(-1, 1).astype(np.float32)


def filter_window(window: np.ndarray) -> np.ndarray:
    """
    Apply bandpass + notch filter, then z-score normalize a single window.

    Args:
        window: Array of shape (WINDOW_SIZE, 1).

    Returns:
        Filtered and normalized array of the same shape.
    """
    filtered = np.zeros_like(window)
    for ch in range(window.shape[1]):
        sig = window[:, ch]
        sig = bandpass_filter(sig, BANDPASS_LOW, BANDPASS_HIGH, SAMPLE_RATE)
        sig = notch_filter(sig, NOTCH_FREQ, NOTCH_Q, SAMPLE_RATE)
        filtered[:, ch] = sig
    return normalize_window(filtered)


def classify_window(
    window: np.ndarray,
    model: torch.nn.Module,
    label_map: dict,
    device: torch.device,
) -> tuple:
    """
    Run a filtered window through the model and return the top prediction.

    Args:
        window:    Filtered array of shape (WINDOW_SIZE, 1).
        model:     Loaded EMGClassifier.
        label_map: {int: str} mapping from integer class to movement name.
        device:    Torch device.

    Returns:
        Tuple (movement_name, confidence) where confidence is a float in [0, 1].
    """
    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_class  = int(np.argmax(probs))
    confidence = float(probs[top_class])
    movement   = label_map.get(top_class, f"class_{top_class}")
    return movement, confidence


def handle_prediction(movement: str, confidence: float) -> None:
    """
    Fire every interval: print the prediction and post to the hub.

    Args:
        movement:   Predicted movement class name.
        confidence: Model's softmax confidence score.
    """
    print(f"  DETECTED: {movement}  ({confidence:.0%} confident)")
    send_movement_event(movement, confidence)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*50)
    print("  EMG REAL-TIME INFERENCE")
    print(f"  Device            : {device}")
    print(f"  Board             : Cyton ({SERIAL_PORT})")
    print(f"  Inference interval: {INFERENCE_INTERVAL}s (fires every interval)")
    print("="*50 + "\n")

    print("  Loading model...")
    model, label_map = load_model(MODEL_PATH, LABEL_MAP_PATH, device)

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

            # Debug: print raw signal stats to verify board is sending real data
            print(f"  [DEBUG] raw: min={window.min():.1f}  max={window.max():.1f}  std={window.std():.1f}")

            filtered_window      = filter_window(window)
            print(f"  [DEBUG] filtered: min={filtered_window.min():.1f}  max={filtered_window.max():.1f}  std={filtered_window.std():.1f}")
            movement, confidence = classify_window(filtered_window, model, label_map, device)

            # Debug: print all class probabilities
            tensor = torch.tensor(filtered_window, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                all_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            prob_str = "  ".join(f"{label_map[i]}={all_probs[i]:.1%}" for i in range(len(all_probs)))
            print(f"  [DEBUG] probs: {prob_str}")

            if confidence >= CONFIDENCE_THRESHOLD:
                handle_prediction(movement, confidence)
                detection_counts[movement] += 1
            else:
                print(f"  (low confidence: {movement} {confidence:.0%} — skipped)")

            # Pace the loop to INFERENCE_INTERVAL
            elapsed = time.time() - loop_start
            sleep_time = INFERENCE_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n" + "="*50)
        print("  Session stopped by user.")
        print("\n  Detection summary:")
        for movement, count in detection_counts.items():
            print(f"    {movement:<14} : {count} times")
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
