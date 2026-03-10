# NeuroHacks — EMG-Powered VR Hand Control

A real-time EMG (electromyography) classification system that detects intended hand and wrist movements from nerve activity via an OpenBCI Cyton board, and then classifies them with a 1D CNN, and drives VR arm animations in Unity.

## How It Works

1. **Collect** EMG signals from your forearm while performing hand gestures
2. **Train** a 1D convolutional neural network on the processed signals
3. **Infer** real-time movements and stream to AR application for psychological support
4. **Apply** recommended signals via TENs devices to alleviate physical pain

### Supported Movements

| Movement | Description |
|---|---|
| `strong_grip` | Clenching the hand |
| `wrist_extension` | Extending the wrist upward |
| `idle` | No significant movement |

## Project Structure

```
emg_classifier/
├── collect.py          # Record EMG data from OpenBCI Cyton
├── preprocess.py       # Filter, segment, and extract features
├── train.py            # Train the 1D CNN classifier
├── inference.py        # Real-time ML-based classification
├── ui_server.py        # Web dashboard with live signal visualization
├── hub_integration.py  # Bridge to Flask hub (VR + TENS routing)
├── data/
│   ├── raw/            # Raw session CSVs
│   └── processed/      # Train/val/test NumPy arrays
└── models/             # Saved model weights + label map

unity/
├── EMGReceiver.cs      # Polls inference server for movement state
└── EMGArmController.cs # Drives VR arm animations from EMG events
```


## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Cyton Board │────▶│  ui_server   │────▶│  Unity VR    │
│  (BrainFlow) │     │  (Flask SSE) │     │  (polling)   │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Flask Hub   │
                     │  (TENS ctrl) │
                     └──────────────┘
```
