"""
ui_server.py
------------
Web-based EMG visualization dashboard.

Runs the rule-based classifier and streams live state to a browser UI
via Server-Sent Events. Open http://localhost:8080 in your browser.

Run:
    python ui_server.py
"""

import sys
import time
import json
import threading

import numpy as np
from flask import Flask, Response, render_template_string, jsonify

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
INFERENCE_INTERVAL = 0.3
CH_1_IDX           = 1
WEB_PORT           = 8080

CLENCH_LOW         = 36
CLENCH_HIGH        = 75
EXTENSION_LOW      = 95
# =============================================================================

# Shared state between inference thread and web server
current_state = {
    "movement": "idle",
    "strength": 0,
    "history": [],       # last 60 strength readings for the graph
}
state_lock = threading.Lock()

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeuroHacks EMG</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  /* Header */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 40px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
  }
  .header h1 {
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
  }
  .header .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e88;
    display: inline-block;
    margin-right: 8px;
    animation: pulse-dot 2s infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .header .status-text {
    font-size: 13px;
    color: #666;
    display: flex;
    align-items: center;
  }

  /* Main content */
  .main {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }

  /* State card */
  .state-card {
    text-align: center;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .state-icon {
    width: 220px;
    height: 220px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 32px;
    font-size: 90px;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
  }
  .state-icon::before {
    content: '';
    position: absolute;
    inset: -4px;
    border-radius: 50%;
    border: 2px solid transparent;
    transition: all 0.4s ease;
  }

  /* State-specific styles */
  .state-idle .state-icon {
    background: rgba(100, 116, 139, 0.1);
    border: 1px solid rgba(100, 116, 139, 0.2);
  }
  .state-clench .state-icon {
    background: rgba(249, 115, 22, 0.12);
    border: 1px solid rgba(249, 115, 22, 0.3);
    box-shadow: 0 0 60px rgba(249, 115, 22, 0.15);
  }
  .state-clench .state-icon::before {
    border-color: rgba(249, 115, 22, 0.4);
    animation: ring-pulse 1.5s infinite;
  }
  .state-wrist_extension .state-icon {
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.3);
    box-shadow: 0 0 60px rgba(59, 130, 246, 0.15);
  }
  .state-wrist_extension .state-icon::before {
    border-color: rgba(59, 130, 246, 0.4);
    animation: ring-pulse 1.5s infinite;
  }

  @keyframes ring-pulse {
    0% { transform: scale(1); opacity: 1; }
    100% { transform: scale(1.15); opacity: 0; }
  }

  .state-label {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 8px;
    transition: color 0.3s ease;
  }
  .state-idle .state-label { color: #64748b; }
  .state-clench .state-label { color: #f97316; }
  .state-wrist_extension .state-label { color: #3b82f6; }

  .state-sub {
    font-size: 14px;
    color: #555;
    font-weight: 300;
  }

  /* Bottom bar */
  .bottom-bar {
    padding: 16px 40px 20px;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 40px;
  }

  /* Strength readout */
  .strength-readout {
    display: flex;
    align-items: baseline;
    gap: 6px;
    min-width: 120px;
  }
  .strength-value {
    font-size: 42px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    transition: color 0.3s ease;
  }
  .strength-unit {
    font-size: 14px;
    color: #555;
    font-weight: 300;
  }
  .state-idle .strength-value { color: #64748b; }
  .state-clench .strength-value { color: #f97316; }
  .state-wrist_extension .strength-value { color: #3b82f6; }

  /* Signal graph */
  .graph-container {
    flex: 1;
    height: 60px;
    position: relative;
  }
  .graph-canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
  .graph-thresholds {
    position: absolute;
    right: 0;
    top: 0;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    pointer-events: none;
  }

  /* Threshold legend */
  .legend {
    display: flex;
    gap: 24px;
    align-items: center;
    min-width: fit-content;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .legend-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
  }
  .legend-dot.idle { background: #64748b; }
  .legend-dot.clench { background: #f97316; }
  .legend-dot.extension { background: #3b82f6; }
</style>
</head>
<body>

<div class="header">
  <h1>NeuroHacks EMG</h1>
  <div class="status-text">
    <span class="status-dot"></span>
    Live — OpenBCI Cyton
  </div>
</div>

<div class="main">
  <div class="state-card" id="stateCard">
    <div class="state-icon" id="stateIcon">
      <span id="stateEmoji"></span>
    </div>
    <div class="state-label" id="stateLabel">CONNECTING...</div>
    <div class="state-sub" id="stateSub">Waiting for signal</div>
  </div>
</div>

<div class="bottom-bar" id="bottomBar">
  <div class="strength-readout">
    <span class="strength-value" id="strengthValue">0</span>
    <span class="strength-unit">RMS</span>
  </div>
  <div class="graph-container">
    <canvas class="graph-canvas" id="graphCanvas"></canvas>
  </div>
  <div class="legend">
    <div class="legend-item"><span class="legend-dot idle"></span>Idle</div>
    <div class="legend-item"><span class="legend-dot clench"></span>Clench</div>
    <div class="legend-item"><span class="legend-dot extension"></span>Extension</div>
  </div>
</div>

<script>
const ICONS = { idle: "\\u{1F590}", clench: "\\u270A", wrist_extension: "\\u{1FAF7}" };
const LABELS = { idle: "IDLE", clench: "CLENCH", wrist_extension: "WRIST EXTENSION" };
const SUBS = {
  idle: "Relaxed \\u2014 no movement detected",
  clench: "Fist clench detected",
  wrist_extension: "Wrist extension detected"
};

const card = document.getElementById('stateCard');
const icon = document.getElementById('stateIcon');
const emoji = document.getElementById('stateEmoji');
const label = document.getElementById('stateLabel');
const sub = document.getElementById('stateSub');
const strengthEl = document.getElementById('strengthValue');
const bottomBar = document.getElementById('bottomBar');
const canvas = document.getElementById('graphCanvas');
const ctx = canvas.getContext('2d');

let history = [];
const MAX_POINTS = 80;
let prevMovement = null;

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function getColor(val) {
  if (val >= 95) return '#3b82f6';
  if (val >= 36 && val <= 75) return '#f97316';
  return '#334155';
}

function drawGraph() {
  const w = canvas.width / window.devicePixelRatio;
  const h = canvas.height / window.devicePixelRatio;
  ctx.clearRect(0, 0, w, h);

  // Threshold lines
  const maxVal = 200;
  const clenchY = h - (36 / maxVal) * h;
  const clenchHiY = h - (75 / maxVal) * h;
  const extY = h - (95 / maxVal) * h;

  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 0.5;

  ctx.strokeStyle = 'rgba(249,115,22,0.25)';
  ctx.beginPath(); ctx.moveTo(0, clenchY); ctx.lineTo(w, clenchY); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, clenchHiY); ctx.lineTo(w, clenchHiY); ctx.stroke();

  ctx.strokeStyle = 'rgba(59,130,246,0.25)';
  ctx.beginPath(); ctx.moveTo(0, extY); ctx.lineTo(w, extY); ctx.stroke();

  ctx.setLineDash([]);

  if (history.length < 2) return;

  // Draw filled area
  const step = w / (MAX_POINTS - 1);
  ctx.beginPath();
  ctx.moveTo(0, h);
  for (let i = 0; i < history.length; i++) {
    const x = i * step;
    const y = h - Math.min(history[i] / maxVal, 1) * h;
    if (i === 0) ctx.lineTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.lineTo((history.length - 1) * step, h);
  ctx.closePath();

  const gradient = ctx.createLinearGradient(0, 0, 0, h);
  const lastVal = history[history.length - 1];
  const color = getColor(lastVal);
  gradient.addColorStop(0, color + '30');
  gradient.addColorStop(1, color + '05');
  ctx.fillStyle = gradient;
  ctx.fill();

  // Draw line
  ctx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const x = i * step;
    const y = h - Math.min(history[i] / maxVal, 1) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Dot at end
  const lastX = (history.length - 1) * step;
  const lastY = h - Math.min(lastVal / maxVal, 1) * h;
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(lastX, lastY, 7, 0, Math.PI * 2);
  ctx.strokeStyle = color + '44';
  ctx.lineWidth = 2;
  ctx.stroke();
}

const evtSource = new EventSource('/stream');
evtSource.onmessage = function(e) {
  const data = JSON.parse(e.data);
  const movement = data.movement;
  const strength = Math.round(data.strength);

  history.push(strength);
  if (history.length > MAX_POINTS) history.shift();

  // Update card
  card.className = 'state-card state-' + movement;
  bottomBar.className = 'bottom-bar state-' + movement;
  emoji.textContent = ICONS[movement] || ICONS.idle;
  label.textContent = LABELS[movement] || 'UNKNOWN';
  sub.textContent = SUBS[movement] || '';
  strengthEl.textContent = strength;

  // Add subtle scale bump on state change
  if (movement !== prevMovement && movement !== 'idle') {
    icon.style.transform = 'scale(1.08)';
    setTimeout(() => { icon.style.transform = 'scale(1)'; }, 300);
  }
  prevMovement = movement;

  drawGraph();
};
</script>
</body>
</html>
"""


def classify_signal(rms: float) -> str:
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


def inference_loop(board: BoardShim):
    """Run inference in a background thread, updating shared state."""
    while True:
        loop_start = time.time()

        window = get_latest_window(board)
        if window is not None:
            filtered = filter_window(window)
            sig = filtered[:, 0]
            rms = float(np.sqrt(np.mean(sig ** 2)))
            movement = classify_signal(rms)

            with state_lock:
                current_state["movement"] = movement
                current_state["strength"] = rms
                current_state["history"].append(rms)
                if len(current_state["history"]) > 80:
                    current_state["history"].pop(0)

            if movement != "idle":
                send_movement_event(movement, 1.0)

            print(f"  {'>>>' if movement != 'idle' else '---'} {movement.upper():<18} (strength={rms:.0f})")

        elapsed = time.time() - loop_start
        sleep_time = INFERENCE_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/state')
def state():
    """JSON endpoint for Unity to poll. Returns {"movement": "clench", "strength": 52.3}"""
    with state_lock:
        return jsonify({
            "movement": current_state["movement"],
            "strength": round(current_state["strength"], 1),
        })


@app.route('/stream')
def stream():
    def generate():
        while True:
            with state_lock:
                data = {
                    "movement": current_state["movement"],
                    "strength": round(current_state["strength"], 1),
                }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.3)
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def main():
    print("\n" + "="*50)
    print("  NEUROHACKS EMG DASHBOARD")
    print(f"  Board             : Cyton ({SERIAL_PORT})")
    print(f"  Inference interval: {INFERENCE_INTERVAL}s")
    print(f"  Dashboard         : http://localhost:{WEB_PORT}")
    print(f"  Thresholds:")
    print(f"    IDLE            : < {CLENCH_LOW} or {CLENCH_HIGH+1}-{EXTENSION_LOW-1}")
    print(f"    CLENCH          : {CLENCH_LOW} - {CLENCH_HIGH}")
    print(f"    WRIST EXTENSION : {EXTENSION_LOW}+")
    print("="*50 + "\n")

    print("  Connecting to board...")
    board = connect_board()

    print("  Starting inference thread...")
    inference_thread = threading.Thread(target=inference_loop, args=(board,), daemon=True)
    inference_thread.start()

    print(f"  Opening dashboard at http://localhost:{WEB_PORT}\n")
    print("-" * 50)

    try:
        app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            board.stop_stream()
            board.release_session()
            print("\n  Board disconnected cleanly.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
