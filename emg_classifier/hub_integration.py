"""
hub_integration.py
------------------
Connects the EMG classifier output to the Flask hub that orchestrates the
Unity VR arm and TENS stimulation device.

Usage:
    from hub_integration import send_movement_event, send_feedback

    send_movement_event("strong_grip", confidence=0.92)
    send_feedback(rating=1)

The Flask hub is expected to be running at HUB_URL (default: http://localhost:5000).
All requests are fire-and-forget with a short timeout so they never block the
inference loop.
"""

import requests

# =============================================================================
# SETTINGS
# =============================================================================
HUB_URL            = "http://localhost:5000"
REQUEST_TIMEOUT    = 0.2    # Seconds — must be short enough not to block inference

# MOVEMENT_TO_PATTERN defines the route and TENS pulse sequence for each class.
# tens_sequence is a list of (delay_ms, channel) tuples.
MOVEMENT_TO_PATTERN = {
    "strong_grip": {
        "route_id":     0,
        "tens_sequence": [(0, 1), (100, 2), (200, 3)],
    },
    "wrist_extension": {
        "route_id":     1,
        "tens_sequence": [(0, 2), (150, 1), (300, 3)],
    },
    "finger_spread": {
        "route_id":     2,
        "tens_sequence": [(0, 3), (100, 1), (200, 2)],
    },
}
# =============================================================================

# Module-level state: tracks the last movement for use in send_feedback
_last_movement = None


def send_movement_event(movement: str, confidence: float) -> bool:
    """
    POST a detected movement event to the Flask hub.

    Looks up the pattern for the given movement, constructs the payload, and
    fires a non-blocking POST. Prints a confirmation or error to the terminal.

    Args:
        movement:   Movement class name (must be a key in MOVEMENT_TO_PATTERN).
        confidence: Softmax confidence score from the classifier (0.0 – 1.0).

    Returns:
        True if the request succeeded (HTTP 2xx), False otherwise.
    """
    global _last_movement
    _last_movement = movement

    pattern = MOVEMENT_TO_PATTERN.get(movement)
    if pattern is None:
        print(f"[HUB] Unknown movement '{movement}' — no pattern defined, skipping.")
        return False

    payload = {
        "event":         "move_detected",
        "movement":      movement,
        "confidence":    round(confidence, 4),
        "route_id":      pattern["route_id"],
        "tens_sequence": pattern["tens_sequence"],
    }

    url = f"{HUB_URL}/event"
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if response.ok:
            print(f"  [HUB] Sent '{movement}' -> route {pattern['route_id']}  (HTTP {response.status_code})")
            return True
        else:
            print(f"  [HUB] Server rejected event (HTTP {response.status_code}): {response.text[:80]}")
            return False
    except requests.exceptions.Timeout:
        print(f"  [HUB] Request timed out after {REQUEST_TIMEOUT}s — hub may be slow or down.")
        return False
    except requests.exceptions.ConnectionError:
        print(f"  [HUB] Could not reach hub at {url}")
        print("    -> Is the Flask hub running? (python hub.py or similar)")
        return False
    except Exception as e:
        print(f"  [HUB] Unexpected error sending event: {e}")
        return False


def send_feedback(rating: int) -> bool:
    """
    POST user feedback (e.g. from a Unity button press) to the Flask hub.

    Associates the rating with the most recently detected movement so the hub
    can log it or use it for online learning.

    Args:
        rating: Integer feedback score (e.g. 1 = good, 0 = bad, or a 1-5 scale).

    Returns:
        True if the request succeeded, False otherwise.
    """
    payload = {
        "rating":   rating,
        "movement": _last_movement,
    }

    url = f"{HUB_URL}/feedback"
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if response.ok:
            print(f"  [HUB] Feedback sent: rating={rating} for '{_last_movement}'  (HTTP {response.status_code})")
            return True
        else:
            print(f"  [HUB] Server rejected feedback (HTTP {response.status_code}): {response.text[:80]}")
            return False
    except requests.exceptions.Timeout:
        print(f"  [HUB] Feedback request timed out after {REQUEST_TIMEOUT}s.")
        return False
    except requests.exceptions.ConnectionError:
        print(f"  [HUB] Could not reach hub at {url} for feedback.")
        return False
    except Exception as e:
        print(f"  [HUB] Unexpected error sending feedback: {e}")
        return False
