"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()
    transform = np.asarray(meas.sensor.sens_to_veh, dtype=float)
    R_s2v = transform[:3, :3]
    t_s2v = transform[:3, 3]

    pos_s = np.asarray(meas.z, dtype=float).ravel()[:3]
    pos_v = R_s2v @ pos_s + t_s2v

    x = np.asmatrix(np.zeros((6, 1), dtype=float))
    x[:3, 0] = pos_v.reshape(3, 1)

    P_pos = R_s2v @ np.asarray(meas.R, dtype=float) @ R_s2v.T
    P_vel = np.diag([
        float(params.sigma_p44)**2,
        float(params.sigma_p55)**2,
        float(params.sigma_p66)**2,
    ])

    P = np.asmatrix(np.zeros((6, 6), dtype=float))
    P[:3, :3] = P_pos
    P[3:, 3:] = P_vel

    score = 1.0 / float(params.window)
    state = "initialized"

    return {
        "x": x,
        "P": P,
        "state": state,
        "score": score,
    }


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    score = float(track["score"])
    state = str(track["state"])
    delta = 1.0 / float(params.window)

    if associated:
        score = min(1.0, score + delta)
        if score > float(params.confirmed_threshold):
            state = "confirmed"
        elif state != "confirmed":
            state = "tentative"
    else:
        score = score - delta
        # confirmed state is preserved after misses

    track["score"] = score
    track["state"] = state
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()
    P = np.asarray(track["P"], dtype=float)
    score = float(track["score"])
    state = str(track["state"])

    # Pxx or Pyy > max_P: delete regardless of score
    if P[0, 0] > float(params.max_P) or P[1, 1] > float(params.max_P):
        return True

    # confirmed tracks delete when score < delete_threshold
    if state == "confirmed":
        if score < float(params.delete_threshold):
            return True
    else:
        # unconfirmed tracks delete when score <= 0
        if score <= 0.0:
            return True

    return False
