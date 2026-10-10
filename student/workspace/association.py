"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from fusion_lab.workspace_support import get_tracking_params
from fusion_lab.workspace_loader import load_workspace_module

kalman = load_workspace_module("kalman")


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)
    dist_sq = gamma.T @ np.linalg.inv(S) @ gamma
    return float(np.asarray(dist_sq).item())


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    params = get_tracking_params()
    threshold = float(chi2.ppf(params.gating_threshold, sensor.dim_meas))
    return bool(mhd_sq <= threshold)


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    num_tracks = len(track_list)
    num_meas = len(meas_list)
    costs = np.full((num_tracks, num_meas), np.inf, dtype=float)

    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            if meas.sensor.in_fov(track.x):
                d_sq = mahalanobis_distance(track, meas)
                if chi2_gate(d_sq, meas.sensor):
                    costs[i, j] = d_sq

    return np.asmatrix(costs)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    matrix = np.asarray(association_matrix, dtype=float)
    if matrix.size == 0 or not np.isfinite(matrix).any():
        return (
            np.nan,
            np.nan,
            np.asmatrix(matrix),
            list(unassigned_tracks),
            list(unassigned_meas),
        )

    min_idx = int(np.argmin(matrix))
    min_row, min_col = np.unravel_index(min_idx, matrix.shape)

    if not np.isfinite(matrix[min_row, min_col]):
        return (
            np.nan,
            np.nan,
            np.asmatrix(matrix),
            list(unassigned_tracks),
            list(unassigned_meas),
        )

    selected_track = unassigned_tracks[min_row]
    selected_meas = unassigned_meas[min_col]

    new_matrix = np.delete(np.delete(matrix, min_row, axis=0), min_col, axis=1)
    remaining_tracks = [t for idx, t in enumerate(unassigned_tracks) if idx != min_row]
    remaining_meas = [m for idx, m in enumerate(unassigned_meas) if idx != min_col]

    return (
        selected_track,
        selected_meas,
        np.asmatrix(new_matrix),
        remaining_tracks,
        remaining_meas,
    )


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)

    if len(unassigned_tracks) == 0 or len(unassigned_meas) == 0:
        manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
        return

    cost_matrix = association_cost_matrix(unassigned_tracks, unassigned_meas)

    while cost_matrix.size > 0:
        (
            track,
            meas,
            cost_matrix,
            unassigned_tracks,
            unassigned_meas,
        ) = pick_next_pair(cost_matrix, unassigned_tracks, unassigned_meas)

        if isinstance(track, float) and np.isnan(track):
            break

        filter_obj.update(track, meas)
        if hasattr(manager, "handle_updated_track"):
            manager.handle_updated_track(track, sensor)

    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
