"""Sequence/number based event matching.

Detections are never paired by array index. The authored schedule and the
recorded detections are two ordered sequences. Alignment uses *relative event
spacing* (which a constant presentation delay does not change), then a RANSAC
fit estimates the single global offset and rejects outliers such as echo
peaks. The distinctive SYNC event anchors the sequences. Missed scheduled
events and extra detections are reported explicitly.
"""
from __future__ import annotations

import numpy as np

from .models import Detection, ScheduledEvent, MatchPair

SPACING_TOLERANCE = 0.35   # fraction of the nominal interval
RANSAC_TOLERANCE = 0.060   # seconds for an inlier once offset is fitted
RANSAC_ITERS = 400


def _strip_diagnostics(dets: list[Detection]) -> list[Detection]:
    return [d for d in dets if d.rejected_reason == ""]


def _sequence_dp(events: list[ScheduledEvent], dets: list[Detection],
                 interval: float) -> list[tuple[int, int]]:
    """Align by spacing. Cost rewards shape agreement (SYNC) and penalises
    deviations from the nominal inter-event spacing. Returns (event_idx,
    detection_idx) pairs in schedule order."""
    n, m = len(events), len(dets)
    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            options = []
            if i > 0:  # skip a scheduled event (drop)
                options.append((dp[i - 1][j] + 1.0, 1))
            if j > 0:  # skip a detection (duplicate/background)
                options.append((dp[i][j - 1] + 1.0, 2))
            if i > 0 and j > 0:
                ev, det = events[i - 1], dets[j - 1]
                # Audio carries a recognisable SYNC pulse shape. Video cannot
                # identify SYNC by shape at low frame rates, so for video the
                # anchor is enforced purely through sequence position/timing.
                shape_ok = det.kind == "video" or (ev.is_sync == det.is_sync)
                cost = 0.0 if shape_ok else 0.8
                # compare spacing to previous matched pair if available
                pi, pj = i - 1, j - 1
                while pi > 0 and pj > 0 and back[pi][pj] != 0:
                    if back[pi][pj] == 1:
                        pi -= 1
                    else:
                        pj -= 1
                if pi > 0 and pj > 0:
                    prev_ev = events[pi - 1]
                    prev_det = dets[pj - 1]
                    exp_gap = ev.nominal_time - prev_ev.nominal_time
                    got_gap = det.time - prev_det.time
                    cost += abs(got_gap - exp_gap) / interval
                    if abs(got_gap - exp_gap) > SPACING_TOLERANCE * interval:
                        cost += 2.0
                options.append((dp[i - 1][j - 1] + cost, 0))
            best = min(options, key=lambda z: z[0])
            dp[i][j], back[i][j] = best

    pairs_idx: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and back[i][j] == 0:
            pairs_idx.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or back[i][j] == 1):
            i -= 1
        else:
            j -= 1
    pairs_idx.reverse()
    return pairs_idx


def _ransac_offset(nom: np.ndarray, obs: np.ndarray) -> tuple[float, np.ndarray]:
    diffs = obs - nom
    if len(diffs) < 2:
        return float(np.mean(diffs)) if len(diffs) else 0.0, np.ones(
            len(diffs), bool)
    rng = np.random.default_rng(0)
    best_inliers = np.abs(diffs - np.median(diffs)) < RANSAC_TOLERANCE
    best_count = int(best_inliers.sum())
    for _ in range(RANSAC_ITERS):
        pick = rng.integers(0, len(diffs))
        inl = np.abs(diffs - diffs[pick]) < RANSAC_TOLERANCE
        if int(inl.sum()) > best_count:
            best_count, best_inliers = int(inl.sum()), inl
    offset = float(np.mean(diffs[best_inliers]))
    return offset, best_inliers


def align(
    events: list[ScheduledEvent], dets: list[Detection], interval: float
) -> tuple[list[MatchPair], list[int], list[Detection]]:
    dets = _strip_diagnostics(dets)
    pairs_idx = _sequence_dp(events, dets, interval)
    used_det = {j for _, j in pairs_idx}

    if pairs_idx:
        nom = np.array([events[i].nominal_time for i, _ in pairs_idx])
        obs = np.array([dets[j].time for _, j in pairs_idx])
        _, inliers = _ransac_offset(nom, obs)
    else:
        inliers = np.array([], dtype=bool)

    pair_map: dict[int, MatchPair] = {}
    outliers: list[Detection] = []
    dropped: list[int] = []
    for k, (ei, dj) in enumerate(pairs_idx):
        ev, det = events[ei], dets[dj]
        if not inliers[k]:
            outliers.append(det)
            used_det.discard(dj)
            continue
        pair_map[ev.index] = MatchPair(
            event_index=ev.index, is_sync=ev.is_sync,
            det_time=det.time, det_lo=det.time_lo, det_hi=det.time_hi,
            det_uncertainty=det.uncertainty, score=det.score)
    matched_events = {ei for ei, _ in pairs_idx}
    dropped = sorted(e.index for e in events if e.index not in
                     {events[ei].index for ei, _ in pairs_idx})

    # Unpaired detections: echo/duplicate near a matched event is logged there;
    # anything else stays an unpaired background candidate (never index-paired).
    unpaired = [d for k, d in enumerate(dets) if k not in used_det]
    for d in unpaired[:]:
        near = min(pair_map.values(),
                   key=lambda p: abs(d.time - p.det_time), default=None)
        if near and abs(d.time - near.det_time) < 0.25:
            near.duplicate_times.append(d.time)
    final_unpaired = [d for d in unpaired if all(
        abs(d.time - p.det_time) >= 0.25 for p in pair_map.values())]
    return list(pair_map.values()), dropped, outliers + final_unpaired
