"""Latency calibration kernel.

Runs detection + sequence alignment per audio channel and for video, then
estimates a constant presentation delay with explicit confidence intervals.
The same object drives the control panel: suggested corrections are derived
only from measured offsets, and failed/uncertain measurements feed back as a
verdict that disables blind delay adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import detect, match
from .media import MediaConfig
from .models import (
    CalibrationResult, ChannelResult, Detection, MatchPair, ScheduledEvent,
)

MIN_MATCHES = 4
SYNC_REQUIRED = True


def _summarise(
    kind: str, channel: int, pairs: list[MatchPair],
    dropped: list[int], unpaired: list[Detection],
    detections: list[Detection], events: list[ScheduledEvent],
    origin_known: bool,
) -> ChannelResult:
    nominal = {e.index: e.nominal_time for e in events}
    n_dup = sum(len(p.duplicate_times) for p in pairs)
    if len(pairs) >= 2:
        res = np.array([p.det_time - nominal[p.event_index]
                        for p in pairs], dtype=np.float64)
        offset = float(np.mean(res))
        std = float(np.std(res, ddof=1)) if len(res) > 1 else 0.0
        ci = 1.96 * std / np.sqrt(len(res)) if len(res) > 1 else 0.0
        # Detection-side uncertainty widens the interval honestly.
        obs_unc = float(np.mean([p.det_uncertainty for p in pairs]))
        half = ci + obs_unc
    else:
        offset = std = ci = 0.0
        half = float("inf")
        res = np.array([])
    return ChannelResult(
        kind=kind, channel=channel,
        n_matched=len(pairs), n_detections=len(detections),
        n_scheduled=len(events), n_dropped=len(dropped),
        n_duplicates=n_dup,
        n_rejected=sum(1 for d in detections if d.rejected_reason),
        residuals=[float(x) for x in res],
        offset=offset,
        offset_lo=offset - half if np.isfinite(half) else offset,
        offset_hi=offset + half if np.isfinite(half) else offset,
        jitter_std=std, ci95=ci, pairings=pairs,
    )


def calibrate(
    events: list[ScheduledEvent],
    audio_dets: list[list[Detection]],
    video_dets: list[Detection],
    cfg: MediaConfig,
    recorded_fps: float | None,
    origin_known: bool,
) -> CalibrationResult:
    interval = cfg.interval
    reasons: list[str] = []
    audio_results: list[ChannelResult] = []

    sync_seen_audio = any(any(d.is_sync for d in ch) for ch in audio_dets)
    if SYNC_REQUIRED and not sync_seen_audio:
        reasons.append("SYNC anchor missing in audio; alignment unanchored")

    for c, dets in enumerate(audio_dets):
        pairs, dropped, unpaired = match.align(events, dets, interval)
        res = _summarise("audio", c, pairs, dropped, unpaired, dets, events,
                         origin_known)
        audio_results.append(res)
        if res.n_matched < MIN_MATCHES:
            reasons.append(f"audio ch{c}: only {res.n_matched} matched events")
        if res.jitter_std > 0.02:
            reasons.append(f"audio ch{c}: high residual jitter "
                           f"{res.jitter_std*1000:.1f} ms")

    vpairs, vdropped, vunpaired = match.align(events, video_dets, interval)
    video_result = _summarise("video", -1, vpairs, vdropped, vunpaired,
                              video_dets, events, origin_known)
    if video_result.n_matched < MIN_MATCHES:
        reasons.append(f"video: only {video_result.n_matched} matched events")
    if recorded_fps and recorded_fps < 25:
        reasons.append(
            f"recorded {recorded_fps:g} fps: frame bracket "
            f"~{1000/recorded_fps:.1f} ms limits video timing precision")

    # Audio-to-video offset: video leads/lags relative to each audio channel.
    av_offsets = []
    for ar in audio_results:
        delta = ar.offset - video_result.offset
        lo = (ar.offset_lo - video_result.offset_hi)
        hi = (ar.offset_hi - video_result.offset_lo)
        av_offsets.append({
            "channel": ar.channel,
            "audio_minus_video_ms": delta * 1000.0,
            "ci95_lo_ms": lo * 1000.0,
            "ci95_hi_ms": hi * 1000.0,
            "interpretation": ("audio later than video" if delta > 0
                               else "audio earlier than video"),
        })

    # Suggested correction per channel: delay the leading path so they align.
    suggested: dict[str, float] = {}
    for av in av_offsets:
        d = av["audio_minus_video_ms"]
        suggested[f"audio_ch{av['channel']}_delay_ms"] = max(0.0, -d)
    suggested["video_delay_ms"] = max(
        0.0, max((a["audio_minus_video_ms"] for a in av_offsets), default=0.0))

    all_res = audio_results + [video_result]
    if any(r.n_matched < MIN_MATCHES for r in all_res):
        verdict = "fail"
    elif reasons or any(r.jitter_std > 0.02 for r in all_res):
        verdict = "warn"
    else:
        verdict = "pass"
    if not origin_known:
        reasons.append(
            "recording start origin unknown: only audio/video relative offset "
            "is measurable, absolute display/speaker delay not claimed")

    return CalibrationResult(
        verdict=verdict, reasons=reasons, audio=audio_results,
        video=video_result, av_offsets=av_offsets, origin_known=origin_known,
        fps_recorded=recorded_fps, sample_rate=cfg.sample_rate,
        suggested_delay_ms=suggested,
    )


def run_on_recording(
    audio: np.ndarray, luminance: np.ndarray, frame_times: np.ndarray,
    events: list[ScheduledEvent], cfg: MediaConfig, recorded_fps: float,
    origin_known: bool,
) -> tuple[CalibrationResult, list[list[Detection]], list[Detection]]:
    audio_dets, video_dets = detect.detect_all(
        audio, luminance, frame_times, recorded_fps, cfg)
    result = calibrate(events, audio_dets, video_dets, cfg,
                       recorded_fps, origin_known)
    return result, audio_dets, video_dets
