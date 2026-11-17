"""Latency calibration kernel.

Runs detection + sequence alignment per audio channel and for video, then
estimates a constant presentation delay with explicit confidence intervals.
That constant-delay estimate stays the baseline. On top of it, a piecewise
drift/jump analysis (module ``drift``) fits the continuous residual series
into interpretable segments - stable clock-rate error, device-restart jumps,
occasional detection errors - with audited outlier exclusion and independent
holdout validation. The same object drives the control panel: suggested
corrections are derived only from measured offsets, and failed/uncertain
measurements feed back as a verdict that disables blind delay adjustment.
"""
from __future__ import annotations

import numpy as np

from . import detect, drift as drift_mod, match
from .media import MediaConfig
from .models import (
    CalibrationResult, ChannelResult, Detection, DriftAnalysis, MatchPair,
    ScheduledEvent,
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

    # Piecewise drift/jump analysis of the continuous residual series. The
    # constant offsets above remain the baseline; this adds interpretable
    # segments, audited exclusions and independent holdout validation.
    nominal = {e.index: e.nominal_time for e in events}
    drift: dict[str, DriftAnalysis] = {}
    for res in all_res:
        if not res.pairings:
            continue
        key = f"audio_{res.channel}" if res.kind == "audio" else "video"
        times = np.array([nominal[p.event_index] for p in res.pairings])
        resid = np.array([p.det_time for p in res.pairings]) - times
        unc = np.array([p.det_uncertainty for p in res.pairings])
        eidx = [p.event_index for p in res.pairings]
        drift[key] = drift_mod.analyze(times, resid, unc, eidx)
    for key, da in drift.items():
        for note in da.notes:
            reasons.append(f"drift[{key}]: {note}")

    return CalibrationResult(
        verdict=verdict, reasons=reasons, audio=audio_results,
        video=video_result, av_offsets=av_offsets, origin_known=origin_known,
        fps_recorded=recorded_fps, sample_rate=cfg.sample_rate,
        suggested_delay_ms=suggested, drift=drift,
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


def export_playback_params(result: CalibrationResult) -> dict:
    """Exportable parameter set for a playback system.

    The baseline constant-delay measurement is kept as the reference; the
    piecewise drift/jump model is attached per channel with its coverage
    range, gaps and independent holdout validation. This only PRODUCES a
    parameter document - the tool never touches real device settings.
    """
    channels: dict[str, dict] = {}
    for key, da in result.drift.items():
        segs = []
        for s in da.segments:
            segs.append({
                "valid_range_s": [s.t_lo, s.t_hi],
                "status": s.status,
                "offset_ms_at_t_ref": (None if s.offset is None
                                       else s.offset * 1000.0),
                "offset_ci95_ms": (None if s.offset_ci95 is None
                                   else s.offset_ci95 * 1000.0),
                "t_ref_s": s.t_ref,
                "slope_ppm": (None if s.slope is None else s.slope * 1e6),
                "slope_ci95_ppm": (None if s.slope_ci95 is None
                                   else s.slope_ci95 * 1e6),
                "gaps_s": s.gaps,
                "classification": s.classification,
                "n_fit": s.n_fit,
            })
        channels[key] = {
            "coverage_s": [da.t_lo, da.t_hi],
            "segments": segs,
            "jumps": [{
                "time_s": j.time, "step_ms": j.step * 1000.0,
                "step_ci95_ms": j.step_ci95 * 1000.0, "note": j.note,
            } for j in da.jumps],
            "excluded_outliers": [e.to_dict() for e in da.excluded],
            "holdout_validation": {
                "n_points": len(da.validation_points),
                "rms_before_ms": (None if da.rms_before is None
                                  else da.rms_before * 1000.0),
                "rms_after_ms": (None if da.rms_after is None
                                 else da.rms_after * 1000.0),
                "max_abs_after_ms": (None if da.max_abs_after is None
                                     else da.max_abs_after * 1000.0),
            },
        }
    return {
        "format": "avcal-playback-params/1",
        "note": ("参数供播放系统采用；本工具不修改任何真实设备设置。"
                 "初版固定延迟测量保留为基准。"),
        "verdict": result.verdict,
        "baseline_constant_delay_ms": dict(result.suggested_delay_ms),
        "av_offsets_ms": result.av_offsets,
        "channels": channels,
        "extrapolation_policy": (
            "参数仅在各段 valid_range_s 内有效；范围外为外推，误差随距离"
            "增大，须按 drift.predict 的区间单独评估；空洞区间内不给确定"
            "参数。"),
    }
