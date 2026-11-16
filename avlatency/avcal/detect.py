"""Event detection from a real (or simulated) recording.

Audio: matched filtering against the authored pulse template. Peak timing is
interpolated between samples (parabolic) and the interpolation uncertainty is
reported; sample-grid resolution is never hidden.

Video: flashes are found against a moving-median baseline. Because a camera
integrates light over a frame, the true transition is bracketed by the two
frame times around the bright block. The detector reports that interval; it
must not claim sub-frame (e.g. millisecond) precision it cannot observe.

Rejection rules keep background hum and slow exposure drift out of the event
list instead of letting them pair by array index.
"""
from __future__ import annotations

import math

import numpy as np

from .media import MediaConfig
from .models import Detection

# Normalised-strength gate. Below this a candidate is treated as noise.
AUDIO_MIN_SCORE = 14.0    # matched-filter z-score gate (SNR); below = noise
VIDEO_MIN_CONTRAST = 0.25  # bright block must rise this far above baseline
VIDEO_MAX_RAMP_FRAMES = 4  # slower ramps are exposure drift, not flashes


def _template(cfg: MediaConfig, sync: bool) -> np.ndarray:
    sr = cfg.sample_rate
    if sync:
        n = int(round(cfg.sync_pulse_duration * sr))
        freq = cfg.sync_freq
    else:
        n = int(round(cfg.pulse_duration * sr))
        freq = cfg.pulse_freq
    tt = np.arange(n) / sr
    env = np.sin(np.pi * np.arange(n) / n)
    return (np.sin(2 * np.pi * freq * tt) * env).astype(np.float64)


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med))) + 1e-12


def detect_audio_channel(
    audio: np.ndarray, channel: int, cfg: MediaConfig
) -> list[Detection]:
    x = audio[:, channel].astype(np.float64)
    sr = cfg.sample_rate
    tpl = _template(cfg, sync=False)
    sync_tpl = _template(cfg, sync=True)

    # Cross-correlation by FFT against both authored pulse shapes.
    fft_n = 1 << math.ceil(math.log2(len(x) + max(len(tpl), len(sync_tpl))))
    corr = np.fft.irfft(np.fft.rfft(x, fft_n) *
                        np.conj(np.fft.rfft(tpl, fft_n)), fft_n)
    corr_n = np.fft.irfft(np.fft.rfft(x, fft_n) *
                          np.conj(np.fft.rfft(sync_tpl, fft_n)), fft_n)

    # Normalise each correlation by its template energy so amplitudes are
    # comparable; the noise floor is estimated from the robust MAD of the
    # normal-pulse response (a template that genuinely correlates throughout),
    # not the SYNC channel, whose floor is near zero and would inflate z.
    e_n = float(np.sqrt(np.sum(tpl * tpl))) + 1e-12
    e_s = float(np.sqrt(np.sum(sync_tpl * sync_tpl))) + 1e-12
    corr = corr / e_n
    corr_n = corr_n / e_s
    floor = 1.4826 * _mad(corr)
    z_n = corr / (floor + 1e-18)
    z_s = corr_n / (floor + 1e-18)

    # Both FFT cross-correlations peak at a lag equal to the pulse ONSET
    # sample (verified empirically), so they already share one time axis;
    # no template-length shift is applied.
    z_n = corr / (floor + 1e-18)
    z_s = corr_n / (floor + 1e-18)
    # Peak picking on local maxima rather than above-gate runs: under a strong
    # signal even noise samples clear an absolute gate, which would fuse many
    # events into one contiguous run. Local maxima + minimum separation stay
    # robust across a wide dynamic range.
    min_sep = max(1, int(0.060 * sr))  # suppress SYNC sidelobes; echoes >
    # this still survive as separate duplicate peaks

    def local_peaks(z: np.ndarray) -> list[tuple[int, float]]:
        cand = np.where((z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:])
                        & (z[1:-1] > AUDIO_MIN_SCORE))[0] + 1
        picked = [(int(j), float(z[j])) for j in cand]
        picked.sort(key=lambda kv: -kv[1])
        kept: list[tuple[int, float]] = []
        for j, v in picked:  # non-maximum suppression by separation
            if all(abs(j - k) >= min_sep for k, _ in kept):
                kept.append((j, v))
        kept.sort()
        return kept

    peaks_n = {j: v for j, v in local_peaks(z_n)}
    peaks_s = {j: v for j, v in local_peaks(z_s)}
    all_idx = sorted(set(peaks_n) | set(peaks_s))

    # Merge peaks belonging to ONE physical pulse. The long SYNC tone can
    # produce several local maxima (sidelobes) across its full duration, so a
    # SYNC-dominated neighbourhood is collapsed over the SYNC length; ordinary
    # pulses use the shorter pulse window. Genuine echoes beyond the window
    # survive and are later reported as duplicates (never index-paired).
    sync_win = max(1, int(cfg.sync_pulse_duration * sr))
    pulse_win = max(1, int(cfg.pulse_duration * sr))
    merged: list[list[int]] = []
    for j in all_idx:
        if merged:
            dom_sync = max(
                (peaks_s.get(k, -1e18) for k in merged[-1]), default=-1e18) > max(
                (peaks_n.get(k, -1e18) for k in merged[-1]), default=-1e18)
            win = sync_win if dom_sync else pulse_win
            if j - merged[-1][-1] <= win:
                merged[-1].append(j)
                continue
        merged.append([j])

    dets: list[Detection] = []
    for grp in merged:
        lo = grp[0]
        # Choose the stronger template across this pulse's neighbourhood.
        j_n = max(grp, key=lambda j: peaks_n.get(j, -1e18)) \
            if any(j in peaks_n for j in grp) else lo
        j_s = max(grp, key=lambda j: peaks_s.get(j, -1e18)) \
            if any(j in peaks_s for j in grp) else lo
        vn = peaks_n.get(j_n, -1e18)
        vs = peaks_s.get(j_s, -1e18)
        if max(vn, vs) < AUDIO_MIN_SCORE:
            continue
        if vs > vn:
            is_sync, j, snr, chosen = True, j_s, vs, z_s
        else:
            is_sync, j, snr, chosen = False, j_n, vn, z_n
        delta = 0.0
        if 0 < j < len(chosen) - 1:
            a, b, cval = chosen[j - 1], chosen[j], chosen[j + 1]
            denom = a - 2 * b + cval
            if abs(denom) > 1e-18:
                delta = float(np.clip(0.5 * (a - cval) / denom, -1.0, 1.0))
        t = (j + delta) / sr
        interp_unc = (0.25 if snr > 15 else 0.75) / sr
        dets.append(Detection(
            kind="audio", channel=channel, time=t,
            time_lo=t - interp_unc, time_hi=t + interp_unc,
            uncertainty=interp_unc,
            score=float(np.clip(snr / 25.0, 0.0, 1.0)),
            is_sync=bool(is_sync), sample_index=int(j),
        ))
    return dets


def detect_video(
    luminance: np.ndarray, frame_times: np.ndarray, recorded_fps: float
) -> list[Detection]:
    """Return flash detections. Each bright block yields one Detection whose
    [time_lo, time_hi] spans the frame bracket enclosing the transition."""
    lum = luminance.astype(np.float64)
    n = len(lum)
    t = np.arange(n, dtype=np.float64)
    # Robust de-trend: fit a line to the dark majority (frames below the
    # overall median), then subtract it. Slow exposure/auto-gain drift is a
    # ramp and disappears; brief flashes remain as upward transients. Frames
    # clipped by extreme drift are reported as a rejection condition below.
    dark = lum < np.median(lum)
    if dark.sum() >= 2:
        slope, intercept = np.polyfit(t[dark], lum[dark], 1)
    else:
        slope, intercept = 0.0, float(np.median(lum))
    detrended = lum - (slope * t + intercept)
    clipped = int(np.sum(lum >= 0.999))
    baseline = float(np.median(detrended[dark])) if dark.any() else 0.0
    scale = float(np.percentile(detrended, 95) - baseline) + 1e-9
    norm_contrast = (detrended - baseline) / scale

    gate = VIDEO_MIN_CONTRAST
    above = np.where(norm_contrast > gate)[0]
    dets: list[Detection] = []
    rejected = 0
    if above.size:
        groups = np.split(above, np.where(np.diff(above) > 2)[0] + 1)
        for g in groups:
            f_bright0, f_bright1 = int(g[0]), int(g[-1])
            width = f_bright1 - f_bright0 + 1
            peak = float(norm_contrast[f_bright0 : f_bright1 + 1].max())
            # Transient test robust to sensor noise: a flash jumps above the
            # gate and then returns toward baseline within a few frames. A slow
            # exposure ramp stays elevated / keeps climbing, so the post-block
            # level does not recover. Compare against the local dark level.
            tail_lo = min(n, f_bright1 + 2)
            tail_hi = min(n, f_bright1 + 5)
            recovered = (tail_hi > tail_lo and
                         float(np.mean(norm_contrast[tail_lo:tail_hi])) < gate)
            transient = recovered or width <= VIDEO_MAX_RAMP_FRAMES
            if peak < gate or not transient:
                rejected += 1
                continue
            # Transition bracket: last dark frame -> first bright frame.
            lo_idx = max(0, f_bright0 - 1)
            t_lo = float(frame_times[lo_idx])
            t_hi = float(frame_times[f_bright0])
            t_mid = 0.5 * (t_lo + t_hi)
            half_int = 0.5 / recorded_fps
            unc = (t_hi - t_lo) / 2.0
            # Video cannot identify the SYNC by shape at low frame rates;
            # SYNC anchoring is driven by the audio pulse shape, and the
            # sequence aligner matches video flashes purely by timing.
            dets.append(Detection(
                kind="video", channel=-1, time=t_mid,
                time_lo=t_lo - half_int, time_hi=t_hi + half_int,
                uncertainty=unc, score=float(np.clip(peak, 0.0, 1.0)),
                is_sync=False, frame_index=f_bright0,
            ))
    # Tag rejected slow drifts / clipping as diagnostics (kept, never paired).
    notes = []
    if rejected:
        notes.append(f"{rejected} slow-ramp/exposure candidate(s) rejected")
    if clipped > n * 0.25:
        notes.append(
            f"{clipped} frames clipped at peak brightness: exposure drift "
            "saturates the sensor, late flashes not observable")
    diag = Detection(
        kind="video", channel=-1, time=0.0, time_lo=0.0, time_hi=0.0,
        uncertainty=0.0, score=0.0, rejected_reason="; ".join(notes),
    )
    dets.sort(key=lambda d: d.time)
    return dets + ([diag] if notes else [])


def detect_all(
    audio: np.ndarray, luminance: np.ndarray, frame_times: np.ndarray,
    recorded_fps: float, cfg: MediaConfig,
) -> tuple[list[list[Detection]], list[Detection]]:
    audio_dets = [detect_audio_channel(audio, c, cfg)
                  for c in range(audio.shape[1])]
    video_dets = detect_video(luminance, frame_times, recorded_fps)
    return audio_dets, video_dets
