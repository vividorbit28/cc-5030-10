"""Controllable recording chain.

Applies *physical* effects to the generated media: per-channel audio delays,
video display delay, additive noise, dropped events, echo (extra peaks),
slow exposure drift, frame-rate conversion, dropped frames, recorder clock
drift (linear rate error), restart-style delay steps and single-event
mis-matches. These model what a camera/microphone capture path does; command
scheduling times never enter the picture.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .models import ScheduledEvent


@dataclass
class ChainConfig:
    sample_rate: int = 48000
    channels: int = 2
    fps_source: float = 30.0
    fps_recorded: float = 30.0
    audio_delay: list[float] = field(default_factory=lambda: [0.120, 0.120])
    video_delay: float = 0.180
    audio_noise_std: float = 0.01
    background_hum: bool = True
    dropped_events: list[int] = field(default_factory=list)
    echo_delay: float = 0.0          # >0 creates duplicate audio peaks
    echo_gain: float = 0.6
    exposure_drift: float = 0.0      # fraction of full scale per second
    brightness_noise_std: float = 0.01
    dropped_frames: list[int] = field(default_factory=list)
    clock_drift_ppm: float = 0.0     # recorder clock rate error (audio+video)
    delay_steps: list[tuple[float, float]] = field(default_factory=list)
    # (time_s, extra_delay_s): device-restart style jumps applied to both
    # audio and video presentation after the given media time
    misaligned_events: list[tuple[int, float]] = field(default_factory=list)
    # (event_index, shift_s): one event's audio pulse physically relocated,
    # modelling a single detection/match error
    seed: int = 7


@dataclass
class ChainOutput:
    audio: np.ndarray                # (n_samples, channels) float32
    luminance: np.ndarray            # (n_recorded_frames,) float 0..1
    frame_times: np.ndarray          # exposure-midpoint time of each frame
    recorded_fps: float
    sample_rate: int
    ground_truth: dict
    config: dict


def _delay_signal(x: np.ndarray, delay_s: float, sr: int) -> np.ndarray:
    """Fractional-sample delay by linear interpolation (delay >= 0)."""
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    d = delay_s * sr
    di = int(np.floor(d))
    frac = d - di
    if di >= n:
        return out
    src = x.astype(np.float64)
    seg_len = n - di
    out[di : di + seg_len] += src[:seg_len] * (1.0 - frac)
    if di + 1 < n:
        out[di + 1 : di + 1 + seg_len - 1] += src[: seg_len - 1] * frac
    return out


def process_audio(
    source: np.ndarray, events: list[ScheduledEvent], cfg: ChainConfig, rng
) -> np.ndarray:
    n_src, ch = source.shape
    out = np.zeros_like(source, dtype=np.float64)
    drop = set(cfg.dropped_events)
    misaligned = dict(cfg.misaligned_events)
    for c in range(ch):
        sig = source[:, c].astype(np.float64).copy()
        # Remove dropped events at the true event windows (window = pulse*3)
        for ev in events:
            if ev.index in drop:
                t0 = ev.nominal_time + ev.audio_offsets[c]
                s0 = int((t0 - 0.005) * cfg.sample_rate)
                s1 = int((t0 + 0.05) * cfg.sample_rate)
                sig[max(0, s0) : min(len(sig), s1)] = 0.0
        # Mis-matched event: physically relocate its pulse in the source so
        # the detection lands at a wrong time (a real observation error, not
        # a bookkeeping trick).
        for ev in events:
            if ev.index in misaligned:
                shift = misaligned[ev.index]
                t0 = ev.nominal_time + ev.audio_offsets[c]
                s0 = int((t0 - 0.005) * cfg.sample_rate)
                s1 = int((t0 + 0.06) * cfg.sample_rate)
                s0, s1 = max(0, s0), min(len(sig), s1)
                seg = sig[s0:s1].copy()
                sig[s0:s1] = 0.0
                sh = int(round(shift * cfg.sample_rate))
                d0, d1 = s0 + sh, s1 + sh
                if 0 <= d0 < len(sig):
                    seg = seg[: max(0, min(len(seg), len(sig) - d0))]
                    sig[d0 : d0 + len(seg)] += seg
        delayed = _delay_signal(sig, cfg.audio_delay[c], cfg.sample_rate)
        # Device-restart jumps: the restart happens at a media time; in the
        # recorded timeline that is media_time + current delay. Presentation
        # drops out during the restart and resumes with the extra delay
        # stacked on (the tail must be copied before zeroing - a view would
        # be erased by its own shift).
        for t_step, extra_d in sorted(cfg.delay_steps):
            s_step = int(round((t_step + cfg.audio_delay[c])
                               * cfg.sample_rate))
            d_extra = int(round(extra_d * cfg.sample_rate))
            if 0 <= s_step < len(delayed) and d_extra > 0:
                tail = delayed[s_step : len(delayed) - d_extra].copy()
                delayed[s_step:] = 0.0
                delayed[s_step + d_extra : s_step + d_extra + len(tail)] = tail
        if cfg.echo_delay > 0:
            echo = _delay_signal(sig, cfg.audio_delay[c] + cfg.echo_delay,
                                 cfg.sample_rate)
            delayed = delayed + cfg.echo_gain * echo
        out[:, c] = delayed
    # Recorder clock drift: the capture clock runs at (1+eps) x nominal, so
    # sample i holds the source at i/(sr*(1+eps)); the detector still reads
    # times off the nominal grid and sees a linearly drifting offset.
    if cfg.clock_drift_ppm:
        eps = cfg.clock_drift_ppm * 1e-6
        src_grid = np.arange(n_src) / (cfg.sample_rate * (1.0 + eps))
        t_src = np.arange(n_src) / cfg.sample_rate
        for c in range(ch):
            out[:, c] = np.interp(src_grid, t_src, out[:, c])
    # Additive capture noise + mains hum (background to be rejected/ignored)
    if cfg.audio_noise_std > 0:
        out += rng.normal(0, cfg.audio_noise_std, out.shape)
    if cfg.background_hum:
        tt = np.arange(n_src) / cfg.sample_rate
        hum = 0.004 * np.sin(2 * np.pi * 50 * tt)
        out += hum[:, None]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _source_luminance(frames: np.ndarray) -> np.ndarray:
    # mean RGB per frame, normalised 0..1
    return frames.mean(axis=(1, 2, 3)) / 255.0


def process_video(
    frames: np.ndarray, events: list[ScheduledEvent], cfg: ChainConfig, rng
) -> tuple[np.ndarray, np.ndarray]:
    src_lum = _source_luminance(frames)
    n_src = len(src_lum)
    src_times = (np.arange(n_src) + 0.5) / cfg.fps_source
    duration = n_src / cfg.fps_source

    n_rec = int(np.floor(duration * cfg.fps_recorded))
    rec_mid = (np.arange(n_rec) + 0.5) / cfg.fps_recorded
    # exposure window = one recorded frame, integrate source frames inside it
    half = 0.5 / cfg.fps_recorded
    lum = np.empty(n_rec, dtype=np.float64)
    for i, mid in enumerate(rec_mid):
        lo, hi = mid - half, mid + half
        sel = (src_times >= lo) & (src_times < hi)
        lum[i] = src_lum[sel].mean() if sel.any() else np.interp(
            mid, src_times, src_lum
        )

    # Physical display delay shifts when flashes actually appear. Recorder
    # clock drift scales the true capture time of frame i, and each restart
    # step adds its extra delay after its step time.
    eps = cfg.clock_drift_ppm * 1e-6
    capture_t = rec_mid / (1.0 + eps)
    extra = np.zeros(n_rec)
    for t_step, d_extra in cfg.delay_steps:
        extra += np.where(rec_mid >= t_step + cfg.video_delay, d_extra, 0.0)
    delayed_t = capture_t - cfg.video_delay - extra
    lum = np.interp(delayed_t, src_times, src_lum, left=src_lum[0])

    # Slow exposure/auto-gain drift (ramp, NOT a transient) must be rejected.
    if cfg.exposure_drift:
        lum = lum + cfg.exposure_drift * rec_mid
    if cfg.brightness_noise_std > 0:
        lum = lum + rng.normal(0, cfg.brightness_noise_std, n_rec)
    lum = np.clip(lum, 0.0, 1.0)

    # Dropped recorded frames: delete those samples and their timestamps so
    # timing gaps remain real (no silent re-indexing onto a uniform grid).
    keep = np.ones(n_rec, dtype=bool)
    for f in cfg.dropped_frames:
        if 0 <= f < n_rec:
            keep[f] = False
    return lum[keep], rec_mid[keep]


def run_chain(
    frames: np.ndarray,
    audio: np.ndarray,
    events: list[ScheduledEvent],
    cfg: ChainConfig,
) -> ChainOutput:
    rng = np.random.default_rng(cfg.seed)
    rec_audio = process_audio(audio, events, cfg, rng)
    lum, times = process_video(frames, events, cfg, rng)
    gt = {
        "audio_delay": list(cfg.audio_delay),
        "video_delay": cfg.video_delay,
        "dropped_events": list(cfg.dropped_events),
        "fps_recorded": cfg.fps_recorded,
        "clock_drift_ppm": cfg.clock_drift_ppm,
        "delay_steps": [list(s) for s in cfg.delay_steps],
        "misaligned_events": [list(m) for m in cfg.misaligned_events],
    }
    return ChainOutput(
        audio=rec_audio,
        luminance=lum,
        frame_times=times,
        recorded_fps=cfg.fps_recorded,
        sample_rate=cfg.sample_rate,
        ground_truth=gt,
        config=asdict(cfg),
    )
