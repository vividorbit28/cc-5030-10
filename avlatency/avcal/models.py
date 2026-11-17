"""Shared data structures for the calibration kernel.

Time values are seconds unless suffixed ``_samples`` or ``_frame``.
Uncertainty is always carried alongside an estimate; a frame boundary is
reported as an interval [time_lo, time_hi], never as an exact instant.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ScheduledEvent:
    """One numbered calibration event as authored into the test media."""

    index: int
    is_sync: bool
    nominal_time: float          # intended presentation time in media timeline
    audio_offsets: list[float]   # per-channel schedule offset added at render
    flash_frames: list[int]      # frame indices rendered bright (source timeline)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Detection:
    """A single detected pulse or flash from the recording."""

    kind: str                    # 'audio' | 'video'
    channel: int                 # audio channel, else -1
    time: float                  # audio: point estimate; video: midpoint of bracket
    time_lo: float               # lower edge of the observation interval
    time_hi: float               # upper edge of the observation interval
    uncertainty: float           # 1-sigma-style estimate, seconds
    score: float                 # normalised strength 0..1 (audio SNR proxy)
    is_sync: bool = False        # sync pattern recognised by shape
    sample_index: int = -1       # audio: fractional grid origin sample (int)
    frame_index: int = -1        # video: first bright frame
    rejected_reason: str = ""    # non-empty when this candidate was rejected

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchPair:
    event_index: int
    is_sync: bool
    det_time: float
    det_lo: float
    det_hi: float
    det_uncertainty: float
    score: float
    duplicate_times: list[float] = field(default_factory=list)


@dataclass
class ChannelResult:
    kind: str                    # 'audio' | 'video'
    channel: int
    n_matched: int
    n_detections: int
    n_scheduled: int
    n_dropped: int               # scheduled events with no detection
    n_duplicates: int            # extra peaks grouped under one event
    n_rejected: int
    residuals: list[float]       # det_time - nominal_time (seconds)
    offset: float                # estimated constant delay (mean residual)
    offset_lo: float
    offset_hi: float
    jitter_std: float
    ci95: float
    pairings: list[MatchPair]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pairings"] = [p.__dict__ for p in self.pairings]
        return d


@dataclass
class SegmentInfo:
    """One interpretable piece of the residual series (offset + clock rate)."""

    t_lo: float                  # first fitted observation time (s)
    t_hi: float                  # last fitted observation time (s)
    t_ref: float                 # centring time: offset is quoted here
    n_fit: int
    n_holdout: int
    status: str                  # 'ok' | 'insufficient'
    offset: float | None         # seconds at t_ref; None when insufficient
    offset_ci95: float | None
    slope: float | None          # s per s (clock rate error); ppm = *1e6
    slope_ci95: float | None
    offset_var: float | None     # regression variances, for prediction bands
    slope_var: float | None
    gaps: list[list[float]]      # [lo, hi] intervals without confirming data
    classification: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExcludedPoint:
    """An outlier removed from fitting. Never silent: the numeric basis of
    the decision is kept so the exclusion can be audited later."""

    event_index: int
    time: float
    residual: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JumpInfo:
    """A detected step between two segments (device-restart style jump)."""

    time: float                  # boundary between segments (s)
    step: float                  # fitted level change at the boundary (s)
    step_ci95: float
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationPoint:
    """One held-out sample. It never took part in fitting; before/after
    residuals on these points are the honest check of the model."""

    event_index: int
    time: float
    residual_before: float       # raw det - nominal (uncorrected)
    residual_after: float | None # residual minus model prediction
    status: str                  # 'ok' | 'gap' | 'extrapolated' | 'insufficient'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftAnalysis:
    """Piecewise drift/jump fit of one channel's residual series."""

    segments: list[SegmentInfo]
    jumps: list[JumpInfo]
    excluded: list[ExcludedPoint]
    validation_points: list[ValidationPoint]
    rms_before: float | None     # holdout RMS of raw residuals (s)
    rms_after: float | None      # holdout RMS after model correction (s)
    max_abs_after: float | None
    t_lo: float                  # coverage range of the whole series
    t_hi: float
    notes: list[str]
    fit_series: list[list[float]] = field(default_factory=list)
    # [time_s, residual_s] pairs actually used for fitting (post-exclusion)

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "jumps": [j.to_dict() for j in self.jumps],
            "excluded": [e.to_dict() for e in self.excluded],
            "validation_points": [v.to_dict() for v in self.validation_points],
            "rms_before": self.rms_before,
            "rms_after": self.rms_after,
            "max_abs_after": self.max_abs_after,
            "t_lo": self.t_lo,
            "t_hi": self.t_hi,
            "notes": self.notes,
            "fit_series": self.fit_series,
        }


@dataclass
class CalibrationResult:
    verdict: str                 # 'pass' | 'warn' | 'fail'
    reasons: list[str]
    audio: list[ChannelResult]
    video: ChannelResult
    av_offsets: list[dict]       # audio-to-video offset per audio channel
    origin_known: bool           # False => absolute delay cannot be claimed
    fps_recorded: float | None
    sample_rate: int | None
    suggested_delay_ms: dict[str, float]
    drift: dict[str, DriftAnalysis] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reasons": self.reasons,
            "audio": [c.to_dict() for c in self.audio],
            "video": self.video.to_dict(),
            "av_offsets": self.av_offsets,
            "origin_known": self.origin_known,
            "fps_recorded": self.fps_recorded,
            "sample_rate": self.sample_rate,
            "suggested_delay_ms": self.suggested_delay_ms,
            "drift": {k: v.to_dict() for k, v in self.drift.items()},
        }
