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
        }
