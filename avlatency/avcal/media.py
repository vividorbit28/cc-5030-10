"""Generate calibration media: numbered white flashes on mid-grey plus indexed
audio pulses. The MP4/WAV are produced by a real ffmpeg subprocess so that a
later encode->decode round trip exercises genuine media timing.
"""
from __future__ import annotations

import json
import math
import os
import wave
from dataclasses import dataclass

import numpy as np

from . import ffmpeg as ff
from .models import ScheduledEvent

# ---------------------------------------------------------------------------
# Minimal 5x7 bitmap font (rows top->bottom), enough for on-screen numbers.
# ---------------------------------------------------------------------------
_GLYPHS = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    " ": ["00000"] * 7,
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
}

GREY = 96
WHITE = 255
BLACK = 16


@dataclass
class MediaConfig:
    width: int = 640
    height: int = 360
    fps: float = 30.0
    sample_rate: int = 48000
    channels: int = 2
    n_events: int = 24
    interval: float = 0.5
    flash_duration: float = 0.0667   # ~2 frames at 30 fps
    pulse_duration: float = 0.015
    pulse_freq: float = 1000.0
    sync_freq: float = 300.0
    sync_pulse_duration: float = 0.08
    seed: int = 1234


def build_schedule(cfg: MediaConfig) -> list[ScheduledEvent]:
    """Deterministic schedule: first event is a SYNC anchor, then numbered
    events with small fixed-seed timing jitter (never random per run)."""
    rng = np.random.default_rng(cfg.seed)
    events: list[ScheduledEvent] = []
    start = 1.0
    for k in range(cfg.n_events):
        is_sync = k == 0
        jitter = 0.0 if is_sync else float(rng.uniform(-0.06, 0.06))
        t = start + k * cfg.interval + jitter
        f0 = int(round(t * cfg.fps))
        n_flash = max(1, int(round(cfg.flash_duration * cfg.fps)))
        frames = list(range(f0, f0 + n_flash)) if is_sync else list(
            range(f0, f0 + max(1, int(round(cfg.flash_duration * cfg.fps))))
        )
        events.append(
            ScheduledEvent(
                index=k,
                is_sync=is_sync,
                nominal_time=t,
                audio_offsets=[0.0] * cfg.channels,
                flash_frames=frames,
            )
        )
    return events


def _draw_text(frame: np.ndarray, text: str, x: int, y: int, scale: int, color) -> None:
    pen = x
    for ch in text:
        glyph = _GLYPHS.get(ch, _GLYPHS[" "])
        for row, bits in enumerate(glyph):
            for col, bit in enumerate(bits):
                if bit == "1":
                    frame[
                        y + row * scale : y + (row + 1) * scale,
                        pen + col * scale : pen + (col + 1) * scale,
                    ] = color
        pen += 6 * scale


def render_frames(cfg: MediaConfig, events: list[ScheduledEvent]) -> np.ndarray:
    """Return uint8 RGB array (n_frames, H, W, 3). Every frame carries its
    source-frame number; flash frames show the event index in large digits."""
    duration = events[-1].nominal_time + 0.8
    n_frames = int(math.ceil(duration * cfg.fps))
    frames = np.full(
        (n_frames, cfg.height, cfg.width, 3), GREY, dtype=np.uint8
    )
    bright = {f: ev for ev in events for f in ev.flash_frames}
    for fi in range(n_frames):
        frame = frames[fi]
        _draw_text(frame, f"F{fi:04d}", 8, 8, 2, (WHITE, WHITE, WHITE))
        if fi in bright:
            ev = bright[fi]
            frame[:] = WHITE
            label = "SYNC" if ev.is_sync else f"EV {ev.index:02d}"
            _draw_text(frame, label, 40, cfg.height // 2 - 28, 5,
                       (BLACK, BLACK, BLACK))
            _draw_text(frame, f"EVENT {ev.index:02d}", 40,
                       cfg.height // 2 + 24, 3, (BLACK, BLACK, BLACK))
    return frames


def _tone(sr: int, dur: float, freq: float) -> np.ndarray:
    n = int(round(dur * sr))
    tt = np.arange(n) / sr
    env = np.sin(np.pi * np.arange(n) / n)  # Hann window, no clicks
    return np.sin(2 * np.pi * freq * tt) * env * 0.9


def render_audio(
    cfg: MediaConfig, events: list[ScheduledEvent]
) -> np.ndarray:
    """Interleaved float32 audio. SYNC uses a distinctive lower/longer tone;
    normal events use a short 1 kHz pulse. Per-channel schedule offsets from
    the event list are honoured."""
    duration = events[-1].nominal_time + 0.8
    total = int(round(duration * cfg.sample_rate))
    audio = np.zeros((total, cfg.channels), dtype=np.float32)
    for ev in events:
        if ev.is_sync:
            tone = _tone(cfg.sample_rate, cfg.sync_pulse_duration, cfg.sync_freq)
        else:
            tone = _tone(cfg.sample_rate, cfg.pulse_duration, cfg.pulse_freq)
        for ch in range(cfg.channels):
            start = int(round((ev.nominal_time + ev.audio_offsets[ch])
                              * cfg.sample_rate))
            end = min(total, start + len(tone))
            audio[start:end, ch] += tone[: end - start]
    return audio


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype("<i2")
    interleaved = pcm16.reshape(-1)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(audio.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved.tobytes())


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return data.reshape(-1, ch), sr


def generate(cfg: MediaConfig, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    events = build_schedule(cfg)
    frames = render_frames(cfg, events)
    audio = render_audio(cfg, events)

    wav_path = os.path.join(outdir, "calibration.wav")
    mp4_path = os.path.join(outdir, "calibration.mp4")
    meta_path = os.path.join(outdir, "schedule.json")
    write_wav(wav_path, audio, cfg.sample_rate)
    ff.encode_mp4(
        frames.tobytes(), cfg.width, cfg.height, cfg.fps, wav_path, mp4_path
    )
    meta = {
        "config": cfg.__dict__,
        "events": [ev.to_dict() for ev in events],
        "media": {"mp4": mp4_path, "wav": wav_path},
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return {"frames": frames, "audio": audio, "events": events,
            "paths": {"mp4": mp4_path, "wav": wav_path, "meta": meta_path}}
