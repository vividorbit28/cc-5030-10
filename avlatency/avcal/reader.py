"""Read a real recording (e.g. camera capture of the played calibration clip)
through ffmpeg and produce the same arrays the simulated chain emits.

Only physical signal content is used. Wall-clock command/launch times are not
part of the data; if the recording's start cannot be tied to the media start,
the origin is marked unknown and the kernel reports relative A/V offset only.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from . import ffmpeg as ff
from .media import MediaConfig


def read_recording(path: str, cfg: MediaConfig, width: int = 320,
                   height: int = 180) -> dict:
    fps = ff.probe_fps(path)
    tmpdir = tempfile.mkdtemp(prefix="avcal_rec_")
    pcm = os.path.join(tmpdir, "audio.pcm")
    rgb = os.path.join(tmpdir, "video.rgb")
    ff.decode_audio(path, pcm, cfg.sample_rate, cfg.channels)
    ff.decode_video(path, rgb, width, height)

    pcm_bytes = np.fromfile(pcm, dtype="<i2")
    if pcm_bytes.size:
        audio = (pcm_bytes.astype(np.float32) / 32768.0).reshape(-1, cfg.channels)
    else:
        audio = np.zeros((0, cfg.channels), dtype=np.float32)

    raw = np.fromfile(rgb, dtype=np.uint8)
    frame_size = width * height * 3
    n_frames = raw.size // frame_size
    if n_frames:
        frames = raw[: n_frames * frame_size].reshape(n_frames, height, width, 3)
        luminance = frames.mean(axis=(1, 2, 3)) / 255.0
    else:
        luminance = np.zeros(0, dtype=np.float64)
    if fps is None:
        fps = cfg.fps
    frame_times = (np.arange(n_frames) + 0.5) / fps
    return {
        "audio": audio, "luminance": luminance,
        "frame_times": frame_times, "recorded_fps": fps,
    }
