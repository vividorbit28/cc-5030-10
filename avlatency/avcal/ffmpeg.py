"""Thin wrapper around a real ffmpeg binary (bundled via imageio-ffmpeg).

Every encode/decode here spawns an actual ffmpeg subprocess; nothing about
presentation timing is inferred from process launch times.
"""
from __future__ import annotations

import re
import shutil
import subprocess

_EXE = None


def ffmpeg_exe() -> str:
    global _EXE
    if _EXE is not None:
        return _EXE
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg  # type: ignore

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("no ffmpeg binary available") from exc
    _EXE = exe
    return exe


def run(args: list[str], stdin_bytes: bytes | None = None) -> tuple[bytes, bytes, int]:
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "info", *args]
    proc = subprocess.run(
        cmd, input=stdin_bytes, capture_output=True, check=False
    )
    return proc.stdout, proc.stderr, proc.returncode


def encode_mp4(
    raw_rgb: bytes,
    width: int,
    height: int,
    fps: float,
    wav_path: str,
    out_path: str,
) -> None:
    """Mux raw RGB24 frames (piped) and a WAV file into an H.264/AAC MP4."""
    args = [
        "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0",
        "-i", wav_path,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path,
    ]
    _, err, rc = run(args, stdin_bytes=raw_rgb)
    if rc != 0:
        raise RuntimeError(f"ffmpeg encode failed: {err.decode(errors='replace')[-800:]}")


_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s+fps")


def probe_fps(path: str) -> float | None:
    _, err, _ = run(["-i", path])
    text = err.decode(errors="replace")
    m = _FPS_RE.search(text)
    return float(m.group(1)) if m else None


def decode_audio(path: str, out_pcm: str, sample_rate: int, channels: int) -> None:
    args = [
        "-y", "-i", path, "-vn",
        "-ac", str(channels), "-ar", str(sample_rate),
        "-f", "s16le", out_pcm,
    ]
    _, err, rc = run(args)
    if rc != 0:
        raise RuntimeError(f"audio decode failed: {err.decode(errors='replace')[-800:]}")


def decode_video(path: str, out_rgb: str, width: int, height: int) -> None:
    vf = f"scale={width}:{height}"
    args = [
        "-y", "-i", path,
        "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", out_rgb,
    ]
    _, err, rc = run(args)
    if rc != 0:
        raise RuntimeError(f"video decode failed: {err.decode(errors='replace')[-800:]}")
