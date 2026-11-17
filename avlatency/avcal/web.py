"""Multi-client calibration control panel.

A dependency-free HTTP server (stdlib only) that drives the calibration
kernel. Any device on the network can open the panel, trigger generation /
simulation / real-recording analysis, inspect waveforms and detected events,
and apply delay corrections. Controls are driven solely by kernel results; a
failed or unanchored measurement returns a failure state and blocks blind
delay adjustment.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import reader as reader_mod
from .calibrate import export_playback_params, run_on_recording
from .chain import ChainConfig, run_chain
from .media import MediaConfig, generate

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _pairs(value) -> list[tuple[float, float]]:
    """Normalise [[a, b], ...] params to a list of float tuples."""
    out = []
    for item in value or []:
        a, b = item
        out.append((float(a), float(b)))
    return out


class JobStore:
    """In-memory state shared across all connected clients/panels."""

    def __init__(self, workdir: str):
        self.workdir = workdir
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.applied_delay: dict[str, float] = {}
        self.current_job: str | None = None
        self.validation_log: list[dict] = []  # one entry per applied correction

    def create(self, kind: str, params: dict) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "kind": kind, "status": "queued",
               "params": params, "error": None, "result": None,
               "series": None, "media": None, "delay_applied": dict(self.applied_delay)}
        with self.lock:
            self.jobs[job_id] = job
            self.current_job = job_id
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            return self.jobs.get(job_id)

    def latest(self) -> dict | None:
        with self.lock:
            return self.jobs.get(self.current_job) if self.current_job else None


def _downsample(arr: np.ndarray, max_points: int) -> list[float]:
    if arr.size <= max_points:
        return [float(x) for x in arr]
    idx = np.linspace(0, arr.size - 1, max_points).astype(int)
    return [float(arr[i]) for i in idx]


def _series_payload(job: dict, audio, luminance, frame_times,
                    audio_dets, video_dets, cfg: MediaConfig) -> dict:
    # One audio waveform series per channel (downsampled for the browser).
    wave = []
    for c in range(audio.shape[1]):
        wave.append({
            "channel": c,
            "t": _downsample(np.arange(len(audio)) / cfg.sample_rate, 4000),
            "v": _downsample(audio[:, c], 4000),
        })
    bright = {
        "t": [float(x) for x in frame_times],
        "v": [float(x) for x in luminance],
    }
    def det_points(dets):
        return [{
            "time": d.time, "lo": d.time_lo, "hi": d.time_hi,
            "score": d.score, "sync": d.is_sync,
            "channel": d.channel, "rejected": bool(d.rejected_reason),
            "reason": d.rejected_reason,
        } for d in dets]
    return {
        "waveforms": wave,
        "brightness": bright,
        "audio_detections": [det_points(ch) for ch in audio_dets],
        "video_detections": det_points(video_dets),
        "sample_rate": cfg.sample_rate,
    }


def run_simulation(job: dict, store: JobStore) -> None:
    try:
        p = job["params"]
        cfg = MediaConfig(
            n_events=int(p.get("n_events", 24)),
            fps=float(p.get("source_fps", 30.0)),
            channels=int(p.get("channels", 2)),
        )
        gen = generate(cfg, os.path.join(store.workdir, job["id"], "media"))
        events = gen["events"]
        chain = ChainConfig(
            sample_rate=cfg.sample_rate, channels=cfg.channels,
            fps_source=cfg.fps,
            fps_recorded=float(p.get("recorded_fps", 30.0)),
            audio_delay=[float(x) for x in p.get("audio_delay", [0.12, 0.12])],
            video_delay=float(p.get("video_delay", 0.18)),
            audio_noise_std=float(p.get("audio_noise", 0.004)),
            dropped_events=[int(x) for x in p.get("dropped_events", [])],
            echo_delay=float(p.get("echo_delay", 0.0)),
            echo_gain=float(p.get("echo_gain", 0.5)),
            exposure_drift=float(p.get("exposure_drift", 0.0)),
            brightness_noise_std=float(p.get("brightness_noise", 0.004)),
            clock_drift_ppm=float(p.get("clock_drift_ppm", 0.0)),
            delay_steps=_pairs(p.get("delay_steps")),
            misaligned_events=[(int(a), b) for a, b in
                               _pairs(p.get("misaligned_events"))],
        )
        if len(chain.audio_delay) != cfg.channels:
            chain.audio_delay = [chain.audio_delay[0]] * cfg.channels
        out = run_chain(gen["frames"], gen["audio"], events, chain)
        result, adets, vdets = run_on_recording(
            out.audio, out.luminance, out.frame_times, events, cfg,
            out.recorded_fps, origin_known=True)
        job["result"] = result.to_dict()
        job["result_obj"] = result
        job["ground_truth"] = out.ground_truth
        job["series"] = _series_payload(
            job, out.audio, out.luminance, out.frame_times, adets, vdets, cfg)
        job["media"] = {"mp4": f"/jobs/{job['id']}/media/calibration.mp4",
                        "wav": f"/jobs/{job['id']}/media/calibration.wav"}
        job["status"] = "done"
    except Exception as exc:  # kernel failure feeds back to the panel
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"


def run_recording_analysis(job: dict, store: JobStore, upload_path: str) -> None:
    try:
        p = job["params"]
        cfg = MediaConfig(channels=int(p.get("channels", 2)))
        media_dir = os.path.join(store.workdir, job["id"], "media")
        gen = generate(cfg, media_dir)  # reference schedule + authored clip
        rec = reader_mod.read_recording(upload_path, cfg)
        if rec["audio"].shape[0] == 0 or len(rec["luminance"]) == 0:
            raise ValueError("recording yielded no decodable audio or video")
        result, adets, vdets = run_on_recording(
            rec["audio"], rec["luminance"], rec["frame_times"],
            gen["events"], cfg, rec["recorded_fps"], origin_known=False)
        job["result"] = result.to_dict()
        job["result_obj"] = result
        job["series"] = _series_payload(
            job, rec["audio"], rec["luminance"], rec["frame_times"],
            adets, vdets, cfg)
        job["media"] = {"mp4": f"/jobs/{job['id']}/media/calibration.mp4",
                        "wav": f"/jobs/{job['id']}/media/calibration.wav"}
        job["status"] = "done"
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"


def _corrected_offsets(result: dict, applied: dict[str, float]):
    """Actual arithmetic behind the panel's post-adjustment display.

    Adding a delay to a path shifts that path later, so the audio-minus-video
    offset moves by (audio_applied - video_applied); per-channel presentation
    offsets move by their own applied delay. Values outside the measured 95%
    interval are flagged as anomalies for the delay-calibration UI."""
    v_ap = float(applied.get("video_delay_ms", 0.0))
    av, channels, anomalies = [], [], []
    for o in result.get("av_offsets", []):
        ch = o["channel"]
        a_ap = float(applied.get(f"audio_ch{ch}_delay_ms", 0.0))
        corr = o["audio_minus_video_ms"] + a_ap - v_ap
        half = (o["ci95_hi_ms"] - o["ci95_lo_ms"]) / 2.0
        av.append({"channel": ch, "corrected_ms": corr,
                   "ci95_half_ms": half})
        if abs(corr) > half:
            anomalies.append(
                f"通道{ch} 修正后音画偏移 {corr:+.2f} ms 超出测量95%区间"
                f"半宽 {half:.2f} ms——请复核延迟设置")
    for a in result.get("audio", []):
        ch = a["channel"]
        a_ap = float(applied.get(f"audio_ch{ch}_delay_ms", 0.0))
        channels.append({"kind": "audio", "channel": ch,
                         "corrected_ms": a["offset"] * 1000.0 + a_ap,
                         "applied_ms": a_ap})
    v = result.get("video")
    if v:
        channels.append({"kind": "video", "channel": -1,
                         "corrected_ms": v["offset"] * 1000.0 + v_ap,
                         "applied_ms": v_ap})
    return {"av": av, "channels": channels}, anomalies


def _holdout_summary(result: dict) -> dict:
    out = {}
    for key, da in (result.get("drift") or {}).items():
        out[key] = {
            "rms_before_ms": (None if da["rms_before"] is None
                              else da["rms_before"] * 1000.0),
            "rms_after_ms": (None if da["rms_after"] is None
                             else da["rms_after"] * 1000.0),
            "n_points": len(da["validation_points"]),
        }
    return out


class PanelHandler(BaseHTTPRequestHandler):
    store: JobStore = None  # injected at server construction

    def log_message(self, *args):  # quiet
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC_DIR, "index.html"), "rb") as fh:
                self._send(200, fh.read(), "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        elif path == "/api/state":
            job = self.store.latest()
            self._json({"job": self._public(job) if job else None,
                        "applied_delay": self.store.applied_delay,
                        "validation_log": self.store.validation_log})
        elif path == "/api/export":
            self._handle_export()
        elif path.startswith("/api/jobs/"):
            job_id = path.split("/")[3]
            job = self.store.get(job_id)
            self._json(self._public(job) if job else {})
        elif path.startswith("/jobs/"):
            self._serve_media(path)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0))
        if path == "/api/simulate":
            params = json.loads(self.rfile.read(length) or b"{}")
            job_id = self.store.create("simulate", params)
            job = self.store.get(job_id)
            threading.Thread(target=run_simulation,
                             args=(job, self.store), daemon=True).start()
            self._json({"job_id": job_id})
        elif path == "/api/analyze":
            self._handle_upload(length)
        elif path == "/api/apply_delay":
            self._handle_apply(length)
        else:
            self._json({"error": "not found"}, 404)

    def _handle_upload(self, length: int):
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._json({"error": "expected multipart upload"}, 400)
            return
        boundary = ctype.split("boundary=")[-1].encode()
        body = self.rfile.read(length)
        upload_path, params = self._parse_multipart(body, boundary)
        if not upload_path:
            self._json({"error": "no file part 'recording'"}, 400)
            return
        job_id = self.store.create("analyze", params)
        job = self.store.get(job_id)
        threading.Thread(target=run_recording_analysis,
                         args=(job, self.store, upload_path),
                         daemon=True).start()
        self._json({"job_id": job_id})

    def _parse_multipart(self, body: bytes, boundary: bytes):
        params: dict = {}
        upload_path = None
        for part in body.split(b"--" + boundary):
            if not part or part in (b"--", b"--\r\n"):
                continue
            if b"\r\n\r\n" not in part:
                continue
            header, content = part.split(b"\r\n\r\n", 1)
            content = content.rstrip(b"\r\n")
            header_text = header.decode(errors="replace")
            if 'name="recording"' in header_text:
                job_tmp = os.path.join(self.store.workdir, "uploads")
                os.makedirs(job_tmp, exist_ok=True)
                upload_path = os.path.join(job_tmp,
                                           f"{uuid.uuid4().hex}.bin")
                with open(upload_path, "wb") as fh:
                    fh.write(content)
            elif 'name="params"' in header_text:
                try:
                    params = json.loads(content)
                except Exception:
                    params = {}
        return upload_path, params

    def _handle_apply(self, length: int):
        payload = json.loads(self.rfile.read(length) or b"{}")
        job = (self.store.get(payload["job_id"]) if payload.get("job_id")
               else self.store.latest())
        # Kernel gate: a failed/unanchored measurement cannot drive controls.
        if not job or job["status"] != "done" or not job.get("result"):
            self._json({"ok": False,
                        "error": "no successful calibration to apply"}, 409)
            return
        result = job["result"]
        if result["verdict"] == "fail":
            self._json({"ok": False,
                        "error": "calibration failed; fix alignment first"}, 409)
            return
        with self.store.lock:
            for key, val in payload.get("delay_ms", {}).items():
                self.store.applied_delay[key] = float(val)
            applied = dict(self.store.applied_delay)
        # Playback markers and measurement offsets update by real arithmetic
        # on the kernel result; anomalies feed back to this UI.
        corrected, anomalies = _corrected_offsets(result, applied)
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "job_id": job["id"],
            "applied_delay": applied,
            "corrected": corrected,
            "anomalies": anomalies,
            "holdout_validation": _holdout_summary(result),
        }
        with self.store.lock:
            self.store.validation_log.append(entry)
        self._json({"ok": True, "applied_delay": applied,
                    "corrected": corrected, "anomalies": anomalies,
                    "validation_log": self.store.validation_log})

    def _handle_export(self):
        job = self.store.latest()
        if not job or job["status"] != "done" or not job.get("result_obj"):
            self._json({"error": "no successful calibration to export"}, 409)
            return
        doc = export_playback_params(job["result_obj"])
        os.makedirs(os.path.join(self.store.workdir, "exports"),
                    exist_ok=True)
        fp = os.path.join(self.store.workdir, "exports",
                          f"playback_params_{job['id']}.json")
        with open(fp, "w") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
        body = json.dumps(doc, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Disposition",
                         "attachment; filename=playback_params.json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _public(self, job: dict) -> dict:
        return {k: job.get(k) for k in
                ("id", "kind", "status", "error", "result", "series",
                 "media", "ground_truth", "delay_applied", "params")}

    def _serve_static(self, name: str):
        safe = os.path.basename(name)
        fp = os.path.join(STATIC_DIR, safe)
        if not os.path.exists(fp):
            self._json({"error": "not found"}, 404)
            return
        ctype = "application/javascript" if safe.endswith(".js") else "text/css"
        with open(fp, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def _serve_media(self, path: str):
        rel = path[len("/jobs/"):]
        fp = os.path.join(self.store.workdir, rel)
        if not os.path.isfile(fp):
            self._json({"error": "not found"}, 404)
            return
        ctype = ("video/mp4" if fp.endswith(".mp4") else
                 "audio/wav" if fp.endswith(".wav") else "application/octet-stream")
        with open(fp, "rb") as fh:
            self._send(200, fh.read(), ctype)


def create_server(host: str, port: int, workdir: str) -> ThreadingHTTPServer:
    os.makedirs(workdir, exist_ok=True)
    store = JobStore(workdir)

    class Handler(PanelHandler):
        pass

    Handler.store = store
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.store = store  # type: ignore[attr-defined]
    return httpd
