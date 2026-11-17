"""Integration tests for the HTTP control panel and its kernel gate."""
import json
import tempfile
import threading
import time
import unittest
import urllib.request

from avcal.web import create_server


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="avcal_web_")
        cls.httpd = create_server("127.0.0.1", 0, cls.tmp)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever,
                                      daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def _wait(self, job_id):
        for _ in range(40):
            job = _get(f"{self.base}/api/jobs/{job_id}")
            if job["status"] != "queued":
                return job
            time.sleep(0.5)
        self.fail("job did not finish")

    def test_full_simulate_and_apply(self):
        code, r = _post(f"{self.base}/api/simulate", {
            "n_events": 20, "recorded_fps": 30, "channels": 2,
            "audio_delay": [0.12, 0.12], "video_delay": 0.18})
        self.assertEqual(code, 200)
        job = self._wait(r["job_id"])
        self.assertEqual(job["status"], "done")
        self.assertIn("verdict", job["result"])
        self.assertIn("waveforms", job["series"])
        # a passing/warn calibration can drive the delay controls
        code, applied = _post(f"{self.base}/api/apply_delay",
                              {"job_id": job["id"],
                               "delay_ms": {"audio_ch0_delay_ms": 12.5}})
        self.assertEqual(code, 200)
        self.assertTrue(applied["ok"])

    def test_failed_calibration_blocks_adjustment(self):
        code, r = _post(f"{self.base}/api/simulate",
                        {"n_events": 2, "recorded_fps": 24})
        job = self._wait(r["job_id"])
        self.assertEqual(job["result"]["verdict"], "fail")
        code, resp = _post(f"{self.base}/api/apply_delay",
                           {"job_id": job["id"],
                            "delay_ms": {"audio_ch0_delay_ms": 1.0}})
        self.assertEqual(code, 409)
        self.assertFalse(resp["ok"])

    def test_state_endpoint_tracks_latest(self):
        state = _get(f"{self.base}/api/state")
        self.assertIn("applied_delay", state)
        self.assertIn("validation_log", state)
        self.assertIsNotNone(state["job"])

    def test_apply_returns_computed_corrections_and_log(self):
        code, r = _post(f"{self.base}/api/simulate", {
            "n_events": 20, "recorded_fps": 30, "channels": 2,
            "audio_delay": [0.12, 0.12], "video_delay": 0.18})
        job = self._wait(r["job_id"])
        self.assertEqual(job["status"], "done")
        # apply exactly the measured audio-video offset to the video path
        d = job["result"]["av_offsets"][0]["audio_minus_video_ms"]
        code, resp = _post(f"{self.base}/api/apply_delay",
                           {"job_id": job["id"],
                            "delay_ms": {"video_delay_ms": d}})
        self.assertEqual(code, 200)
        self.assertTrue(resp["ok"])
        # corrected offsets are real arithmetic: d - applied ~= 0
        corr = resp["corrected"]["av"][0]
        self.assertAlmostEqual(corr["corrected_ms"], 0.0, delta=0.5)
        self.assertIn("anomalies", resp)
        # the apply is logged for the before/after comparison view
        self.assertTrue(resp["validation_log"])
        entry = resp["validation_log"][-1]
        self.assertIn("holdout_validation", entry)
        state = _get(f"{self.base}/api/state")
        self.assertTrue(state["validation_log"])

    def test_export_playback_params(self):
        code, r = _post(f"{self.base}/api/simulate", {
            "n_events": 24, "recorded_fps": 30, "channels": 2,
            "audio_delay": [0.12, 0.12], "video_delay": 0.18,
            "clock_drift_ppm": 1500.0})
        job = self._wait(r["job_id"])
        self.assertEqual(job["status"], "done")
        with urllib.request.urlopen(f"{self.base}/api/export",
                                    timeout=10) as res:
            doc = json.loads(res.read())
        self.assertEqual(doc["format"], "avcal-playback-params/1")
        self.assertIn("baseline_constant_delay_ms", doc)
        self.assertIn("audio_0", doc["channels"])
        ch = doc["channels"]["audio_0"]
        self.assertIn("holdout_validation", ch)
        ok = [s for s in ch["segments"] if s["status"] == "ok"]
        self.assertTrue(ok)
        # exported slope brackets the simulated clock drift
        s = ok[0]
        self.assertLessEqual(s["slope_ppm"] - s["slope_ci95_ppm"], 1500.0)
        self.assertGreaterEqual(s["slope_ppm"] + s["slope_ci95_ppm"], 1500.0)
        self.assertIn("不修改任何真实设备设置", doc["note"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
