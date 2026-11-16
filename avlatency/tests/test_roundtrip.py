"""Real media encode -> ffmpeg decode round trip (no simulated arrays)."""
import unittest

import numpy as np

from avcal.media import MediaConfig, generate
from avcal import ffmpeg as ff
from avcal.reader import read_recording
from avcal.calibrate import run_on_recording


class RoundTripTests(unittest.TestCase):
    def test_mp4_roundtrip_detects_events(self):
        cfg = MediaConfig(n_events=12)
        gen = generate(cfg, "/workspace/avlatency/workdir/t_roundtrip")
        mp4 = gen["paths"]["mp4"]

        # Probe and decode the actual encoded file via ffmpeg.
        fps = ff.probe_fps(mp4)
        self.assertIsNotNone(fps)
        rec = read_recording(mp4, cfg)
        self.assertGreater(rec["audio"].shape[0], 0)
        self.assertGreater(len(rec["luminance"]), 0)

        # The decoded file has no added physical delay (offset ~ 0); the point
        # is that the real codec path still yields the authored event count.
        result, adets, vdets = run_on_recording(
            rec["audio"], rec["luminance"], rec["frame_times"],
            gen["events"], cfg, rec["recorded_fps"], origin_known=True)
        real_video = [d for d in vdets if not d.rejected_reason]
        self.assertGreaterEqual(len(real_video), cfg.n_events - 1)
        self.assertGreaterEqual(result.audio[0].n_matched,
                                cfg.n_events - 1)
        # near-zero physical delay through the codec path
        self.assertAlmostEqual(result.audio[0].offset, 0.0, delta=0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
