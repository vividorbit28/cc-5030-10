import unittest

import numpy as np

from avcal.media import MediaConfig, generate
from avcal.chain import ChainConfig, run_chain
from avcal.calibrate import run_on_recording
from avcal import detect, match
from avcal.models import Detection, ScheduledEvent


def run_case(cfg: MediaConfig, chain: ChainConfig, tmpdir):
    gen = generate(cfg, tmpdir)
    events = gen["events"]
    out = run_chain(gen["frames"], gen["audio"], events, chain)
    result, adets, vdets = run_on_recording(
        out.audio, out.luminance, out.frame_times, events, cfg,
        out.recorded_fps, origin_known=True)
    return gen, out, result, adets, vdets


class FixedDelayTests(unittest.TestCase):
    def setUp(self):
        self.cfg = MediaConfig()
        self.tmp = "/workspace/avlatency/workdir/t_fixed"

    def test_known_constant_delay_recovered(self):
        chain = ChainConfig(
            fps_source=30, fps_recorded=30,
            audio_delay=[0.100, 0.150], video_delay=0.200,
            audio_noise_std=0.005, exposure_drift=0.0)
        _, out, result, _, _ = run_case(self.cfg, chain, self.tmp)
        for ch, gt in enumerate(out.ground_truth["audio_delay"]):
            est = result.audio[ch].offset
            self.assertAlmostEqual(est, gt, delta=0.002,
                                   msg=f"audio ch{ch} offset error too large")
        # video is frame-quantised: GT must fall inside the reported bracket
        v = result.video
        self.assertLessEqual(v.offset_lo, out.ground_truth["video_delay"] + 0.001)
        self.assertGreaterEqual(v.offset_hi,
                                out.ground_truth["video_delay"] - 0.001)
        self.assertEqual(result.verdict, "pass")

    def test_audio_uncertainty_below_one_ms_clean(self):
        chain = ChainConfig(audio_delay=[0.1, 0.1], video_delay=0.1,
                            audio_noise_std=0.002, fps_recorded=30)
        _, _, result, _, _ = run_case(self.cfg, chain, self.tmp)
        for a in result.audio:
            self.assertLess(a.jitter_std, 0.001)
            self.assertLess(a.offset_hi - a.offset_lo, 0.002)


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.cfg = MediaConfig()
        self.tmp = "/workspace/avlatency/workdir/t_noise"

    def test_high_noise_still_estimates_within_tolerance(self):
        chain = ChainConfig(audio_delay=[0.12, 0.12], video_delay=0.17,
                            audio_noise_std=0.03, brightness_noise_std=0.03,
                            fps_recorded=30)
        _, out, result, _, _ = run_case(self.cfg, chain, self.tmp)
        for ch, gt in enumerate(out.ground_truth["audio_delay"]):
            self.assertAlmostEqual(result.audio[ch].offset, gt, delta=0.005)
        self.assertGreaterEqual(result.video.n_matched, 18)

    def test_background_hum_does_not_create_events(self):
        chain = ChainConfig(audio_delay=[0.1, 0.1], video_delay=0.1,
                            background_hum=True, audio_noise_std=0.002,
                            echo_delay=0.0, fps_recorded=30)
        gen = generate(self.cfg, self.tmp)
        out = run_chain(gen["frames"], gen["audio"], gen["events"], chain)
        dets = detect.detect_audio_channel(out.audio, 0, self.cfg)
        # exactly one detection per scheduled event, none from the 50 Hz hum
        self.assertEqual(len(dets), self.cfg.n_events)


class DropEventTests(unittest.TestCase):
    def setUp(self):
        self.cfg = MediaConfig()
        self.tmp = "/workspace/avlatency/workdir/t_drop"

    def test_dropped_events_reported_not_index_paired(self):
        drops = [3, 9, 17]
        chain = ChainConfig(audio_delay=[0.11, 0.11], video_delay=0.16,
                            dropped_events=drops, fps_recorded=30)
        _, _, result, _, _ = run_case(self.cfg, chain, self.tmp)
        for a in result.audio:
            self.assertEqual(sorted(
                idx for idx in range(self.cfg.n_events)
                if idx not in {p.event_index for p in a.pairings}), drops)
            self.assertEqual(a.n_dropped, len(drops))
        # offset remains accurate despite the gaps
        self.assertAlmostEqual(result.audio[0].offset, 0.11, delta=0.003)

    def test_match_refuses_positional_pairing_under_shift(self):
        # Build a synthetic detection list with one missing event; pairing
        # must follow the numbering/sequence, never the array position.
        ev = ScheduledEvent(index=0, is_sync=True, nominal_time=1.0,
                            audio_offsets=[0.0], flash_frames=[0])
        events = [ev] + [
            ScheduledEvent(i, False, 1.0 + 0.5 * i, [0.0], [2 * i])
            for i in range(1, 6)]
        dets = [Detection("audio", 0, 1.2, 1.19, 1.21, 0.001, 1.0, True)]
        dets += [Detection("audio", 0, 1.2 + 0.5 * i, 0, 0, 0.001, 1.0)
                 for i in range(1, 6) if i != 3]  # event 3 missing
        pairs, dropped, unpaired = match.align(events, dets, 0.5)
        self.assertEqual(dropped, [3])
        self.assertEqual([p.event_index for p in pairs],
                         [0, 1, 2, 4, 5])  # no shift of later indices


class DuplicatePeakTests(unittest.TestCase):
    def setUp(self):
        self.cfg = MediaConfig()
        self.tmp = "/workspace/avlatency/workdir/t_dup"

    def test_echo_peaks_counted_as_duplicates(self):
        chain = ChainConfig(audio_delay=[0.12, 0.12], video_delay=0.16,
                            echo_delay=0.05, echo_gain=0.55,
                            audio_noise_std=0.004, fps_recorded=30)
        _, _, result, adets, _ = run_case(self.cfg, chain, self.tmp)
        # raw detections exceed scheduled events because echoes survive
        self.assertGreater(len(adets[0]), self.cfg.n_events)
        # but the matched count is unchanged and duplicates are logged
        self.assertEqual(result.audio[0].n_matched, self.cfg.n_events)
        self.assertGreater(result.audio[0].n_duplicates, 0)
        self.assertAlmostEqual(result.audio[0].offset, 0.12, delta=0.003)


class FrameRateTests(unittest.TestCase):
    def _run(self, rec_fps):
        cfg = MediaConfig()
        chain = ChainConfig(fps_source=30, fps_recorded=rec_fps,
                            audio_delay=[0.1, 0.1], video_delay=0.18,
                            audio_noise_std=0.004)
        gen = generate(cfg, f"/workspace/avlatency/workdir/t_fps_{rec_fps}")
        out = run_chain(gen["frames"], gen["audio"], gen["events"], chain)
        result, _, vdets = run_on_recording(
            out.audio, out.luminance, out.frame_times, gen["events"], cfg,
            rec_fps, origin_known=True)
        return out, result, vdets

    def test_multiple_frame_rates_bracket_truth(self):
        for rec_fps in (15, 24, 30, 60):
            out, result, _ = self._run(rec_fps)
            gt = out.ground_truth["video_delay"]
            v = result.video
            # every video detection uncertainty must reflect the frame period
            expected_half = 0.5 / rec_fps
            for p in v.pairings:
                self.assertGreaterEqual(p.det_hi - p.det_lo,
                                        expected_half * 0.9)
            self.assertLessEqual(v.offset_lo, gt + 0.002)
            self.assertGreaterEqual(v.offset_hi, gt - 0.002)
            # never claim sub-frame precision
            self.assertGreaterEqual(v.offset_hi - v.offset_lo,
                                  (1.0 / rec_fps) * 0.5)

    def test_low_fps_emits_precision_warning(self):
        _, result, _ = self._run(15)
        self.assertTrue(any("frame bracket" in r for r in result.reasons))


class ExposureRejectionTests(unittest.TestCase):
    def test_slow_drift_is_rejected_not_paired(self):
        cfg = MediaConfig()
        # A pure linear exposure ramp is removed by robust de-trending, so it
        # must produce zero flash events (the correct rejection outcome).
        n = 300
        lum = np.linspace(0.3, 0.7, n)
        times = (np.arange(n) + 0.5) / 30.0
        dets = detect.detect_video(lum, times, 30.0)
        real = [d for d in dets if not d.rejected_reason]
        self.assertEqual(real, [])

    def test_ramp_plus_real_flash_only_yields_flash(self):
        cfg = MediaConfig()
        n = 300
        lum = np.linspace(0.3, 0.6, n)
        times = (np.arange(n) + 0.5) / 30.0
        lum[120:122] = 1.0  # one genuine two-frame flash on the drifting base
        dets = detect.detect_video(lum, times, 30.0)
        real = [d for d in dets if not d.rejected_reason]
        self.assertEqual(len(real), 1)
        self.assertTrue(abs(real[0].time - times[120]) < 0.1)

    def test_saturated_drift_flagged(self):
        cfg = MediaConfig()
        gen = generate(cfg, "/workspace/avlatency/workdir/t_sat")
        chain = ChainConfig(audio_delay=[0.1, 0.1], video_delay=0.15,
                            exposure_drift=0.12, fps_recorded=30)
        out = run_chain(gen["frames"], gen["audio"], gen["events"], chain)
        _, vdets = detect.detect_all(
            out.audio, out.luminance, out.frame_times, 30.0, cfg)
        self.assertTrue(any("clipped" in d.rejected_reason
                            for d in vdets if d.rejected_reason))


class UncertaintyTests(unittest.TestCase):
    def test_video_never_reports_subframe_point(self):
        cfg = MediaConfig()
        chain = ChainConfig(audio_delay=[0.1, 0.1], video_delay=0.15,
                            fps_recorded=24)
        gen = generate(cfg, "/workspace/avlatency/workdir/t_unc")
        out = run_chain(gen["frames"], gen["audio"], gen["events"], chain)
        _, vdets = detect.detect_all(
            out.audio, out.luminance, out.frame_times, 24.0, cfg)
        for d in vdets:
            if d.rejected_reason:
                continue
            self.assertGreater(d.time_hi - d.time_lo, 0.0)
            self.assertTrue(d.time_lo <= d.time <= d.time_hi)


if __name__ == "__main__":
    unittest.main(verbosity=2)
