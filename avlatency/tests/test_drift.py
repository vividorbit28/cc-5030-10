"""Drift/jump analysis: slow clock drift, sudden delay steps, single
mis-matched events and long event absence. Identification boundaries and
estimation intervals are checked against ground truth from the recording
chain (flash pulses + clock drift), plus synthetic confidence-coverage runs.
"""
import unittest

import numpy as np

from avcal import drift
from avcal.media import MediaConfig, generate
from avcal.chain import ChainConfig, run_chain
from avcal.calibrate import run_on_recording

TMP = "/workspace/avlatency/workdir"


def run_case(cfg, chain, tmp):
    gen = generate(cfg, tmp)
    out = run_chain(gen["frames"], gen["audio"], gen["events"], chain)
    result, _, _ = run_on_recording(
        out.audio, out.luminance, out.frame_times, gen["events"], cfg,
        out.recorded_fps, origin_known=True)
    return gen, out, result


class SlowDriftTests(unittest.TestCase):
    """Slow linear clock-rate error must come out as one segment with a
    significant slope - not as jumps, not as outliers."""

    def test_linear_drift_single_segment_ci_covers_truth(self):
        cfg = MediaConfig(n_events=24)
        chain = ChainConfig(fps_source=30, fps_recorded=30,
                            audio_delay=[0.12, 0.12], video_delay=0.18,
                            audio_noise_std=0.004, clock_drift_ppm=2000.0)
        _, out, result = run_case(cfg, chain, f"{TMP}/t_drift")
        da = result.drift["audio_0"]
        ok = [s for s in da.segments if s.status == "ok"]
        self.assertEqual(len(ok), 1, "pure drift must not split into segments")
        self.assertEqual(len(da.jumps), 0, "pure drift must not raise a jump")
        self.assertEqual(len(da.excluded), 0)
        seg = ok[0]
        truth = 2000e-6
        self.assertLessEqual(seg.slope - seg.slope_ci95, truth)
        self.assertGreaterEqual(seg.slope + seg.slope_ci95, truth)
        # estimation error itself is small for sample-level audio
        self.assertLess(abs(seg.slope - truth), 100e-6)
        self.assertIn("时钟速差", seg.classification)
        # offset at t_ref: base delay is itself stretched by the clock error,
        # residual(t) = delay*(1+eps) + eps*t, so truth at t_ref is:
        want = 0.12 * (1.0 + truth) + truth * seg.t_ref
        self.assertLessEqual(seg.offset - seg.offset_ci95, want)
        self.assertGreaterEqual(seg.offset + seg.offset_ci95, want)
        # independent holdout validation: small residuals after correction
        self.assertIsNotNone(da.rms_after)
        self.assertLess(da.rms_after, 0.002)
        self.assertGreater(da.rms_before, 10 * da.rms_after)

    def test_flash_pulses_drift_ci_also_covers_truth(self):
        cfg = MediaConfig(n_events=24)
        chain = ChainConfig(fps_source=30, fps_recorded=30,
                            audio_delay=[0.12, 0.12], video_delay=0.18,
                            audio_noise_std=0.004, clock_drift_ppm=2000.0)
        _, _, result = run_case(cfg, chain, f"{TMP}/t_drift_v")
        dv = result.drift["video"]
        ok = [s for s in dv.segments if s.status == "ok"]
        self.assertEqual(len(ok), 1)
        seg = ok[0]
        # frame-limited video is honestly wide, but must still bracket truth
        self.assertLessEqual(seg.slope - seg.slope_ci95, 2000e-6)
        self.assertGreaterEqual(seg.slope + seg.slope_ci95, 2000e-6)


class JumpTests(unittest.TestCase):
    """A sudden fixed extra delay (device restart) must be a jump between
    two flat segments, not a slope, not a swarm of outliers."""

    def test_step_detected_located_and_sized(self):
        cfg = MediaConfig(n_events=24)
        chain = ChainConfig(fps_source=30, fps_recorded=30,
                            audio_delay=[0.12, 0.12], video_delay=0.18,
                            audio_noise_std=0.004,
                            delay_steps=[(7.0, 0.05)])
        _, _, result = run_case(cfg, chain, f"{TMP}/t_step")
        da = result.drift["audio_0"]
        self.assertEqual(len(da.jumps), 1)
        j = da.jumps[0]
        self.assertAlmostEqual(j.time, 7.0, delta=0.75,
                               msg="jump located too far from the true step")
        self.assertLessEqual(j.step - j.step_ci95, 0.05)
        self.assertGreaterEqual(j.step + j.step_ci95, 0.05)
        ok = [s for s in da.segments if s.status == "ok"]
        self.assertEqual(len(ok), 2)
        for s in ok:
            # each side is flat: slope consistent with zero (not drift)
            self.assertLessEqual(abs(s.slope), s.slope_ci95)
            self.assertEqual(s.classification, "速率与标称时钟一致")
        # first segment recovers the base delay
        self.assertAlmostEqual(ok[0].offset, 0.12, delta=0.005)
        self.assertEqual(len(da.excluded), 0,
                         "a real jump must not be eaten by outlier rejection")

    def test_video_step_interval_contains_truth(self):
        cfg = MediaConfig(n_events=24)
        chain = ChainConfig(fps_source=30, fps_recorded=30,
                            audio_delay=[0.12, 0.12], video_delay=0.18,
                            audio_noise_std=0.004,
                            delay_steps=[(7.0, 0.05)])
        _, _, result = run_case(cfg, chain, f"{TMP}/t_step_v")
        dv = result.drift["video"]
        self.assertGreaterEqual(len(dv.jumps), 1)
        j = dv.jumps[0]
        self.assertLessEqual(j.step - j.step_ci95, 0.05)
        self.assertGreaterEqual(j.step + j.step_ci95, 0.05)


class MismatchTests(unittest.TestCase):
    """One mis-matched event is an audited exclusion, never a silent drop,
    and it must not bend the segment parameters."""

    def test_single_error_excluded_with_record(self):
        cfg = MediaConfig(n_events=24)
        chain = ChainConfig(fps_source=30, fps_recorded=30,
                            audio_delay=[0.12, 0.12], video_delay=0.18,
                            audio_noise_std=0.004,
                            misaligned_events=[(10, 0.03)])
        _, _, result = run_case(cfg, chain, f"{TMP}/t_mis")
        da = result.drift["audio_0"]
        self.assertEqual(len(da.excluded), 1)
        ex = da.excluded[0]
        self.assertEqual(ex.event_index, 10)
        self.assertTrue(ex.reason)
        self.assertIn("阈值", ex.reason)  # numeric basis is on record
        ok = [s for s in da.segments if s.status == "ok"]
        self.assertEqual(len(ok), 1)
        self.assertAlmostEqual(ok[0].offset, 0.12, delta=0.003)
        # neighbours survived: only the one bad point was removed
        self.assertGreaterEqual(ok[0].n_fit, 16)
        # the excluded point is not recycled into validation either
        self.assertNotIn(10, {v.event_index for v in da.validation_points})


class LongAbsenceTests(unittest.TestCase):
    """A long run of missing events is a coverage gap: no parameters are
    claimed inside it, and extrapolation beyond the data is flagged with
    its own (widening) interval."""

    def test_gap_and_extrapolation_reported(self):
        cfg = MediaConfig(n_events=24)
        drops = list(range(8, 17))  # ~4.5 s with no events at all
        chain = ChainConfig(fps_source=30, fps_recorded=30,
                            audio_delay=[0.12, 0.12], video_delay=0.18,
                            audio_noise_std=0.004, dropped_events=drops)
        gen, _, result = run_case(cfg, chain, f"{TMP}/t_gap")
        da = result.drift["audio_0"]
        nom = {e.index: e.nominal_time for e in gen["events"]}
        gap_lo, gap_hi = nom[8], nom[16]
        gaps = [g for s in da.segments for g in s.gaps]
        self.assertTrue(any(lo <= gap_lo + 0.6 and hi >= gap_hi - 0.6
                            for lo, hi in gaps),
                        f"no reported gap covers [{gap_lo},{gap_hi}]: {gaps}")
        mid = 0.5 * (gap_lo + gap_hi)
        p_gap = drift.predict(da, mid)
        self.assertEqual(p_gap["status"], "gap",
                         "inside the hole the model must not claim precision")
        p_in = drift.predict(da, nom[4])
        self.assertEqual(p_in["status"], "ok")
        p_out = drift.predict(da, nom[23] + 5.0)
        self.assertEqual(p_out["status"], "extrapolated")
        # extrapolation error is reported separately and grows with distance
        half_in = p_in["hi"] - p_in["lo"]
        half_out = p_out["hi"] - p_out["lo"]
        self.assertGreater(half_out, half_in)
        # both sides of the gap still carry valid parameters
        ok = [s for s in da.segments if s.status == "ok"]
        self.assertTrue(any(s.t_lo < gap_lo for s in ok))
        self.assertTrue(any(s.t_hi > gap_hi for s in ok))


class InsufficientDataTests(unittest.TestCase):
    def test_few_points_get_no_parameters(self):
        t = np.array([1.0, 1.5, 2.0, 2.5])
        r = np.array([0.10, 0.10, 0.10, 0.10])
        u = np.full(4, 1e-4)
        da = drift.analyze(t, r, u, [0, 1, 2, 3])
        self.assertTrue(all(s.status == "insufficient" for s in da.segments))
        self.assertTrue(all(s.offset is None for s in da.segments))
        self.assertTrue(any("数据不足" in n for n in da.notes))

    def test_empty_series_is_safe(self):
        da = drift.analyze(np.array([]), np.array([]), np.array([]), [])
        self.assertEqual(da.segments[0].status, "insufficient")
        p = drift.predict(da, 1.0)
        self.assertEqual(p["status"], "insufficient")
        self.assertIsNone(p["value"])


class ConfidenceCoverageTests(unittest.TestCase):
    """Synthetic coverage check: a 95% interval fails ~5% of the time by
    design, so single runs prove nothing. Over many independent noise
    realisations the empirical coverage must be close to nominal."""

    def test_empirical_coverage_near_nominal(self):
        rng = np.random.default_rng(2026)
        n_runs, n = 150, 30
        true_off, true_slope = 0.10, 0.002
        slope_hits = off_hits = 0
        for _ in range(n_runs):
            t = 1.0 + 0.5 * np.arange(n)
            r = true_off + true_slope * t + rng.normal(0, 0.0003, n)
            u = np.full(n, 1e-4)
            da = drift.analyze(t, r, u, list(range(n)))
            ok = [s for s in da.segments if s.status == "ok"]
            self.assertEqual(len(ok), 1)
            seg = ok[0]
            if seg.slope - seg.slope_ci95 <= true_slope <= \
                    seg.slope + seg.slope_ci95:
                slope_hits += 1
            want = true_off + true_slope * seg.t_ref
            if seg.offset - seg.offset_ci95 <= want <= \
                    seg.offset + seg.offset_ci95:
                off_hits += 1
        for name, hits in (("slope", slope_hits), ("offset", off_hits)):
            self.assertGreaterEqual(hits / n_runs, 0.85,
                                    f"{name} CI coverage too low")
            self.assertLessEqual(hits / n_runs, 1.0)

    def test_single_run_intervals_are_tight(self):
        # spot-check one run: audio-grade noise gives sub-ms offset CI
        rng = np.random.default_rng(7)
        n = 40
        t = 1.0 + 0.5 * np.arange(n)
        r = 0.10 + 0.002 * t + rng.normal(0, 0.0003, n)
        da = drift.analyze(t, r, np.full(n, 1e-4), list(range(n)))
        seg = [s for s in da.segments if s.status == "ok"][0]
        self.assertLess(seg.offset_ci95, 0.001)
        self.assertLess(seg.slope_ci95, 200e-6)
        self.assertEqual(len(da.jumps), 0)

    def test_validation_points_are_independent_of_fit(self):
        rng = np.random.default_rng(7)
        n = 32
        t = 1.0 + 0.5 * np.arange(n)
        r = 0.1 + rng.normal(0, 0.0003, n)
        u = np.full(n, 1e-4)
        da = drift.analyze(t, r, u, list(range(n)))
        n_fit = sum(s.n_fit for s in da.segments)
        n_hold = sum(s.n_holdout for s in da.segments)
        # every sample is accounted for: fitted, held out, or excluded
        self.assertEqual(n_fit + n_hold + len(da.excluded), n)
        ok_idx = {v.event_index for v in da.validation_points
                  if v.status == "ok"}
        self.assertTrue(ok_idx, "expected covered holdout points")
        # rms_after is computed on holdout points only, and correction must
        # not make them worse than the raw residuals
        self.assertIsNotNone(da.rms_after)
        self.assertLessEqual(da.rms_after, da.rms_before + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
