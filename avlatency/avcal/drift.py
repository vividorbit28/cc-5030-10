"""Piecewise drift / jump analysis of the per-event residual series.

The baseline calibration reports one constant offset. Long continuous
observation also shows slow clock-rate drift, sudden jumps (device restart /
buffer rebuild) and occasional detection errors. This module fits the
residual series (det_time - nominal_time versus nominal_time) as an
interpretable piecewise-linear model:

* jumps: BIC-gated binary segmentation. A split must beat a single straight
  line by a strong evidence margin AND the fitted step must clear both a
  physical minimum and its own uncertainty, so slow linear drift is not
  misread as a restart and vice versa;
* per segment: least-squares line in centred form -> offset + slope, each
  with a 95% interval; segments with too few points are reported as
  'insufficient' and get NO parameters;
* outliers: iterative MAD rejection. Every excluded point is kept in an
  audit record with the numeric reason - points are never silently dropped
  and the system is never declared stable by deleting large residuals;
* validation: a deterministic holdout (every HOLDOUT_EVERY-th point) is
  never used for fitting. Residuals before/after correction are reported
  for those independent points only;
* coverage: observation gaps longer than GAP_FACTOR x median spacing are
  listed explicitly. Predictions inside gaps or outside the observed range
  are marked and carry the regression's own (widening) error band, reported
  separately from in-range estimates.
"""
from __future__ import annotations

import math

import numpy as np

from .models import (
    DriftAnalysis, ExcludedPoint, JumpInfo, SegmentInfo, ValidationPoint,
)

MIN_SEG_POINTS = 5      # fewer fit points => 'insufficient', no parameters
HOLDOUT_EVERY = 4       # every 4th sorted point is held out for validation
BIC_MARGIN = 6.0        # strong-evidence threshold for accepting a split
MIN_STEP_S = 0.002      # a jump smaller than this is not worth reporting
STEP_SIG = 3.0          # step must also exceed 3x its own standard error
OUTLIER_K = 3.5         # MAD-sigma multiplier for outlier rejection
MAX_OUTLIER_ITERS = 3
GAP_FACTOR = 4.0        # gap > 4x median spacing => coverage gap
MAX_DEPTH = 4
SIGMA_FLOOR = 1e-4      # 0.1 ms: never claim better than this


def _line_fit(t: np.ndarray, r: np.ndarray) -> dict:
    """Least-squares line r = a + b*(t - t_ref), centred so a and b are
    independent. Sigma^2 is floored by the detection uncertainty scale so
    intervals never shrink below what the observations can support."""
    n = len(t)
    t_ref = float(t.mean())
    tc = t - t_ref
    sxx = float(tc @ tc)
    b = float(tc @ r / sxx) if sxx > 0 else 0.0
    a = float(r.mean())
    res = r - (a + b * tc)
    rss = float(res @ res)
    dof = max(n - 2, 1)
    return {"a": a, "b": b, "t_ref": t_ref, "rss": rss, "n": n,
            "sxx": sxx, "dof": dof}


def _sigma2(fit: dict, unc_floor: float) -> float:
    return max(fit["rss"] / fit["dof"], (0.5 * unc_floor) ** 2, SIGMA_FLOOR ** 2)


def _predict(fit: dict, t: float) -> float:
    return fit["a"] + fit["b"] * (t - fit["t_ref"])


def _predict_var(fit: dict, t: float, s2: float) -> float:
    var_b = s2 / fit["sxx"] if fit["sxx"] > 0 else float("inf")
    return s2 / fit["n"] + (t - fit["t_ref"]) ** 2 * var_b


def _bic(n: int, rss: float, p: int) -> float:
    return n * math.log(max(rss, 1e-24) / n) + p * math.log(max(n, 3))


def _find_splits(t: np.ndarray, r: np.ndarray, unc_floor: float,
                 depth: int, out: list[dict], base: int = 0) -> None:
    n = len(t)
    if n < 2 * MIN_SEG_POINTS or depth >= MAX_DEPTH:
        return
    bic1 = _bic(n, _line_fit(t, r)["rss"], 2)
    best_gain, best_k = BIC_MARGIN, None
    for k in range(MIN_SEG_POINTS, n - MIN_SEG_POINTS + 1):
        rss2 = _line_fit(t[:k], r[:k])["rss"] + _line_fit(t[k:], r[k:])["rss"]
        gain = bic1 - _bic(n, rss2, 4)
        if gain > best_gain:
            best_gain, best_k = gain, k
    if best_k is None:
        return
    k = best_k
    f1, f2 = _line_fit(t[:k], r[:k]), _line_fit(t[k:], r[k:])
    tb = 0.5 * (t[k - 1] + t[k])
    s2_1, s2_2 = _sigma2(f1, unc_floor), _sigma2(f2, unc_floor)
    step = _predict(f2, tb) - _predict(f1, tb)
    se = math.sqrt(_predict_var(f1, tb, s2_1) + _predict_var(f2, tb, s2_2))
    if abs(step) <= max(MIN_STEP_S, STEP_SIG * se):
        return  # step not physically/statistically significant: keep one line
    out.append({"index": base + k, "time": tb, "step": step,
                "ci95": 1.96 * se})
    _find_splits(t[:k], r[:k], unc_floor, depth + 1, out, base)
    _find_splits(t[k:], r[k:], unc_floor, depth + 1, out, base + k)


def _mad_sigma(res: np.ndarray, unc_floor: float) -> float:
    mad = float(np.median(np.abs(res - np.median(res))))
    return max(1.4826 * mad, 0.5 * unc_floor, SIGMA_FLOOR)


def _robust_line(t: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """Theil-Sen line: median pairwise slope + median intercept. An edge
    outlier has full leverage on least squares (it bends the line and then
    hides behind the inflated MAD); the median slope does not move."""
    n = len(t)
    if n < 2:
        return float(r.mean()) if n else 0.0, 0.0
    slopes = []
    for i in range(n - 1):
        dt = t[i + 1 :] - t[i]
        ok = np.abs(dt) > 1e-12
        if ok.any():
            slopes.extend(((r[i + 1 :][ok] - r[i]) / dt[ok]).tolist())
    b = float(np.median(slopes)) if slopes else 0.0
    a = float(np.median(r - b * t))
    return a, b


def _reject_outliers(t: np.ndarray, r: np.ndarray, unc: np.ndarray,
                     idx: list[int], unc_floor: float,
                     excluded: list[ExcludedPoint], phase: str):
    """Iterative MAD rejection around a robust (Theil-Sen) line. Every
    rejected point is recorded with the numbers that condemned it; nothing
    disappears quietly."""
    keep_t, keep_r, keep_u, keep_i = t, r, unc, idx
    for _ in range(MAX_OUTLIER_ITERS):
        if len(keep_t) < 3:
            break
        a, b = _robust_line(keep_t, keep_r)
        res = keep_r - (a + b * keep_t)
        sigma = _mad_sigma(res, unc_floor)
        thresh = OUTLIER_K * sigma
        bad = np.where(np.abs(res) > thresh)[0]
        if bad.size == 0:
            break
        mask = np.ones(len(keep_t), bool)
        for j in bad:
            excluded.append(ExcludedPoint(
                event_index=int(keep_i[j]), time=float(keep_t[j]),
                residual=float(keep_r[j]),
                reason=(f"{phase}: |残差 {res[j]*1000:.2f} ms| > 阈值 "
                        f"{thresh*1000:.2f} ms = {OUTLIER_K} x 稳健σ "
                        f"({sigma*1000:.2f} ms, MAD法)")))
            mask[j] = False
        keep_t, keep_r, keep_u = keep_t[mask], keep_r[mask], keep_u[mask]
        keep_i = [i for i, m in zip(keep_i, mask) if m]
    return keep_t, keep_r, keep_u, keep_i


def _gaps(times: np.ndarray) -> list[list[float]]:
    if len(times) < 2:
        return []
    order = np.argsort(times)
    ts = times[order]
    med = float(np.median(np.diff(ts)))
    if med <= 0:
        return []
    return [[float(ts[i]), float(ts[i + 1])]
            for i in range(len(ts) - 1)
            if ts[i + 1] - ts[i] > GAP_FACTOR * med]


def analyze(times: np.ndarray, residuals: np.ndarray,
            uncertainties: np.ndarray, event_indices: list[int]) -> DriftAnalysis:
    """Fit the piecewise drift/jump model for one channel.

    times: nominal event times (s); residuals: det_time - nominal_time (s);
    uncertainties: per-detection 1-sigma values (s)."""
    order = np.argsort(times)
    t_all = np.asarray(times, dtype=np.float64)[order]
    r_all = np.asarray(residuals, dtype=np.float64)[order]
    u_all = np.asarray(uncertainties, dtype=np.float64)[order]
    i_all = [int(event_indices[k]) for k in order]
    notes: list[str] = []
    excluded: list[ExcludedPoint] = []
    unc_floor = float(np.median(u_all)) if len(u_all) else 0.0

    # Deterministic holdout: never fitted, only validated.
    fit_mask = np.ones(len(t_all), bool)
    fit_mask[np.arange(len(t_all)) % HOLDOUT_EVERY == HOLDOUT_EVERY - 1] = False
    t, r, u = t_all[fit_mask], r_all[fit_mask], u_all[fit_mask]
    idx = [i for i, m in zip(i_all, fit_mask) if m]
    hold = [(t_all[k], r_all[k], i_all[k])
            for k in range(len(t_all)) if not fit_mask[k]]

    # Gross-outlier pre-pass on a single global line. With a genuine jump the
    # global MAD is huge and this rejects nothing; with a clean drift plus one
    # bad detection it removes the outlier before segmentation so a single
    # error cannot masquerade as a break.
    t, r, u, idx = _reject_outliers(t, r, u, idx, unc_floor, excluded,
                                    "全程粗筛")

    # Segmentation and outlier rejection iterate: an outlier near a boundary
    # can pull the split to itself, so after any exclusion the splits are
    # re-found on the cleaned series until stable.
    splits: list[dict] = []
    seg_points: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]] = []
    for _ in range(3):
        splits = []
        _find_splits(t, r, unc_floor, 0, splits)
        splits.sort(key=lambda s: s["index"])
        bounds = [0] + [s["index"] for s in splits] + [len(t)]
        n_excl_before = len(excluded)
        seg_points = []
        for si in range(len(bounds) - 1):
            lo, hi = bounds[si], bounds[si + 1]
            ts, rs, us, is_ = _reject_outliers(
                t[lo:hi], r[lo:hi], u[lo:hi], idx[lo:hi], unc_floor,
                excluded, f"段{si}精筛")
            seg_points.append((ts, rs, us, is_))
        if len(excluded) == n_excl_before:
            break
        t = np.concatenate([p[0] for p in seg_points])
        r = np.concatenate([p[1] for p in seg_points])
        u = np.concatenate([p[2] for p in seg_points])
        idx = [i for p in seg_points for i in p[3]]

    # Final segments from the cleaned series.
    segments: list[SegmentInfo] = []
    seg_fit: list[tuple[dict, float] | None] = []  # (fit, sigma2) per segment
    seg_ranges: list[tuple[float, float]] = []     # time coverage per segment
    for si, (ts, rs, us, is_) in enumerate(seg_points):
        b_lo = float(ts[0]) if len(ts) else 0.0
        b_hi = float(ts[-1]) if len(ts) else b_lo
        seg_ranges.append((b_lo, b_hi))
        all_obs = np.sort(np.concatenate([
            ts, [h[0] for h in hold
                 if (h[0] - b_lo) * (h[0] - b_hi) <= 0]])) if len(ts) else ts
        gaps = _gaps(all_obs)
        if len(ts) < MIN_SEG_POINTS:
            segments.append(SegmentInfo(
                t_lo=b_lo, t_hi=b_hi,
                t_ref=float(ts.mean()) if len(ts) else b_lo,
                n_fit=len(ts), n_holdout=0, status="insufficient",
                offset=None, offset_ci95=None, slope=None, slope_ci95=None,
                offset_var=None, slope_var=None, gaps=gaps,
                classification="数据不足，不给确定校准参数"))
            seg_fit.append(None)
            notes.append(
                f"段{si} 仅 {len(ts)} 个拟合点 (< {MIN_SEG_POINTS})："
                "数据不足，不给确定校准参数")
            continue
        fit = _line_fit(ts, rs)
        s2 = _sigma2(fit, unc_floor)
        var_a = s2 / fit["n"]
        var_b = s2 / fit["sxx"] if fit["sxx"] > 0 else float("inf")
        slope = fit["b"]
        slope_ci = 1.96 * math.sqrt(var_b)
        if abs(slope) > slope_ci:
            cls = f"稳定时钟速差 {slope * 1e6:+.1f} ppm"
        else:
            cls = "速率与标称时钟一致"
        segments.append(SegmentInfo(
            t_lo=float(ts[0]), t_hi=float(ts[-1]), t_ref=fit["t_ref"],
            n_fit=len(ts), n_holdout=0, status="ok",
            offset=fit["a"], offset_ci95=1.96 * math.sqrt(var_a),
            slope=slope, slope_ci95=slope_ci,
            offset_var=var_a, slope_var=var_b, gaps=gaps,
            classification=cls))
        seg_fit.append((fit, s2))

    jumps = [JumpInfo(time=s["time"], step=s["step"], step_ci95=s["ci95"],
                      note="设备重启/缓冲重建式跳变") for s in splits]
    for s in splits:
        notes.append(f"t≈{s['time']:.2f} s 检测到跳变 "
                     f"{s['step']*1000:+.1f} ms（95% 区间 ±{s['ci95']*1000:.1f} ms，"
                     "设备重启/缓冲重建特征）")
    if excluded:
        notes.append(f"{len(excluded)} 个偶发检测误差已剔除，"
                     "依据与数值见剔除记录（未静默删除）")

    # Validation on the held-out points only. Points landing where the model
    # claims no coverage (gap / extrapolation / insufficient segment) are NOT
    # folded into the corrected-RMS: they are their own visible category.
    vpoints: list[ValidationPoint] = []
    after_vals, before_vals = [], []
    n_uncovered = 0
    seg_holdout = [0] * len(segments)
    for ht, hr, hi_ in hold:
        si = _segment_of(seg_ranges, ht)
        status, after = "insufficient", None
        if si is not None and seg_fit[si] is not None:
            fit, s2 = seg_fit[si]
            status = _classify_pos(segments[si], ht)
            seg_holdout[si] += 1
            if status == "ok":
                after = hr - _predict(fit, ht)
                before_vals.append(hr)
                after_vals.append(after)
            else:
                n_uncovered += 1
        else:
            n_uncovered += 1
        vpoints.append(ValidationPoint(
            event_index=hi_, time=float(ht), residual_before=float(hr),
            residual_after=None if after is None else float(after),
            status=status))
    for si, seg in enumerate(segments):
        seg.n_holdout = seg_holdout[si]
    rms_before = (float(np.sqrt(np.mean(np.square(before_vals))))
                  if before_vals else None)
    rms_after = (float(np.sqrt(np.mean(np.square(after_vals))))
                 if after_vals else None)
    max_abs_after = (float(np.max(np.abs(after_vals)))
                     if after_vals else None)
    if not hold:
        notes.append("样本过少，无独立留出验证点")
    elif not after_vals:
        notes.append("留出点均落在数据不足/未覆盖区间，无法给出独立验证残差")
    if n_uncovered:
        notes.append(
            f"{n_uncovered} 个留出点位于空洞/外推/数据不足区间，模型在这些"
            "位置不声称精度，未计入校正后 RMS（逐点状态见验证记录）")

    return DriftAnalysis(
        segments=segments, jumps=jumps, excluded=excluded,
        validation_points=vpoints,
        rms_before=rms_before, rms_after=rms_after,
        max_abs_after=max_abs_after,
        t_lo=float(t_all[0]) if len(t_all) else 0.0,
        t_hi=float(t_all[-1]) if len(t_all) else 0.0,
        notes=notes,
        fit_series=[[float(x), float(y)] for x, y in zip(t, r)])


def _segment_of(seg_ranges: list[tuple[float, float]], x: float) -> int | None:
    if not seg_ranges:
        return None
    # boundaries sit halfway between neighbouring segment edges
    for i in range(len(seg_ranges) - 1):
        boundary = 0.5 * (seg_ranges[i][1] + seg_ranges[i + 1][0])
        if x < boundary:
            return i
    return len(seg_ranges) - 1


def _classify_pos(seg: SegmentInfo, x: float) -> str:
    if seg.status != "ok":
        return "insufficient"
    if x < seg.t_lo or x > seg.t_hi:
        return "extrapolated"
    for lo, hi in seg.gaps:
        if lo <= x <= hi:
            return "gap"
    return "ok"


def predict(analysis: DriftAnalysis, x: float) -> dict:
    """Model prediction at time x with an honest interval.

    In-range => status 'ok'. Inside a coverage gap => 'gap' (no confirming
    data). Outside the observed range => 'extrapolated'; the band comes from
    the regression itself and widens with distance, and is reported
    separately from in-range estimates."""
    seg = None
    for s in analysis.segments:
        if s.t_lo - 1e-9 <= x <= s.t_hi + 1e-9:
            seg = s
            break
    if seg is None and analysis.segments:
        # nearest segment by edge distance -> extrapolation
        seg = min(analysis.segments,
                  key=lambda s: min(abs(x - s.t_lo), abs(x - s.t_hi)))
    if seg is None or seg.status != "ok":
        return {"value": None, "lo": None, "hi": None, "status": "insufficient"}
    status = _classify_pos(seg, x)
    value = seg.offset + seg.slope * (x - seg.t_ref)
    var = seg.offset_var + (x - seg.t_ref) ** 2 * seg.slope_var
    half = 1.96 * math.sqrt(max(var, 0.0))
    return {"value": value, "lo": value - half, "hi": value + half,
            "status": status}
