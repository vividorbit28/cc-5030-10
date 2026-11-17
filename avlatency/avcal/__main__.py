"""Command-line entry point.

Start the multi-client web panel (the primary delay-calibration entry):
    python -m avcal serve --host 0.0.0.0 --port 8000

Headless helpers (auxiliary only; delay calibration itself lives in the
panel's 延迟调整 section):
    python -m avcal generate --out workdir/clip
    python -m avcal simulate --out workdir/run [chain options]
    python -m avcal analyze  --recording path/to/capture.mp4
"""
from __future__ import annotations

import argparse
import json
import os

from .calibrate import export_playback_params, run_on_recording
from .chain import ChainConfig, run_chain
from .media import MediaConfig, generate
from .reader import read_recording


def _print_result(result, ground_truth=None):
    d = result.to_dict()
    print(f"verdict: {d['verdict'].upper()}")
    for r in d["reasons"]:
        print(f"  - {r}")
    for a in d["audio"]:
        print(f"audio ch{a['channel']}: offset={a['offset']*1000:.2f} ms "
              f"[{a['offset_lo']*1000:.2f},{a['offset_hi']*1000:.2f}] "
              f"jitter={a['jitter_std']*1000:.3f} ms "
              f"matched={a['n_matched']} dropped={a['n_dropped']} "
              f"dup={a['n_duplicates']}")
    v = d["video"]
    print(f"video:        offset={v['offset']*1000:.2f} ms "
          f"[{v['offset_lo']*1000:.2f},{v['offset_hi']*1000:.2f}] "
          f"matched={v['n_matched']} dropped={v['n_dropped']} "
          f"rejected={v['n_rejected']}")
    for o in d["av_offsets"]:
        print(f"AV ch{o['channel']}: audio-video="
              f"{o['audio_minus_video_ms']:.2f} ms ({o['interpretation']})")
    for key, da in d.get("drift", {}).items():
        print(f"drift[{key}]:")
        for s in da["segments"]:
            if s["status"] != "ok":
                print(f"  段 [{s['t_lo']:.2f},{s['t_hi']:.2f}] 数据不足，"
                      "不给确定参数")
                continue
            print(f"  段 [{s['t_lo']:.2f},{s['t_hi']:.2f}] "
                  f"offset={s['offset']*1000:.2f}±{s['offset_ci95']*1000:.2f} ms "
                  f"@t={s['t_ref']:.2f}s slope={s['slope']*1e6:+.1f}±"
                  f"{s['slope_ci95']*1e6:.1f} ppm ({s['classification']})"
                  + (f" 空洞={['%.2f-%.2f' % tuple(g) for g in s['gaps']]}"
                     if s["gaps"] else ""))
        for j in da["jumps"]:
            print(f"  跳变 t={j['time']:.2f}s step={j['step']*1000:+.1f}±"
                  f"{j['step_ci95']*1000:.1f} ms ({j['note']})")
        for e in da["excluded"]:
            print(f"  剔除 ev{e['event_index']} "
                  f"residual={e['residual']*1000:.1f} ms: {e['reason']}")
        if da["rms_after"] is not None:
            print(f"  独立留出验证: RMS {da['rms_before']*1000:.2f} -> "
                  f"{da['rms_after']*1000:.2f} ms "
                  f"(n={len(da['validation_points'])})")
    if ground_truth:
        print("ground truth:", json.dumps(ground_truth))
    print("suggested delay (ms):",
          json.dumps({k: round(x, 2) for k, x in
                      d["suggested_delay_ms"].items()}))


def cmd_generate(args):
    cfg = MediaConfig(n_events=args.events, fps=args.fps,
                      channels=args.channels)
    out = generate(cfg, args.out)
    print("wrote", out["paths"])


def _parse_pairs(items):
    out = []
    for raw in items or []:
        a, b = raw.split(":")
        out.append((float(a), float(b)))
    return out


def cmd_simulate(args):
    cfg = MediaConfig(n_events=args.events, fps=args.fps,
                      channels=args.channels)
    gen = generate(cfg, os.path.join(args.out, "media"))
    drops = [int(x) for x in args.drop.split(",") if x.strip()]
    chain = ChainConfig(
        sample_rate=cfg.sample_rate, channels=cfg.channels,
        fps_source=cfg.fps, fps_recorded=args.recorded_fps,
        audio_delay=[args.audio_delay] * cfg.channels,
        video_delay=args.video_delay, audio_noise_std=args.noise,
        dropped_events=drops, echo_delay=args.echo,
        echo_gain=args.echo_gain, exposure_drift=args.drift,
        clock_drift_ppm=args.clock_drift_ppm,
        delay_steps=_parse_pairs(args.delay_step),
        misaligned_events=[(int(a), b) for a, b in
                           _parse_pairs(args.misalign)])
    rec = run_chain(gen["frames"], gen["audio"], gen["events"], chain)
    result, _, _ = run_on_recording(
        rec.audio, rec.luminance, rec.frame_times, gen["events"], cfg,
        rec.recorded_fps, origin_known=True)
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
    if args.export_params:
        doc = export_playback_params(result)
        with open(args.export_params, "w") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
        print(f"播放系统参数已导出: {args.export_params} "
              "（仅生成参数文件，不修改任何设备设置）")
    _print_result(result, rec.ground_truth)


def cmd_analyze(args):
    cfg = MediaConfig(channels=args.channels)
    gen = generate(cfg, os.path.join(args.out, "reference"))
    rec = read_recording(args.recording, cfg)
    result, _, _ = run_on_recording(
        rec["audio"], rec["luminance"], rec["frame_times"], gen["events"],
        cfg, rec["recorded_fps"], origin_known=False)
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    _print_result(result)


def cmd_serve(args):
    from .web import create_server
    workdir = os.path.abspath(args.workdir)
    httpd = create_server(args.host, args.port, workdir)
    print(f"校准面板已启动: http://{args.host}:{args.port}/")
    print(f"局域网其它设备访问: http://<本机IP>:{args.port}/  (工作目录 {workdir})")
    print("按 Ctrl+C 停止。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


def build_parser():
    p = argparse.ArgumentParser(prog="avcal",
                                description="音画输出延迟校准工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="启动多端 Web 控制面板")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workdir", default="workdir/web")
    s.set_defaults(func=cmd_serve)

    g = sub.add_parser("generate", help="仅生成校准媒体")
    g.add_argument("--out", required=True)
    g.add_argument("--events", type=int, default=24)
    g.add_argument("--fps", type=float, default=30)
    g.add_argument("--channels", type=int, default=2)
    g.set_defaults(func=cmd_generate)

    m = sub.add_parser("simulate", help="可控录制链 + 校准（辅助）")
    m.add_argument("--out", required=True)
    m.add_argument("--events", type=int, default=24)
    m.add_argument("--fps", type=float, default=30)
    m.add_argument("--recorded-fps", type=float, default=30)
    m.add_argument("--channels", type=int, default=2)
    m.add_argument("--audio-delay", type=float, default=0.12)
    m.add_argument("--video-delay", type=float, default=0.18)
    m.add_argument("--noise", type=float, default=0.01)
    m.add_argument("--drop", default="", help="逗号分隔的丢弃事件编号")
    m.add_argument("--echo", type=float, default=0.0)
    m.add_argument("--echo-gain", type=float, default=0.5)
    m.add_argument("--drift", type=float, default=0.0)
    m.add_argument("--clock-drift-ppm", type=float, default=0.0,
                   help="录制时钟速差 ppm（线性漂移）")
    m.add_argument("--delay-step", action="append", default=[],
                   metavar="T:DT", help="在媒体时刻 T 秒处增加 DT 秒固定延迟"
                   "（设备重启式跳变），可重复")
    m.add_argument("--misalign", action="append", default=[],
                   metavar="IDX:SHIFT", help="把事件 IDX 的脉冲平移 SHIFT 秒"
                   "（单次误匹配），可重复")
    m.add_argument("--export-params", default=None,
                   help="把播放系统参数写入指定 JSON 文件")
    m.set_defaults(func=cmd_simulate)

    a = sub.add_parser("analyze", help="分析真实录制文件")
    a.add_argument("--recording", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--channels", type=int, default=2)
    a.set_defaults(func=cmd_analyze)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "serve":
        os.makedirs(os.path.abspath(args.workdir), exist_ok=True)
    else:
        os.makedirs(args.out, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
