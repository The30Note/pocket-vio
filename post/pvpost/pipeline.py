"""End-to-end processing of one session bundle."""
import json
from pathlib import Path

import numpy as np

from . import align, export
from .bundle import load_bundle


def process(bundle_path, out, ronin=None, use_mapanything=True, refine=False, scale_source="auto", chunk=32, overlap=8,
            ma_model="facebook/map-anything", max_side=None, device=None, io=None, ma_runner=None, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    b = load_bundle(bundle_path, out / "bundle")
    n = len(b.photos)
    if n < 3:
        raise SystemExit(f"bundle has {n} photos; need at least 3")
    names = [p.name for p in b.photos]
    times = np.array([p.t for p in b.photos])
    phone = b.phone_poses("kf")
    report = {"bundle": str(bundle_path), "photos": n, "imu_samples": int(len(b.imu)), "sources": {}}
    log(f"{n} photos, {len(b.imu)} IMU samples, phone path {export.path_length(phone):.2f} m")

    # ---- 1. vision poses (metric if MapAnything is used)
    vision, pts, cols, used_ma = phone.copy(), np.zeros((0, 3)), np.zeros((0, 3)), False
    if use_mapanything:
        runner = ma_runner
        if runner is None:
            try:
                from .mapanything_stage import MapAnythingRunner
                log(f"loading MapAnything ({ma_model})")
                runner = MapAnythingRunner(ma_model, device)
            except ImportError as e:
                log(f"MapAnything not installed ({e}); using the phone poses as the vision trajectory")
        if runner is not None:
            from .mapanything_stage import run_chunks
            chunks = run_chunks(runner, b, chunk, overlap, max_side=max_side, log=log)
            ma, pts, cols, cs = align.chain_chunks(chunks, n, log)
            k = 1.0 / float(np.median(cs))  # chunk 0's metric estimate is no better than any other chunk's
            ma[:, :3, 3] *= k
            pts = pts * k
            vision, pts, _ = align.gravity_align(ma, phone, pts)
            used_ma = True
            report["mapanything"] = {"chunks": len(chunks), "chunk_scale_spread": float(np.std(cs * k)), "chunk_scales": (cs * k).round(4).tolist()}
    vis_c = vision[:, :3, 3]
    report["sources"]["vision"] = {"kind": "mapanything (metric)" if used_ma else "phone VIO", "path_m": export.path_length(vision)}

    # ---- 2. scale candidates, all expressed as a factor applied to the vision trajectory
    cand = {}
    if used_ma:
        cand["mapanything"] = 1.0
        cand["phone_vio"] = align.sim3_scale(vis_c, phone[:, :3, 3])
    else:
        cand["phone_vio"] = 1.0
    if io is None and ronin:
        from .inertial import load_ronin, run_ronin
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log(f"running RoNIN inertial odometry ({ronin}) on {dev}")
        bg = b.session.get("estimator", {}).get("lastBg") or b.session.get("estimator", {}).get("initBg")
        io = run_ronin(load_ronin(ronin, dev), b.imu, bg=bg, device=dev)
    if io is not None:
        r = align.inertial_scale(io, times, vis_c, log)
        if r:
            cand["inertial"] = r["scale"]
            report["sources"]["inertial"] = {**r, "path_m": io.path_length(times[0], times[-1])}
    report["scale_candidates"] = {k: round(v, 5) for k, v in cand.items()}
    src = scale_source
    if src == "auto":
        src = "inertial" if "inertial" in cand else "mapanything" if "mapanything" in cand else "phone_vio"
    if src not in cand:
        raise SystemExit(f"scale source '{src}' unavailable; have {list(cand)}")
    f = cand[src]
    log("scale candidates (factor on the vision trajectory): " + ", ".join(f"{k}={v:.4f}" for k, v in cand.items()) + f"  -> using {src}")
    o = vis_c[0].copy()
    final = vision.copy()
    final[:, :3, 3] = o + f * (vis_c - o)
    pts = o + f * (pts - o) if len(pts) else pts
    report["scale_source"], report["scale_factor"] = src, f

    # ---- 3. optional photogrammetric refinement (keeps the chosen scale)
    img_dir = export.link_images(b, out, names)
    K, size = b.K(), (b.camera["width"], b.camera["height"])
    if refine:
        from .refine import refine as do_refine
        final, rpts, rcols, info = do_refine(img_dir, names, K, size, final, out / "colmap_work", log=log)
        report["refine"] = info
        if len(rpts):
            pts, cols = rpts, rcols
    report["final_path_m"] = export.path_length(final)

    # ---- 4. outputs
    export.write_colmap_text(out / "sparse" / "0", K, size, names, final, pts if len(pts) else None, cols if len(cols) else None)
    export.write_transforms_json(out / "transforms.json", K, size, names, final)
    export.write_tum(out / "poses_tum.txt", times, final)
    if len(pts):
        export.write_ply(out / "points.ply", pts, cols)
    report["walk_tests_on_phone"] = b.session.get("walks", [])
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    md = [f"# Pocket VIO post-processing\n", f"- photos: {n}", f"- vision trajectory: {report['sources']['vision']['kind']}, path {report['sources']['vision']['path_m']:.2f} m",
          "- scale candidates (factor applied to the vision trajectory):"] + [f"  - {k}: {v:.4f}" for k, v in cand.items()] + \
         [f"- chosen: **{src}** → final path **{report['final_path_m']:.2f} m**"]
    if "inertial" in report["sources"]:
        r = report["sources"]["inertial"]
        md.append(f"- inertial ratio ±{r['sigma']:.4f} (per-pair spread {r['spread']:.3f}, {r['pairs']} pairs)")
    if "mapanything" in report:
        md.append(f"- MapAnything chunk scale spread: {report['mapanything']['chunk_scale_spread']:.4f}")
    if "refine" in report:
        md.append(f"- refinement: {report['refine']['points']} points, mean reprojection error {report['refine']['mean_reproj_error']:.2f} px")
    (out / "report.md").write_text("\n".join(md) + "\n")
    log(f"done: {out}/sparse/0 (COLMAP), transforms.json, poses_tum.txt, report.md")
    return report, final
