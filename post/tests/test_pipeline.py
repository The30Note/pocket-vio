"""python tests/test_pipeline.py   (or pytest)  — synthetic end-to-end checks of pvpost."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pvpost.geom import umeyama, rot_angle_deg  # noqa: E402
from pvpost.inertial import InertialOdometry  # noqa: E402
from pvpost.pipeline import process  # noqa: E402
from tests.synth import MockMapAnything, body_pose, make_session  # noqa: E402

RESULTS = []


def check(name, ok, msg):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {msg}")


def oracle_io(seconds, noise=0.03, seed=3):
    """Stand-in for RoNIN: true horizontal velocity (1 s window means) with 3 % noise, rotated by an
    unknown heading (as RoNIN's gravity-aligned frame has arbitrary yaw)."""
    rng = np.random.default_rng(seed)
    tc = np.arange(0.5, seconds - 0.5, 0.05)
    v = np.array([(body_pose(t + 0.5)[0] - body_pose(t - 0.5)[0])[:2] for t in tc]) * (1 + rng.normal(0, noise, (len(tc), 1)))
    a = 1.1
    Rz = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return InertialOdometry(tc, v @ Rz.T)


def errors(final, gt):
    c, g = final[:, :3, 3], gt[:, :3, 3]
    s, R, t = umeyama(c, g, True)          # scale left over after the best similarity alignment
    _, R1, t1 = umeyama(c, g, False)       # rigid alignment only: what metric accuracy looks like
    ate = np.sqrt(np.mean(np.sum((c @ R1.T + t1 - g) ** 2, 1)))
    rot = np.mean([rot_angle_deg(R1 @ final[i, :3, :3], gt[i, :3, :3]) for i in range(len(gt))])
    return s, ate, rot


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pvpost_"))
    print("rendering synthetic session in", tmp)
    gt, hashes, imu = make_session(tmp / "session", seconds=10.0)
    path = sum(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1))
    print(f"{len(gt)} photos, true camera path {path:.2f} m")
    io = oracle_io(10.0)
    quiet = lambda *a: None

    rep, fin = process(tmp / "session", tmp / "out_ma", use_mapanything=True, chunk=10, overlap=4, io=io, ma_runner=MockMapAnything(gt, hashes), log=quiet)
    s, ate, rot = errors(fin, gt)
    check("MapAnything chunks + inertial scale", abs(s - 1) < 0.03 and ate < 0.1, f"residual scale {s:.4f}, ATE {ate:.3f} m, rot {rot:.2f}°, candidates {rep['scale_candidates']}")

    rep, fin = process(tmp / "session", tmp / "out_ma2", use_mapanything=True, chunk=10, overlap=4, scale_source="mapanything", ma_runner=MockMapAnything(gt, hashes), log=quiet)
    s, ate, rot = errors(fin, gt)
    check("MapAnything metric scale only", abs(s - 1) < 0.04, f"residual scale {s:.4f}, ATE {ate:.3f} m, chunk spread {rep['mapanything']['chunk_scale_spread']:.4f}")

    rep, fin = process(tmp / "session", tmp / "out_phone", use_mapanything=False, io=io, log=quiet)
    s, ate, rot = errors(fin, gt)
    check("phone poses rescaled by inertial odometry", abs(s - 1) < 0.03, f"residual scale {s:.4f} (phone VIO alone was 1/0.8 = 1.25 off)")

    import pycolmap
    rec = pycolmap.Reconstruction(tmp / "out_ma" / "sparse" / "0")
    check("COLMAP text model loads in pycolmap", rec.num_images() == len(gt) and rec.num_cameras() == 1, f"{rec.num_images()} images, {rec.num_points3D()} points")

    rep, fin = process(tmp / "session", tmp / "out_refine", use_mapanything=False, io=io, refine=True, log=quiet)
    s, ate, rot = errors(fin, gt)
    r = rep["refine"]
    check("pycolmap refinement keeps metric scale and improves poses", abs(s - 1) < 0.03 and r["points"] > 200,
          f"{r['points']} points, reproj {r['mean_reproj_error']:.2f} px, residual scale {s:.4f}, ATE {ate:.3f} m, rot {rot:.2f}°")

    try:
        import torch
        from pvpost.inertial import build_ronin_resnet, load_ronin, run_ronin
        net = build_ronin_resnet()
        keys = list(net.state_dict())
        ok = keys[0] == "input_block.0.weight" and "output_block.fc.6.bias" in keys and any(k.startswith("residual_groups.1.0.downsample.0") for k in keys)
        torch.save({"model_state_dict": net.state_dict(), "epoch": 0}, tmp / "ronin.pt")
        io2 = run_ronin(load_ronin(tmp / "ronin.pt"), imu)
        check("RoNIN ResNet1D: checkpoint layout + inference shapes", ok and io2.v.shape[1] == 2, f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M params, {len(io2.t)} windows")
    except ImportError:
        print("SKIP  RoNIN network test (torch not installed)")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    return all(RESULTS)


def test_pipeline():
    assert main()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
