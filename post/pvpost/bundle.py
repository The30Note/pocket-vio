"""Read a Pocket VIO session bundle (.zip or extracted folder)."""
import csv
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geom import make_T, quat_to_R


@dataclass
class Photo:
    name: str
    t: float
    live: np.ndarray            # camera-to-world (OpenCV camera), phone live estimate
    kf: np.ndarray | None       # camera-to-world from the refined keyframe trajectory
    gyro_rate: float = 0.0


@dataclass
class Bundle:
    root: Path
    camera: dict
    session: dict
    photos: list
    imu: np.ndarray             # (N, 7): t, gx, gy, gz, ax, ay, az (device frame)
    keyframes: np.ndarray       # (M, 8): t, x, y, z, qx, qy, qz, qw (IMU body in world)
    extra: dict = field(default_factory=dict)

    @property
    def image_dir(self):
        return self.root / "images"

    def K(self, w=None, h=None):
        c = self.camera
        sx = 1.0 if w is None else w / c["width"]
        sy = 1.0 if h is None else h / c["height"]
        return np.array([[c["fx"] * sx, 0, c["cx"] * sx], [0, c["fy"] * sy, c["cy"] * sy], [0, 0, 1.0]])

    def phone_poses(self, which="kf"):
        """Phone camera-to-world poses per photo ('kf' falls back to 'live')."""
        return np.stack([(p.kf if which == "kf" and p.kf is not None else p.live) for p in self.photos])


def _pose(vals):
    if any(v == "" for v in vals):
        return None
    v = [float(x) for x in vals]
    return make_T(quat_to_R(v[3:7]), v[0:3])


def load_bundle(path, workdir=None):
    path = Path(path)
    if path.is_file() and path.suffix == ".zip":
        workdir = Path(workdir or path.with_suffix(""))
        with zipfile.ZipFile(path) as z:
            z.extractall(workdir)
        roots = [p for p in workdir.iterdir() if (p / "session.json").exists()]
        root = roots[0] if roots else workdir
    else:
        root = path
    camera = json.loads((root / "camera.json").read_text())
    session = json.loads((root / "session.json").read_text())
    photos = []
    with open(root / "photos.csv") as f:
        for r in csv.DictReader(f):
            live = _pose([r[k] for k in ("live_px", "live_py", "live_pz", "live_qx", "live_qy", "live_qz", "live_qw")])
            kf = _pose([r[k] for k in ("kf_px", "kf_py", "kf_pz", "kf_qx", "kf_qy", "kf_qz", "kf_qw")])
            photos.append(Photo(r["name"], float(r["t"]), live, kf, float(r.get("gyro_rate") or 0)))
    photos.sort(key=lambda p: p.t)
    imu = np.loadtxt(root / "imu.csv", delimiter=",", skiprows=1, ndmin=2)
    kfp = root / "keyframes_tum.txt"
    keyframes = np.loadtxt(kfp, comments="#", ndmin=2) if kfp.exists() and kfp.stat().st_size > 60 else np.zeros((0, 8))
    return Bundle(root, camera, session, photos, imu, keyframes)
