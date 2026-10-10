"""Outputs: COLMAP text model (LichtFeld-Studio / gsplat / COLMAP), nerfstudio transforms.json,
TUM trajectory, PLY points and a JSON/Markdown report."""
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .geom import R_to_quat


def link_images(bundle, out, names):
    d = Path(out) / "images"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        src, dst = bundle.image_dir / n, d / n
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return d


def write_colmap_text(path, K, size, names, poses, points=None, colors=None, image_ids=None, camera_id=1):
    """poses: camera-to-world (OpenCV). COLMAP stores world-to-camera."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    w, h = size
    (p / "cameras.txt").write_text("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
                                   f"{camera_id} PINHOLE {w} {h} {K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}\n")
    lines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", "# POINTS2D[] as (X, Y, POINT3D_ID)"]
    for k, (n, T) in enumerate(zip(names, poses)):
        Rcw = T[:3, :3].T
        tcw = -Rcw @ T[:3, 3]
        x, y, z, qw = R_to_quat(Rcw)
        iid = image_ids[n] if image_ids else k + 1
        lines += [f"{iid} {qw:.9f} {x:.9f} {y:.9f} {z:.9f} {tcw[0]:.6f} {tcw[1]:.6f} {tcw[2]:.6f} {camera_id} {n}", ""]
    (p / "images.txt").write_text("\n".join(lines) + "\n")
    pl = ["# POINT3D_ID X Y Z R G B ERROR TRACK[]"]
    if points is not None:
        cols = np.clip(colors if colors is not None else np.full((len(points), 3), 200), 0, 255).astype(int)
        for i, (X, c) in enumerate(zip(points, cols)):
            pl.append(f"{i + 1} {X[0]:.5f} {X[1]:.5f} {X[2]:.5f} {c[0]} {c[1]} {c[2]} 0")
    (p / "points3D.txt").write_text("\n".join(pl) + "\n")


def write_transforms_json(path, K, size, names, poses):
    """nerfstudio convention: camera-to-world with OpenGL camera axes (y up, z backwards)."""
    flip = np.diag([1, -1, -1, 1.0])
    frames = [{"file_path": f"images/{n}", "transform_matrix": (T @ flip).tolist()} for n, T in zip(names, poses)]
    data = {"camera_model": "OPENCV", "w": size[0], "h": size[1], "fl_x": K[0, 0], "fl_y": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
            "k1": 0, "k2": 0, "p1": 0, "p2": 0, "frames": frames}
    Path(path).write_text(json.dumps(data, indent=1))


def write_tum(path, times, poses):
    with open(path, "w") as f:
        f.write("# t x y z qx qy qz qw  (camera centre and orientation, camera-to-world, OpenCV camera)\n")
        for t, T in zip(times, poses):
            q = R_to_quat(T[:3, :3])
            f.write(f"{t:.6f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


def write_ply(path, points, colors=None):
    cols = np.clip(colors if colors is not None else np.full((len(points), 3), 200), 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for X, c in zip(points, cols):
            f.write(f"{X[0]:.5f} {X[1]:.5f} {X[2]:.5f} {c[0]} {c[1]} {c[2]}\n")


def path_length(poses):
    c = poses[:, :3, 3]
    c = c[~np.isnan(c).any(1)]
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) if len(c) > 1 else 0.0
