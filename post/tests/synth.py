"""Synthetic Pocket VIO session: a textured corridor, a 10 m walk with gait sway, photos
ray-cast from the true poses, 200 Hz IMU, and deliberately wrong phone poses (scale 0.8)."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from pvpost.geom import R_to_quat, make_T, so3_exp

BOX = np.array([[-2.0, 2.0], [-1.0, 13.0], [-1.4, 1.4]])
RBC = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])  # camera axes in the device frame (portrait, back camera)
R0 = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])     # device upright, back camera looking at world +y
W, H, F = 320, 240, 250.0
RES = 80  # texture px per metre


def textures(rng):
    tex = {}
    for face, (a, b) in {0: (1, 2), 1: (1, 2), 2: (0, 2), 3: (0, 2), 4: (0, 1), 5: (0, 1)}.items():
        sa, sb = BOX[a, 1] - BOX[a, 0], BOX[b, 1] - BOX[b, 0]
        n = rng.random((int(sa * RES) + 2, int(sb * RES) + 2))
        t = 0.6 * gaussian_filter(n, 1.2) + 0.4 * gaussian_filter(rng.random(n.shape), 6)
        t = (t - t.min()) / (t.max() - t.min())
        tex[face] = (40 + 200 * t).astype(np.uint8)
    return tex


def render(tex, Twc):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], -1) @ Twc[:3, :3].T
    c = Twc[:3, 3]
    best = np.full(u.shape, np.inf)
    out = np.zeros(u.shape, np.uint8)
    for ax in range(3):
        for side in range(2):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (BOX[ax, side] - c[ax]) / d[..., ax]
            P = c + t[..., None] * d
            o = [k for k in range(3) if k != ax]
            ok = (t > 1e-3) & (t < best)
            for k in o:
                ok &= (P[..., k] >= BOX[k, 0]) & (P[..., k] <= BOX[k, 1])
            face = 2 * ax + side
            ia = ((P[..., o[0]] - BOX[o[0], 0]) * RES).astype(int).clip(0, tex[face].shape[0] - 1)
            ib = ((P[..., o[1]] - BOX[o[1], 0]) * RES).astype(int).clip(0, tex[face].shape[1] - 1)
            out = np.where(ok, tex[face][ia, ib], out)
            best = np.where(ok, t, best)
    return out


def body_pose(t):
    """Walking along +y at ~1 m/s with lateral sway, vertical bounce and slow yaw sweeps."""
    e = min(1.0, t / 1.5)  # ease in
    y = 1.0 * (t - 1.5 * (1 - e) ** 2 * 0.5) if t < 1.5 else 1.0 * (t - 0.375)
    p = np.array([0.04 * np.sin(2 * np.pi * 0.9 * t) * e, max(0.0, y), 0.025 * np.sin(2 * np.pi * 1.8 * t) * e])
    yaw = 0.3 * np.sin(0.6 * t) * e
    pitch = 0.05 * np.sin(1.1 * t)
    R = so3_exp(np.array([0, 0, yaw])) @ so3_exp(np.array([pitch, 0, 0])) @ R0
    return p, R


def make_session(out, seconds=10.0, photo_every=0.4, seed=0):
    rng = np.random.default_rng(seed)
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    tex = textures(rng)
    # IMU at 200 Hz from the true motion (finite differences)
    ts = np.arange(0, seconds, 1 / 200)
    h = 1e-3
    rows = []
    for t in ts:
        p0, R = body_pose(t)
        acc = (body_pose(t + h)[0] - 2 * p0 + body_pose(t - h)[0]) / h ** 2
        Rm, Rp = body_pose(t - h)[1], body_pose(t + h)[1]
        from pvpost.geom import so3_log
        w = so3_log(Rm.T @ Rp) / (2 * h)
        a = R.T @ (acc + np.array([0, 0, 9.81]))
        rows.append([t, *(w + rng.normal(0, 0.01, 3)), *(a + rng.normal(0, 0.08, 3))])
    imu = np.array(rows)
    np.savetxt(out / "imu.csv", imu, delimiter=",", header="t,gx,gy,gz,ax,ay,az", comments="", fmt="%.6f")
    # photos every photo_every seconds; phone poses: scale 0.8 + drift + noise (a poor VIO)
    gt, lines, hashes = [], [], {}
    hdr = "name,t,status,live_px,live_py,live_pz,live_qx,live_qy,live_qz,live_qw,kf_px,kf_py,kf_pz,kf_qx,kf_qy,kf_qz,kf_qw,kf_extrap_s,gyro_rate"
    for k, t in enumerate(np.arange(1.0, seconds - 0.2, photo_every)):
        p, R = body_pose(t)
        Twc = make_T(R @ RBC, p)
        img = render(tex, Twc)
        name = f"img_{k:05d}.jpg"
        Image.fromarray(img).convert("RGB").save(out / "images" / name, quality=95)
        hashes[hashlib.md5(np.asarray(Image.open(out / "images" / name).convert("RGB")).tobytes()).hexdigest()] = k
        gt.append(Twc)
        pp = 0.8 * p + np.array([0.02 * t, 0, 0]) + rng.normal(0, 0.02, 3)
        Rp = so3_exp(rng.normal(0, 0.015, 3)) @ R @ RBC
        q = R_to_quat(Rp)
        pose = ",".join(f"{x:.6f}" for x in [*pp, *q])
        lines.append(f"{name},{t:.6f},tracking,{pose},{pose},0.000,0.3")
    (out / "photos.csv").write_text(hdr + "\n" + "\n".join(lines) + "\n")
    (out / "camera.json").write_text(json.dumps({"model": "PINHOLE", "width": W, "height": H, "fx": F, "fy": F, "cx": W / 2, "cy": H / 2}))
    (out / "session.json").write_text(json.dumps({"app": "pocket-vio", "format": 1, "estimator": {"lastBg": [0, 0, 0]}}))
    (out / "keyframes_tum.txt").write_text("# none\n")
    (out / "frames.csv").write_text("t,tracking,px,py,pz,qx,qy,qz,qw\n")
    return np.stack(gt), hashes, imu


class MockMapAnything:
    """Stands in for MapAnything: returns the true poses of the views it is given, each chunk in
    its own random frame and with a +-2 % metric scale error, plus points on the corridor walls."""

    def __init__(self, gt, hashes, seed=1):
        self.gt, self.hashes, self.rng = gt, hashes, np.random.default_rng(seed)

    def infer(self, images, K=None, poses=None):
        idx = [self.hashes[hashlib.md5(np.ascontiguousarray(im).tobytes()).hexdigest()] for im in images]
        s = 1 + self.rng.normal(0, 0.02)
        Rr = so3_exp(self.rng.normal(0, 1.0, 3))
        tr = self.rng.normal(0, 3, 3)
        out = []
        for i in idx:
            T = self.gt[i]
            Tl = np.eye(4); Tl[:3, :3] = Rr @ T[:3, :3]; Tl[:3, 3] = s * Rr @ T[:3, 3] + tr
            P = self.rng.uniform(BOX[:, 0], BOX[:, 1], (300, 3)); P[:, 0] = np.where(P[:, 0] > 0, BOX[0, 1], BOX[0, 0])
            pts = (s * (Rr @ P.T).T + tr).reshape(15, 20, 3)
            out.append({"pose": Tl, "K": K[0] if K is not None else None, "pts": pts, "mask": np.ones((15, 20), bool), "img": np.full((15, 20, 3), 128.0), "scale": s})
        return out
