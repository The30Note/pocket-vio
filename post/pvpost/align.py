"""Chunk chaining, gravity alignment and the scale consensus between independent sources."""
import numpy as np

from .geom import apply_sim3, rotation_mean, umeyama


def chain_chunks(chunks, n, log=print):
    """Bring every chunk into chunk 0's frame with a Sim(3) fitted on shared photos.

    Each chunk is metric on its own; c_i is the scale applied to chunk i. With perfectly metric
    chunks c_i = 1. Returns poses (n,4,4; NaN where missing), points, colors, chunk scales."""
    poses = np.full((n, 4, 4), np.nan)
    pts, cols, scales = [], [], []
    for ci, ch in enumerate(chunks):
        idx, P = ch["idx"], ch["poses"]
        if ci == 0:
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            shared = [k for k, i in enumerate(idx) if not np.isnan(poses[i, 0, 0])]
            if len(shared) < 3:
                raise RuntimeError(f"chunk {ci} shares {len(shared)} photos with earlier chunks; increase the overlap")
            src = P[shared, :3, 3]
            dst = np.stack([poses[idx[k], :3, 3] for k in shared])
            s, R, t = umeyama(src, dst, True)
            # rotation from camera orientations is better conditioned than from a short baseline
            R = rotation_mean([poses[idx[k], :3, :3] @ P[k, :3, :3].T for k in shared])
            t = (dst - s * (R @ src.T).T).mean(0)
        scales.append(s)
        for k, i in enumerate(idx):
            if np.isnan(poses[i, 0, 0]):
                poses[i] = apply_sim3(P[k], s, R, t)
        pts.append(s * (R @ ch["pts"].T).T + t)
        cols.append(ch["cols"])
    scales = np.array(scales)
    log(f"  chunk scales relative to chunk 0: {np.round(scales, 3).tolist()}")
    return poses, np.concatenate(pts), np.concatenate(cols), scales


def gravity_align(poses, phone_poses, points=None):
    """Rotate/translate the vision frame so orientations match the phone's gravity-aligned frame
    (z up) and the first photo sits where the phone put it. No scale change."""
    ok = ~np.isnan(poses[:, 0, 0])
    R = rotation_mean([phone_poses[i, :3, :3] @ poses[i, :3, :3].T for i in np.where(ok)[0]])
    i0 = np.where(ok)[0][0]
    t = phone_poses[i0, :3, 3] - R @ poses[i0, :3, 3]
    out = np.array([apply_sim3(P, 1.0, R, t) if ok[i] else P for i, P in enumerate(poses)])
    return out, (None if points is None else (R @ points.T).T + t), (R, t)


def horizontal_pairs(times, centers, min_dt=2.0, max_dt=8.0, min_dist=0.4):
    """Photo pairs a few seconds apart with real horizontal travel between them."""
    pairs, n = [], len(times)
    j = 0
    for i in range(n):
        j = max(j, i + 1)
        while j < n and times[j] - times[i] < min_dt:
            j += 1
        if j >= n or times[j] - times[i] > max_dt:
            continue
        if np.linalg.norm(centers[j, :2] - centers[i, :2]) >= min_dist:
            pairs.append((i, j))
    return pairs


def inertial_scale(io, times, centers, log=print):
    """Ratio |IMU displacement| / |vision displacement| over horizontal pairs (robust median).

    Norms are compared, so the IMU frame's heading drift does not matter."""
    pairs = horizontal_pairs(times, centers)
    if len(pairs) < 3:
        return None
    r = np.array([np.linalg.norm(io.displacement(times[i], times[j])) / np.linalg.norm(centers[j, :2] - centers[i, :2]) for i, j in pairs])
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) * 1.4826
    log(f"  inertial/vision ratio over {len(pairs)} pairs: {med:.4f} ± {mad / np.sqrt(len(r)):.4f} (spread {mad:.3f})")
    return {"scale": med, "sigma": mad / np.sqrt(len(r)), "spread": mad, "pairs": len(pairs)}


def sim3_scale(src_centers, dst_centers):
    ok = ~(np.isnan(src_centers).any(1) | np.isnan(dst_centers).any(1))
    s, _, _ = umeyama(src_centers[ok], dst_centers[ok], True)
    return float(s)
