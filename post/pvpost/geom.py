"""Small rigid-body helpers. Quaternions are (x, y, z, w); poses are 4x4 camera-to-world."""
import numpy as np


def quat_to_R(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def project_to_SO3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def rotation_mean(Rs):
    """Chordal L2 mean of rotation matrices."""
    return project_to_SO3(np.sum(Rs, axis=0))


def umeyama(src, dst, with_scale=True):
    """Find s, R, t minimising |dst - (s R src + t)|^2. src, dst: (N, 3)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    S = b.T @ a / len(src)
    U, D, Vt = np.linalg.svd(S)
    E = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ E @ Vt
    var = (a ** 2).sum() / len(src)
    s = (D * np.diag(E)).sum() / var if with_scale and var > 0 else 1.0
    return s, R, md - s * R @ ms


def apply_sim3(T, s, R, t):
    """Transform a camera-to-world pose by x -> s R x + t (orientation rotates, scale on position)."""
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = s * R @ T[:3, 3] + t
    return out


def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return 0.5 * v if th < 1e-6 else th / (2 * np.sin(th)) * v


def so3_exp(w):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def rot_angle_deg(Ra, Rb):
    return np.degrees(np.linalg.norm(so3_log(Ra.T @ Rb)))
