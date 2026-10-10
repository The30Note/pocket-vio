"""Learned inertial odometry (RoNIN ResNet) -> metric horizontal velocity from the IMU alone.

RoNIN (Herath et al., ICRA 2020) regresses the average horizontal velocity of a 1 s window
of 200 Hz gyro + accelerometer data expressed in a gravity-aligned frame (heading arbitrary).
We re-implement its ResNet1D so the published checkpoint (model_state_dict) loads as-is:
download the RoNIN models from https://doi.org/10.20383/102.0543 and pass the ResNet
checkpoint path. The network is never used for orientation; only its speed/displacement
feeds the scale estimate, so heading drift of the IMU-only frame does not matter.
"""
import numpy as np

GRAV = 9.81


# ---------------------------------------------------------------- IMU preprocessing
def resample(imu, rate=200.0):
    """Linear resampling of (t, gx, gy, gz, ax, ay, az) rows onto a uniform grid."""
    t = imu[:, 0]
    keep = np.concatenate([[True], np.diff(t) > 0])
    imu = imu[keep]
    t = imu[:, 0]
    tg = np.arange(t[0], t[-1], 1.0 / rate)
    out = np.empty((len(tg), 7))
    out[:, 0] = tg
    for k in range(1, 7):
        out[:, k] = np.interp(tg, t, imu[:, k])
    return out


def _qmul(a, b):  # (w, x, y, z)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qrot(q, v):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R @ v, R


def gravity_aligned(imu, bg=None, kp=1.0, ki=0.0):
    """Mahony filter: body->world rotation with world z up (yaw free). Returns (N,3,3)."""
    t, g, a = imu[:, 0], imu[:, 1:4].copy(), imu[:, 4:7]
    if bg is not None:
        g = g - np.asarray(bg)[None]
    n0 = max(1, min(len(t), int(0.5 / max(1e-3, np.median(np.diff(t))))))
    a0 = a[:n0].mean(0)
    roll, pitch = np.arctan2(a0[1], a0[2]), np.arctan2(-a0[0], np.hypot(a0[1], a0[2]))
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    q = _qmul(np.array([cp, 0, sp, 0]), np.array([cr, sr, 0, 0]))  # Ry(pitch) Rx(roll)
    Rs = np.empty((len(t), 3, 3))
    integ = np.zeros(3)
    for i in range(len(t)):
        _, R = _qrot(q, np.zeros(3))
        Rs[i] = R
        if i == len(t) - 1:
            break
        dt = t[i + 1] - t[i]
        w = g[i].copy()
        an = np.linalg.norm(a[i])
        if abs(an - GRAV) < 0.2 * GRAV:  # trust gravity only when not accelerating hard
            v = R.T @ np.array([0, 0, 1.0])
            e = np.cross(a[i] / an, v)
            integ += ki * e * dt
            w += kp * e + integ
        th = np.linalg.norm(w) * dt
        if th > 0:
            ax = w / np.linalg.norm(w)
            q = _qmul(q, np.array([np.cos(th / 2), *(np.sin(th / 2) * ax)]))
            q /= np.linalg.norm(q)
    return Rs


def to_world_features(imu, Rs):
    """RoNIN features: gyro and gravity-included accel rotated into the gravity-aligned frame."""
    gw = np.einsum("nij,nj->ni", Rs, imu[:, 1:4])
    aw = np.einsum("nij,nj->ni", Rs, imu[:, 4:7])
    return np.concatenate([gw, aw], axis=1).astype(np.float32)


# ---------------------------------------------------------------- RoNIN ResNet1D
def build_ronin_resnet(window=200):
    import torch.nn as nn

    def conv3(i, o, stride=1, k=3):
        return nn.Conv1d(i, o, kernel_size=k, stride=stride, padding=k // 2, bias=False)

    class BasicBlock1D(nn.Module):
        expansion = 1

        def __init__(self, i, o, stride=1, downsample=None, kernel_size=3):
            super().__init__()
            self.conv1 = conv3(i, o, stride, kernel_size)
            self.bn1 = nn.BatchNorm1d(o)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3(o, o, 1, kernel_size)
            self.bn2 = nn.BatchNorm1d(o)
            self.downsample = downsample

        def forward(self, x):
            idt = x if self.downsample is None else self.downsample(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + idt)

    class FCOutputModule(nn.Module):
        def __init__(self, in_planes, num_outputs, fc_dim=512, dropout=0.5, in_dim=7, trans_planes=128):
            super().__init__()
            self.transition = nn.Sequential(nn.Conv1d(in_planes, trans_planes, kernel_size=1, bias=False), nn.BatchNorm1d(trans_planes))
            self.fc = nn.Sequential(nn.Linear(trans_planes * in_dim, fc_dim), nn.ReLU(True), nn.Dropout(dropout),
                                    nn.Linear(fc_dim, fc_dim), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(fc_dim, num_outputs))

        def forward(self, x):
            x = self.transition(x)
            return self.fc(x.view(x.size(0), -1))

    class ResNet1D(nn.Module):
        def __init__(self, num_inputs=6, num_outputs=2, group_sizes=(2, 2, 2, 2), base_plane=64, kernel_size=3, in_dim=7):
            super().__init__()
            self.input_block = nn.Sequential(nn.Conv1d(num_inputs, base_plane, kernel_size=7, stride=2, padding=3, bias=False),
                                             nn.BatchNorm1d(base_plane), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
            self.inplanes = base_plane
            groups = []
            for gi, n in enumerate(group_sizes):
                planes, stride = base_plane * 2 ** gi, 1 if gi == 0 else 2
                ds = None
                if stride != 1 or self.inplanes != planes:
                    ds = nn.Sequential(nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(planes))
                blocks = [BasicBlock1D(self.inplanes, planes, stride, ds, kernel_size)]
                self.inplanes = planes
                blocks += [BasicBlock1D(planes, planes, kernel_size=kernel_size) for _ in range(1, n)]
                groups.append(nn.Sequential(*blocks))
            self.residual_groups = nn.Sequential(*groups)
            self.output_block = FCOutputModule(self.inplanes, num_outputs, in_dim=in_dim)

        def forward(self, x):
            return self.output_block(self.residual_groups(self.input_block(x)))

    return ResNet1D(in_dim=window // 32 + 1)


def load_ronin(path, device="cpu", window=200):
    import torch
    net = build_ronin_resnet(window)
    ck = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    return net.to(device).eval()


# ---------------------------------------------------------------- inference + odometry
class InertialOdometry:
    """Horizontal velocity samples (t, vx, vy) in the gravity-aligned IMU frame, integrable."""

    def __init__(self, t, v):
        self.t, self.v = np.asarray(t), np.asarray(v)

    def displacement(self, t0, t1, n=50):
        ts = np.linspace(t0, t1, n)
        vx, vy = np.interp(ts, self.t, self.v[:, 0]), np.interp(ts, self.t, self.v[:, 1])
        return np.array([np.trapezoid(vx, ts), np.trapezoid(vy, ts)])

    def path_length(self, t0=None, t1=None):
        t0 = self.t[0] if t0 is None else t0
        t1 = self.t[-1] if t1 is None else t1
        m = (self.t >= t0) & (self.t <= t1)
        sp = np.linalg.norm(self.v[m], axis=1)
        return float(np.trapezoid(sp, self.t[m])) if m.sum() > 1 else 0.0


def run_ronin(net, imu, bg=None, rate=200.0, window=200, step=10, device="cpu", batch=512):
    import torch
    r = resample(imu, rate)
    Rs = gravity_aligned(r, bg)
    feat = to_world_features(r if bg is None else np.concatenate([r[:, :1], r[:, 1:4] - np.asarray(bg)[None], r[:, 4:7]], 1), Rs)
    starts = np.arange(0, len(r) - window, step)
    if not len(starts):
        raise ValueError("IMU recording shorter than one RoNIN window (1 s)")
    X = np.stack([feat[s:s + window].T for s in starts])
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(net(torch.from_numpy(X[i:i + batch]).to(device)).cpu().numpy())
    v = np.concatenate(out)
    tc = r[starts + window // 2, 0]  # each prediction = mean velocity over its 1 s window
    return InertialOdometry(tc, v)
