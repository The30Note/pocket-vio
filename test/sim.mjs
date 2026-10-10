// Pocket VIO offline verification.
//   node test/sim.mjs            (runs every test against ../index.html)
//   node test/sim.mjs vio --seconds 40 --seed 3
// Extracts the <script id="vio-core"> block from index.html and runs it in Node on
// synthetic data: a KLT unit test, a calibration recovery test and the full
// visual-inertial estimator on a noisy 200 Hz IMU + pixel-noise trajectory.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? Number(args[i + 1]) : d; };
const which = args.filter(a => !a.startsWith('--') && isNaN(Number(a)));
const srcPath = process.env.VIO_SRC || path.join(here, '..', 'index.html');
let code = fs.readFileSync(srcPath, 'utf8');
const m = code.match(/<script id="vio-core">([\s\S]*?)<\/script>/);
if (m) code = m[1];
const VIO = new Function(code + '\nreturn VIO;')();
const { mm, mtm, mv, mtv, expSO3, logSO3, sub3, add3, scl3, norm3, makeCam } = VIO;

function rng(seed) {
  let s = seed >>> 0;
  const u = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const n = () => { let a = 0; while (a === 0) a = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * u()); };
  return { u, n, n3: (s) => [n() * s, n() * s, n() * s] };
}
let failures = 0;
const check = (name, ok, msg) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${msg}`); if (!ok) failures++; };

// --------------------------------------------------------------- KLT test
function testKLT() {
  const W = 270, H = 360, R = rng(7);
  const waves = Array.from({ length: 40 }, () => ({ fx: (R.u() - 0.5) * 0.6, fy: (R.u() - 0.5) * 0.6, ph: R.u() * 6.28, a: 10 + 20 * R.u() }));
  const tex = (x, y) => { let s = 128; for (const w of waves) s += w.a * Math.sin(w.fx * x + w.fy * y + w.ph) / 4; return s; };
  const img = (dx, dy) => { const g = new Float32Array(W * H); for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) g[y * W + x] = tex(x - dx, y - dy); return g; };
  for (const [dx, dy, pred] of [[2.3, -1.6, false], [9.4, 6.2, false], [24.5, -17.3, true]]) {
    const trk = new VIO.Tracker({});
    const a = trk.process(img(0, 0), W, H, null).map(q => ({ ...q }));
    const b = trk.process(img(dx, dy), W, H, pred ? (x, y) => [x + dx + 1.5, y + dy - 1.2] : null);
    const byId = new Map(a.map(q => [q.id, q]));
    const errs = b.filter(q => byId.has(q.id)).map(q => Math.hypot(q.x - byId.get(q.id).x - dx, q.y - byId.get(q.id).y - dy));
    const med = VIO.median(errs), m = 14; // points that stay inside the trackable area
    const inside = a.filter(q => q.x + dx > m && q.y + dy > m && q.x + dx < W - m && q.y + dy < H - m).length;
    check(`KLT shift (${dx},${dy})${pred ? ' +gyro prediction' : ''}`, a.length >= 100 && errs.length > 0.7 * inside && med < 0.1,
      `detected=${a.length} inside=${inside} tracked=${errs.length} medianErr=${med.toFixed(3)}px`);
  }
}

// ------------------------------------------------------------ trajectory
const env = u => u <= 0 ? 0 : u * u / (u * u + 1);
const R0 = [1, 0, 0, 0, 0, -1, 0, 1, 0]; // portrait, back camera looking at world +y
function makeMotion(tStatic, amp = 1) {
  const pos = t => { const u = t - tStatic, e = env(u) * amp;
    return [e * 0.9 * Math.sin(0.55 * u), e * (0.5 * Math.sin(0.8 * u) + 0.15 * Math.sin(2.1 * u)), e * 0.25 * Math.sin(0.9 * u + 0.3 * Math.sin(1.7 * u))]; };
  const rot = t => { const u = t - tStatic, e = env(u);
    return mm(expSO3([e * 0.25 * Math.sin(0.6 * u), e * 0.3 * Math.sin(0.45 * u), e * (0.35 * Math.sin(0.5 * u) + 0.1 * Math.sin(1.9 * u))]), R0); };
  const h = 1e-4;
  const acc = t => scl3(add3(sub3(pos(t + h), scl3(pos(t), 2)), pos(t - h)), 1 / (h * h));
  const vel = t => scl3(sub3(pos(t + h), pos(t - h)), 0.5 / h);
  const omega = t => scl3(logSO3(mtm(rot(t - h), rot(t + h))), 0.5 / h);
  return { pos, rot, acc, vel, omega };
}

function umeyama(A, B) { // find s,R,t : B ≈ s R A + t
  const n = A.length, ma = [0, 0, 0], mb = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { ma[k] += A[i][k] / n; mb[k] += B[i][k] / n; }
  const S = new Array(9).fill(0); let va = 0;
  for (let i = 0; i < n; i++) { const a = sub3(A[i], ma), b = sub3(B[i], mb); va += VIO.dot3(a, a) / n;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[3 * r + c] += b[r] * a[c] / n; }
  // polar decomposition of S via iteration (R = closest rotation)
  let Rm = S.slice();
  for (let it = 0; it < 60; it++) { const inv = inv3(tr(Rm)); Rm = Rm.map((v, i) => 0.5 * (v + inv[i])); }
  let trc = 0; for (let i = 0; i < 9; i++) trc += Rm[i] * S[i];
  const s = trc / va, t = sub3(mb, scl3(mv(Rm, ma), s));
  return { s, R: Rm, t };
}
const tr = A => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
function inv3(A) {
  const [a, b, c, d, e, f, g, h, i] = A, det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  return [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det, (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det];
}

// ------------------------------------------------------------ VIO test
function testVIO({ seconds = 30, seed = 1, da3 = false, metricBias = 1, td = 0, label = 'VIO', scanAt = 0, initBgErr = null } = {}) {
  const R = rng(seed), tStatic = 2.0, mo = makeMotion(tStatic);
  const W = 270, H = 360, cam = makeCam(W, H, 65, 0);
  const bgT = [0.012, -0.018, 0.009], baT = [0.06, -0.08, 0.1];
  const est = new VIO.Estimator({ td });
  est.setCamera(cam);
  if (initBgErr) est.p.initBg = add3(bgT, initBgErr); // gyro bias from an earlier rest check, slightly stale
  let scanning = false;
  // landmarks on the walls of a room
  const LM = [];
  const box = [[-5, 5], [-5, 7], [-1.6, 2.4]];
  for (let i = 0; i < 5000; i++) {
    const ax = Math.floor(R.u() * 3), side = R.u() < 0.5 ? 0 : 1, X = [0, 0, 0];
    for (let k = 0; k < 3; k++) X[k] = box[k][0] + R.u() * (box[k][1] - box[k][0]);
    X[ax] = box[ax][side];
    LM.push(X);
  }
  const imuRate = 200, gyroN = 0.01, accN = 0.08; // per-sample std
  let tImu = 0, bgD = bgT.slice(), baD = baT.slice();
  const pushImuUntil = (t) => {
    while (tImu <= t) {
      const Rw = mo.rot(tImu), aw = mo.acc(tImu);
      bgD = add3(bgD, R.n3(1e-5)); baD = add3(baD, R.n3(1e-4));
      const g = add3(add3(mo.omega(tImu), bgD), R.n3(gyroN));
      const a = add3(add3(mtv(Rw, sub3(aw, [0, 0, -9.81])), baD), R.n3(accN));
      est.addImu(tImu, g, a);
      tImu += 1 / imuRate;
    }
  };
  const active = new Map(); let nextId = 1;
  const project = (X, Rw, p) => { const Xc = mtv(cam.Rbc, mtv(Rw, sub3(X, p))); if (Xc[2] < 0.3) return null; const q = cam.project(Xc);
    return (q[0] < 10 || q[1] < 10 || q[0] > W - 10 || q[1] > H - 10) ? null : { q, Z: Xc[2] }; };
  const errLive = [], gtAt = t => ({ p: mo.pos(t), R: mo.rot(t) });
  let maxSolve = 0, sumSolve = 0, nKf = 0, kills = 0, resets = 0;
  const startResets = est.resets + 1; // + startScan
  for (let k = 0; k * (1 / 30) < seconds; k++) {
    const t = k / 30, tc = t - td; // camera clock lags IMU clock by td
    pushImuUntil(t + 0.02);
    const Rw = mo.rot(t), p = mo.pos(t);
    const vis = new Map();
    for (let i = 0; i < LM.length; i++) { const r = project(LM[i], Rw, p); if (r) vis.set(i, r); }
    for (const [li, a] of active) if (!vis.has(li) || R.u() < 0.004) active.delete(li);
    const cand = [...vis.keys()].filter(i => !active.has(i));
    for (let i = cand.length - 1; i > 0; i--) { const j = Math.floor(R.u() * (i + 1)); [cand[i], cand[j]] = [cand[j], cand[i]]; }
    const placed = [...active.keys()].map(i => vis.get(i).q);
    for (const li of cand) {
      if (active.size >= 140) break;
      const q = vis.get(li).q;
      if (placed.some(o => Math.hypot(o[0] - q[0], o[1] - q[1]) < 12)) continue;
      placed.push(q);
      const bad = R.u() < 0.03 ? [R.n() * 0.3, R.n() * 0.3] : null;
      active.set(li, { id: nextId++, bad, off: [0, 0] });
    }
    const tracks = [];
    for (const [li, a] of active) {
      const q = vis.get(li).q;
      if (a.bad) { a.off[0] += a.bad[0]; a.off[1] += a.bad[1]; }
      tracks.push({ id: a.id, x: q[0] + R.n() * 0.5 + a.off[0], y: q[1] + R.n() * 0.5 + a.off[1], li });
    }
    if (!scanning && t >= scanAt) { est.startScan(); scanning = true; }
    const out = est.addFrame(tc, tracks);
    if (out.kf && est.status === 'tracking' && !est._logged) { est._logged = true;
      const k0 = est.kfs[0], vt = mo.vel(k0.t), Rt = mo.rot(k0.t);
      const tilt = Math.acos(Math.min(1, VIO.dot3(mtv(k0.R, [0, 0, 1]), mtv(Rt, [0, 0, 1])))) * 180 / Math.PI;
      // velocity in body frame (yaw-free comparison)
      console.log(`  init ${est.initMode} at t=${k0.t.toFixed(2)} kfs=${est.kfs.length} align=${JSON.stringify(est.alignInfo)} |v|true=${norm3(vt).toFixed(3)} vBody est=[${mtv(k0.R, k0.v).map(v => v.toFixed(2))}] true=[${mtv(Rt, vt).map(v => v.toFixed(2))}] tiltErr=${tilt.toFixed(2)}°`);
    }
    if (out.reset) { resets++; console.log(`  reset at t=${t.toFixed(2)}: ${out.reset}`); }
    if (out.kill.length) { kills += out.kill.length; const ks = new Set(out.kill); for (const [li, a] of active) if (ks.has(a.id)) active.delete(li); }
    if (out.kf && est.status === 'tracking') {
      nKf++; maxSolve = Math.max(maxSolve, est.solveMs); sumSolve += est.solveMs;
      if (da3) { // synthetic "DA3": inverse depth with unknown scale + 10% noise
        const dw = 64, dh = 86, data = new Float32Array(dw * dh), kf = est.kfs[est.kfs.length - 1];
        // render depth by splatting the true landmarks (nearest), fill holes with neighbours
        const zb = new Float32Array(dw * dh).fill(0);
        for (const X of LM) { const r = project(X, Rw, p); if (!r) continue;
          const x = Math.floor(r.q[0] * dw / W), y = Math.floor(r.q[1] * dh / H);
          for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) { const xx = x + i, yy = y + j; if (xx < 0 || yy < 0 || xx >= dw || yy >= dh) continue;
            const o = yy * dw + xx; if (zb[o] === 0 || r.Z < zb[o]) zb[o] = r.Z; } }
        const metric = da3 === 'metric' || da3 === 'monitor', monitorOnly = da3 === 'monitor';
        for (let i = 0; i < zb.length; i++) data[i] = zb[i] > 0 ? (metric ? metricBias * zb[i] : 3.7 / zb[i]) * (1 + 0.1 * R.n()) : 0;
        est.addDepthMap(kf.id, data, dw, dh, { metric, monitorOnly });
      }
    }
    if (est.status === 'tracking' && est.pose) errLive.push(norm3(sub3(est.pose.p, mo.pos(est.pose.t))));
  }
  // ---- evaluation on the (smoothed) keyframe trajectory
  const traj = est.traj;
  const E = traj.map(e => e.p), G = traj.map(e => gtAt(e.t).p);
  const rms = a => Math.sqrt(a.reduce((s, x) => s + x * x, 0) / Math.max(1, a.length));
  const ateRaw = rms(E.map((e, i) => norm3(sub3(e, G[i]))));
  const al = umeyama(E, G);
  const ateSim3 = rms(E.map((e, i) => norm3(sub3(add3(scl3(mv(al.R, e), al.s), al.t), G[i]))));
  // rigid (scale-free) alignment: the honest error measure when a scan starts mid-motion
  const mE = scl3(E.reduce((a, b) => add3(a, b), [0, 0, 0]), 1 / E.length), mG = scl3(G.reduce((a, b) => add3(a, b), [0, 0, 0]), 1 / G.length);
  const tR = sub3(mG, mv(al.R, mE));
  const ateSE3 = rms(E.map((e, i) => norm3(sub3(add3(mv(al.R, e), tR), G[i]))));
  let pathGT = 0; for (let i = 1; i < G.length; i++) pathGT += norm3(sub3(G[i], G[i - 1]));
  const yawErr = Math.abs(logSO3(mtm(traj[traj.length - 1].R, gtAt(traj[traj.length - 1].t).R)).reduce((s, x) => s + x * x, 0)) ** 0.5 * 180 / Math.PI;
  const kfLast = est.kfs[est.kfs.length - 1] || { bg: [NaN, NaN, NaN], ba: [NaN, NaN, NaN] };
  console.log(`  ${label}: ${seconds}s, KFs=${traj.length}, path(GT)=${pathGT.toFixed(2)} m, landmarks(window)=${est.nLm}, mapPts=${est.mapPts.length}, kills=${kills}`);
  console.log(`  solve: mean ${(sumSolve / Math.max(1, nKf)).toFixed(1)} ms, max ${maxSolve.toFixed(1)} ms, LM iters(last)=${est.lastIters}`);
  console.log(`  scale(sim3)=${al.s.toFixed(4)}  ATE raw=${ateRaw.toFixed(3)} m  ATE sim3=${ateSim3.toFixed(3)} m  ATE se3=${ateSE3.toFixed(3)} m  final pos err=${norm3(sub3(E[E.length - 1], G[G.length - 1])).toFixed(3)} m  final rot err=${yawErr.toFixed(2)}°`);
  console.log(`  live (IMU-propagated) error: rms ${rms(errLive).toFixed(3)} m, max ${Math.max(...errLive).toFixed(3)} m`);
  console.log(`  bias est bg=[${kfLast.bg.map(v => v.toFixed(4))}] true≈[${bgD.map(v => v.toFixed(4))}]  ba=[${kfLast.ba.map(v => v.toFixed(3))}] true≈[${baD.map(v => v.toFixed(3))}]`);
  if (da3) console.log(`  DA3 prior: mode=${est.da3.mode} scale=${est.da3.scale.toFixed(3)} relErr=${est.da3.err.toFixed(3)} maps=${est.da3.maps} VIO/DA3=${est.da3.vioRatio}`);
  check(`${label} no resets`, resets === 0 && est.resets === startResets, `resets=${resets}`);
  if (metricBias === 1) check(`${label} metric scale ≈ 1`, Math.abs(al.s - 1) < 0.05, `scale=${al.s.toFixed(4)}`);
  else console.log(`  INFO  ${label}: DA3 metric bias ×${metricBias} -> trajectory scale ${(1 / al.s).toFixed(4)} (1 = unaffected)`);
  const ate = scanAt > 0 ? ateSE3 : ateRaw, ateName = scanAt > 0 ? 'ATE (rigid-aligned)' : 'ATE raw';
  check(`${label} trajectory error small`, ate < Math.max(0.1, 0.03 * pathGT), `${ateName}=${ate.toFixed(3)} m over ${pathGT.toFixed(1)} m (${(100 * ate / pathGT).toFixed(2)}%)`);
  if (da3 === true && scanAt === 0) check(`${label} depth mode detected`, est.da3.mode === 'inverse depth', est.da3.mode);
}

// ------------------------------------------------------ calibration test
function testCalib() {
  const R = rng(11), W = 270, H = 360, trueOff = 90, trueTd = 0.034, trueFov = 71;
  const cam = makeCam(W, H, trueFov, trueOff);
  const imu = new VIO.ImuBuf();
  const rot = t => mm(expSO3([0.8 * Math.sin(2.1 * t), 0.7 * Math.sin(1.7 * t + 1), 0.5 * Math.sin(2.9 * t + 2)]), R0);
  for (let t = 0; t < 6; t += 0.005) { const w = scl3(logSO3(mtm(rot(t - 1e-4), rot(t + 1e-4))), 0.5e4); imu.push(t, add3(w, R.n3(0.01)), [0, 9.81, 0]); }
  const pts = Array.from({ length: 3000 }, () => { const d = [R.n(), R.n(), R.n()]; return scl3(d, (2 + 4 * R.u()) / norm3(d)); });
  const pairs = []; let prev = null;
  for (let t = 0.3; t < 5.3; t += 1 / 30) {
    const tc = t - trueTd, Rw = rot(t), obs = new Map();
    pts.forEach((X, i) => { const Xc = mtv(cam.Rbc, mtv(Rw, X)); if (Xc[2] < 0.5) return; const q = cam.project(Xc);
      if (q[0] > 5 && q[1] > 5 && q[0] < W - 5 && q[1] < H - 5 && i % 20 === 0) obs.set(i, [q[0] + 0.5 * R.n(), q[1] + 0.5 * R.n()]); });
    if (prev) { const a = [], b = []; for (const [i, q] of obs) if (prev.obs.has(i)) { a.push(...prev.obs.get(i)); b.push(...q); } pairs.push({ t0: prev.tc, t1: tc, a, b }); }
    prev = { tc, obs };
  }
  const t0 = Date.now();
  const r = VIO.calibrate(pairs, imu, { W, H, screenAngle: 0 });
  check('calibration recovers offset/td/FOV', r && r.offset === trueOff && Math.abs(r.td - trueTd) <= 0.003 && Math.abs(r.fov - trueFov) < 1.0,
    r ? `offset=${r.offset} td=${(r.td * 1000).toFixed(1)}ms fov=${r.fov.toFixed(2)}° rms=${r.rms.toFixed(2)}px (uncompensated ${r.rmsNoComp.toFixed(1)}px) in ${Date.now() - t0} ms` : 'null');
}

const run = n => which.length === 0 || which.includes(n);
if (which.includes('sweep')) { // moving starts with DA3 relative, several seeds / start times
  for (const [seed, at] of [[8, 4], [9, 7.5], [10, 10], [11, 12.5], [12, 15], [13, 5.5]])
    testVIO({ seconds: at + 14, seed, scanAt: at, da3: true, initBgErr: [0.002, -0.003, 0.002], label: `sweep seed${seed} start@${at}s` });
}
if (run('klt')) testKLT();
if (run('calib')) testCalib();
if (run('vio')) testVIO({ seconds: opt('seconds', 30), seed: opt('seed', 1) });
if (run('td')) testVIO({ seconds: 20, seed: 2, td: 0.025, label: 'VIO (td=25ms)' });
if (run('da3')) testVIO({ seconds: 20, seed: 4, da3: true, label: 'VIO + DA3 prior' });
if (run('moving')) testVIO({ seconds: 30, seed: 5, scanAt: 8, initBgErr: [0.003, -0.002, 0.002], label: 'VIO started while moving' });
if (run('moving7')) testVIO({ seconds: 25, seed: 7, scanAt: 6, initBgErr: [0.002, 0.002, -0.003], label: 'seed7 no DA3' });
if (run('movingda3')) testVIO({ seconds: 25, seed: 7, scanAt: 6, da3: true, initBgErr: [0.002, 0.002, -0.003], label: 'VIO + DA3 started while moving' });
if (run('moving2')) testVIO({ seconds: 30, seed: 6, scanAt: 13.5, initBgErr: [-0.002, 0.003, 0.001], label: 'VIO started while moving (2)' });
if (run('metric')) testVIO({ seconds: 20, seed: 4, da3: 'metric', label: 'VIO + DA3 metric' });
if (run('monitor')) testVIO({ seconds: 20, seed: 4, da3: 'monitor', metricBias: 1.1, label: 'VIO + DA3 metric-monitor (10% biased)' });
if (run('metricbias')) testVIO({ seconds: 20, seed: 4, da3: 'metric', metricBias: 1.1, label: 'VIO + DA3 metric (10% biased)' });
console.log(failures ? `\n${failures} check(s) FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
