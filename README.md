# Pocket VIO

Monocular visual-inertial odometry in one HTML file (`index.html`), aimed at Chrome on Android.

- **Run:** open https://raw.githack.com/The30Note/pocket-vio/main/index.html (or serve this folder over HTTPS) or `http://localhost`, open it on the phone and tap **Start**. Then hold the phone still for about 1 s.
- **Calibrate:** tap **Calibrate** and rotate the phone in place for 5 s. This solves the camera rotation offset, the camera-IMU time offset and the FOV, and saves them to localStorage.
- **DA3 depth prior:** runs the ONNX model directly on ONNX Runtime Web (WebGPU, falling back to WASM); progress and a load log show under the toggle. Model repo/file/backend/input size are in Settings.
- **Verify offline:** `node test/sim.mjs`. This pulls the `<script id="vio-core">` block out of `index.html` and runs four checks: KLT, calibration recovery, and the full estimator on a synthetic noisy 200 Hz IMU + pixel-noise trajectory (with td and DA3-prior variants). It checks metric scale ≈ 1 and that the trajectory error stays small.
