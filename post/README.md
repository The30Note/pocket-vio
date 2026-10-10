# pvpost — metric poses from a Pocket VIO session

Pocket VIO on the phone records a **session bundle** (Export → *Session bundle (.zip)*):
photos, the raw IMU stream, the phone's rough poses and the camera model. `pvpost` turns it
into metric camera poses you can refine further (COLMAP, LichtFeld-Studio, gsplat, nerfstudio).

```
python pvpost.py pocketvio_<id>.zip -o out/ --ronin ronin_resnet/checkpoint_gsn_latest.pt --refine
```

## What it does

1. **Vision trajectory** — [MapAnything](https://github.com/facebookresearch/map-anything) runs on
   overlapping chunks of photos (default 32 views, 8 shared), with the photo intrinsics and the
   phone poses as hints (`is_metric_scale=False`, so MapAnything sets the metric scale itself).
   Chunks are chained with a Sim(3) on the shared photos and rotated into the phone's gravity frame
   (z up). Without MapAnything (`--no-mapanything` or not installed) the phone poses are used.
2. **Scale** — candidates, each a factor on the vision trajectory:
   - `inertial`: [RoNIN](https://github.com/Sachini/ronin) ResNet inertial odometry on the IMU alone
     (gravity-aligned with a Mahony filter, 200 Hz, 1 s windows). Its horizontal displacement is
     compared with the vision displacement between photos 2–8 s apart; the median ratio is the scale.
     Heading drift in the IMU frame does not matter (only distances are compared).
   - `mapanything`: MapAnything's own metric scale.
   - `phone_vio`: whatever the phone estimated.
   `--scale-source auto` uses inertial if available, else MapAnything, else the phone.
3. **Refinement** (`--refine`) — pycolmap SIFT, sequential matching, triangulation from the poses,
   bundle adjustment, then one Sim(3) back onto the metric poses so the chosen scale and gravity
   frame are kept.
4. **Outputs** in `out/`: `sparse/0/` COLMAP text model + `images/` (load directly in
   LichtFeld-Studio / gsplat / COLMAP), `transforms.json` (nerfstudio), `poses_tum.txt`,
   `points.ply`, `report.md` / `report.json` (all scale candidates side by side, plus the walk tests
   logged on the phone).

## Setup (Arch, RTX 3060)

```
python -m venv ~/venvs/pvpost && . ~/venvs/pvpost/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
git clone https://github.com/facebookresearch/map-anything && pip install -e ./map-anything
```

RoNIN weights: download the pretrained models from <https://doi.org/10.20383/102.0543>
(the RoNIN dataset page) and pass the ResNet checkpoint with `--ronin`. The network is
re-implemented here so the published `model_state_dict` loads unchanged.

Licences: MapAnything code is Apache 2.0; the default weights `facebook/map-anything` are
CC-BY-NC 4.0, use `--ma-model facebook/map-anything-apache` for Apache 2.0 weights. RoNIN is GPL-3.0.

## Check that it works

`python tests/test_pipeline.py` renders a synthetic corridor walk (photos ray-cast from known
poses, 200 Hz IMU with gait sway, phone poses with a 20 % scale error) and checks every stage
against ground truth. MapAnything and RoNIN are replaced by a mock / an oracle with realistic
errors there, because their weights can't be bundled; the RoNIN network itself is checked for
checkpoint layout and inference shapes.

## Getting good scale

- Calibrate on the phone first (setup step 4) and walk normally; RoNIN was trained on walking.
- Hold the phone roughly steady relative to your body; photos every ~0.3 m are plenty.
- Compare the `report.md` scale candidates. If `inertial` and `mapanything` agree within a few
  percent, trust them; log a measured walk (Walk scale test) to see which source is right on your phone.
