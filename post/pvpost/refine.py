"""Optional photogrammetric refinement with pycolmap: SIFT + sequential matching, triangulation
from the metric poses, bundle adjustment, then a Sim(3) back onto the metric poses so the
refinement can improve consistency but never changes the chosen scale or gravity frame."""
from pathlib import Path

import numpy as np

from .export import write_colmap_text
from .geom import apply_sim3, make_T, umeyama


def refine(image_dir, names, K, size, poses, workdir, overlap=10, log=print):
    import pycolmap
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    if db.exists():
        db.unlink()
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "PINHOLE"
    ro.camera_params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"
    log("  pycolmap: extracting SIFT features")
    pycolmap.extract_features(db, image_dir, image_names=list(names), camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro)
    po = pycolmap.SequentialPairingOptions()
    po.overlap = overlap
    log("  pycolmap: sequential matching")
    pycolmap.match_sequential(db, pairing_options=po)
    with pycolmap.Database.open(db) as d:
        ids = {im.name: im.image_id for im in d.read_all_images()}
        cam_id = d.read_all_cameras()[0].camera_id
    seed = work / "seed"
    write_colmap_text(seed, K, size, names, poses, image_ids=ids, camera_id=cam_id)
    rec = pycolmap.Reconstruction(seed)
    log("  pycolmap: triangulating from known poses")
    tri = work / "triangulated"
    tri.mkdir(exist_ok=True)
    rec = pycolmap.triangulate_points(rec, db, image_dir, tri, clear_points=True)
    n_tri = rec.num_points3D()
    log(f"  pycolmap: {n_tri} points; bundle adjustment")
    pycolmap.bundle_adjustment(rec)
    # read refined poses back, then pull them onto the metric input frame with one Sim(3)
    ref = {}
    for im in rec.images.values():
        cfw = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world
        Rcw, tcw = np.asarray(cfw.rotation.matrix()), np.asarray(cfw.translation)
        ref[im.name] = make_T(Rcw.T, -Rcw.T @ tcw)
    keep = [i for i, n in enumerate(names) if n in ref]
    s, R, t = umeyama(np.stack([ref[names[i]][:3, 3] for i in keep]), poses[keep, :3, 3], True)
    out = poses.copy()
    for i in keep:
        out[i] = apply_sim3(ref[names[i]], s, R, t)
    pts = np.array([p.xyz for p in rec.points3D.values()]) if n_tri else np.zeros((0, 3))
    cols = np.array([p.color for p in rec.points3D.values()]) if n_tri else np.zeros((0, 3))
    pts = (s * (R @ pts.T).T + t) if n_tri else pts
    return out, pts, cols, {"points": n_tri, "registered": len(keep), "sim3_scale": float(s), "mean_reproj_error": float(rec.compute_mean_reprojection_error())}
