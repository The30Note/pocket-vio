"""Metric multi-view poses with MapAnything (Meta, 2025), run in overlapping chunks.

API as documented in https://github.com/facebookresearch/map-anything (README):
  model = MapAnything.from_pretrained("facebook/map-anything")
  views: dicts with "img" (H,W,3, 0..255), optional "intrinsics" (3,3),
         "camera_poses" (4,4, OpenCV cam2world), "is_metric_scale" (bool tensor (1,))
  processed = preprocess_inputs(views); preds = model.infer(processed, ...)
  each pred: "camera_poses" (B,4,4), "intrinsics" (B,3,3), "pts3d" (B,H,W,3), "conf", "mask", ...
Phone poses are passed with is_metric_scale=False: they guide the geometry while MapAnything
decides the metric scale itself.
"""
import numpy as np
from PIL import Image


def _np(x):
    try:
        return x.detach().float().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


def chunks_of(n, size, overlap):
    if n <= size:
        return [list(range(n))]
    step = max(1, size - overlap)
    out, s = [], 0
    while True:
        e = min(n, s + size)
        out.append(list(range(s, e)))
        if e == n:
            return out
        s += step


class MapAnythingRunner:
    def __init__(self, model_id="facebook/map-anything", device=None, amp="bf16"):
        import torch
        from mapanything.models import MapAnything
        from mapanything.utils.image import preprocess_inputs
        self.torch, self.preprocess = torch, preprocess_inputs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MapAnything.from_pretrained(model_id).to(self.device)
        self.amp = amp

    def infer(self, images, K=None, poses=None):
        torch = self.torch
        views = []
        for i, im in enumerate(images):
            v = {"img": torch.from_numpy(np.asarray(im, dtype=np.uint8))}
            if K is not None:
                v["intrinsics"] = torch.from_numpy(np.asarray(K[i], dtype=np.float32))
            if poses is not None:
                v["camera_poses"] = torch.from_numpy(np.asarray(poses[i], dtype=np.float32))
                v["is_metric_scale"] = torch.tensor([False])
            views.append(v)
        preds = self.model.infer(self.preprocess(views), memory_efficient_inference=True, use_amp=self.amp is not None,
                                 amp_dtype=self.amp or "fp16", apply_mask=True, mask_edges=True)
        out = []
        for p in preds:
            pts, mask = _np(p["pts3d"])[0], _np(p["mask"])[0].reshape(_np(p["pts3d"])[0].shape[:2]) > 0.5
            img = _np(p["img_no_norm"])[0] if "img_no_norm" in p else None
            out.append({"pose": _np(p["camera_poses"])[0], "K": _np(p["intrinsics"])[0], "pts": pts, "mask": mask, "img": img,
                        "scale": float(_np(p["metric_scaling_factor"]).reshape(-1)[0]) if "metric_scaling_factor" in p else None})
        return out


def run_chunks(runner, bundle, size=32, overlap=8, use_poses=True, use_intrinsics=True, max_side=None, log=print):
    """Returns per-chunk lists of (photo index, cam2world pose) plus sampled colored points."""
    n = len(bundle.photos)
    phone = bundle.phone_poses("kf")
    results = []
    for ci, idx in enumerate(chunks_of(n, size, overlap)):
        imgs, Ks = [], []
        for i in idx:
            im = Image.open(bundle.image_dir / bundle.photos[i].name).convert("RGB")
            if max_side and max(im.size) > max_side:
                s = max_side / max(im.size)
                im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
            imgs.append(np.asarray(im))
            Ks.append(bundle.K(im.size[0], im.size[1]))
        log(f"  MapAnything chunk {ci + 1}: photos {idx[0]}..{idx[-1]} ({len(idx)} views)")
        preds = runner.infer(imgs, Ks if use_intrinsics else None, phone[idx] if use_poses else None)
        pts, cols = [], []
        for p in preds:
            m = p["mask"]
            P = p["pts"][m]
            C = p["img"][m] if p["img"] is not None else np.full((len(P), 3), 200.0)
            sel = np.random.default_rng(ci).choice(len(P), size=min(len(P), 4000), replace=False) if len(P) else []
            pts.append(P[sel]); cols.append(C[sel])
        results.append({"idx": idx, "poses": np.stack([p["pose"] for p in preds]),
                        "pts": np.concatenate(pts) if pts else np.zeros((0, 3)), "cols": np.concatenate(cols) if cols else np.zeros((0, 3))})
    return results
