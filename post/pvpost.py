#!/usr/bin/env python3
"""Pocket VIO post-processing.

  python pvpost.py session.zip -o out/ [--ronin ronin_resnet.pt] [--refine] [--scale-source auto]

Produces metric camera poses for the photos in a Pocket VIO session bundle:
  out/sparse/0/{cameras,images,points3D}.txt  COLMAP text model (LichtFeld-Studio, gsplat, COLMAP)
  out/transforms.json                         nerfstudio
  out/poses_tum.txt, out/points.ply, out/report.md / report.json
"""
import argparse

from pvpost.pipeline import process


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="session .zip exported by Pocket VIO (or its extracted folder)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--ronin", help="RoNIN ResNet checkpoint (.pt) for IMU-based metric scale")
    ap.add_argument("--no-mapanything", action="store_true", help="skip MapAnything; refine the phone poses instead")
    ap.add_argument("--ma-model", default="facebook/map-anything", help="or facebook/map-anything-apache (Apache 2.0 weights)")
    ap.add_argument("--chunk", type=int, default=32, help="photos per MapAnything pass")
    ap.add_argument("--overlap", type=int, default=8, help="photos shared between consecutive passes")
    ap.add_argument("--max-side", type=int, default=None, help="downscale photos for MapAnything (e.g. 1024)")
    ap.add_argument("--refine", action="store_true", help="pycolmap SIFT + triangulation + bundle adjustment (keeps the scale)")
    ap.add_argument("--scale-source", default="auto", choices=["auto", "inertial", "mapanything", "phone_vio"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    process(a.bundle, a.out, ronin=a.ronin, use_mapanything=not a.no_mapanything, refine=a.refine, scale_source=a.scale_source,
            chunk=a.chunk, overlap=a.overlap, ma_model=a.ma_model, max_side=a.max_side, device=a.device)


if __name__ == "__main__":
    main()
