import math
import rhinoscriptsyntax as rs
import Rhino.Geometry as rg

def build_profile(pts, half):
    pts_sorted = sorted(pts, key=lambda p: p.X)
    if half:
        crown = max(pts_sorted, key=lambda p: p.Z)
        crown_index = pts_sorted.index(crown)
        profile_pts = pts_sorted[crown_index:]
    else:
        profile_pts = pts_sorted
    return rg.Curve.CreateInterpolatedCurve(profile_pts, 3)

def revolve(crv, axis_pt, start_deg, end_deg):
    ap = rg.Point3d(axis_pt.X, axis_pt.Y, axis_pt.Z)
    axis_line = rg.Line(ap, ap + rg.Vector3d(0, 0, 1))
    revsrf = rg.RevSurface.Create(
        crv, axis_line, math.radians(start_deg), math.radians(end_deg)
    )
    return revsrf.ToBrep()

profile_crv = build_profile(pts, half)
dome = revolve(profile_crv, axis_pt, start_deg, end_deg)

a = dome
b = profile_crv