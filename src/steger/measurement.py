"""Neck bending angle from two explicitly specified arclength windows."""
from dataclasses import asdict, dataclass

import numpy as np

from .types import Contour


@dataclass(frozen=True)
class BendingMeasurement:
    angle_deg: float
    direction_start: tuple[float,float]
    direction_end: tuple[float,float]
    start_window: tuple[float,float]
    end_window: tuple[float,float]
    start_fit_rms_px: float
    end_fit_rms_px: float
    arc_length_px: float
    cumulative_turn_deg: float
    method: str = "arclength-resampled orthogonal line fits; straight = 0 degrees"

    def to_dict(self):return asdict(self)


def _fit(points):
    center=points.mean(axis=0)
    _,_,vt=np.linalg.svd(points-center,full_matrices=False)
    direction=vt[0]
    if np.dot(direction,points[-1]-points[0])<0:direction=-direction
    normal=np.array([direction[1],-direction[0]])
    rms=float(np.sqrt(np.mean(((points-center)@normal)**2)))
    return direction,rms


def measure_bending(contour: Contour, *, window_fraction=0.2, trim_fraction=0.05,
                     min_window_length=3.0, sample_spacing=0.25):
    """Fit end segments; no hidden smoothing or change to the source contour.

    Reported angle depends on the explicit windows, not the mathematical
    endpoint tangents. For S-curves inspect cumulative_turn_deg as well.
    """
    if contour.closed:raise ValueError("Neck must be an open contour")
    if not 0<window_fraction<0.5 or not 0<=trim_fraction<0.5-window_fraction:
        raise ValueError("Require nonoverlapping trimmed end windows")
    if sample_spacing<=0 or min_window_length<=0:raise ValueError("Lengths must be positive")
    p=contour.points
    completed=contour.attributes.get("junction_completed",np.zeros(len(p),dtype=bool))
    if completed.any():raise ValueError("Contour contains inferred junction points; select measured neck segment first")
    if len(p)<6:raise ValueError("Too few centerline points for reliable two-segment measurement")
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    keep=np.r_[True,np.diff(s)>1e-9];p,s=p[keep],s[keep]
    length=float(s[-1])
    if window_fraction*length<min_window_length:
        raise ValueError("Neck is too short for requested fit windows; do not silently report a precise angle")
    windows=((trim_fraction*length,(trim_fraction+window_fraction)*length),
             ((1-trim_fraction-window_fraction)*length,(1-trim_fraction)*length))
    directions=[];rms=[]
    for lo,hi in windows:
        q=np.linspace(lo,hi,max(5,int(np.ceil((hi-lo)/sample_spacing))+1))
        pts=np.column_stack([np.interp(q,s,p[:,k]) for k in range(2)])
        v,r=_fit(pts);directions.append(v);rms.append(r)
    v1,v2=directions
    angle=float(np.rad2deg(np.arctan2(abs(v1[0]*v2[1]-v1[1]*v2[0]),np.dot(v1,v2))))
    # Cumulative turn at a deliberately coarser 1 px scale; never sum
    # direction differences of adjacent subpixel samples (noise explodes).
    q=np.linspace(windows[0][0],windows[1][1],max(4,int(length)))
    pts=np.column_stack([np.interp(q,s,p[:,k]) for k in range(2)])
    tangents=np.gradient(pts,axis=0)
    tangent_angles=np.unwrap(np.arctan2(tangents[:,0],tangents[:,1]))
    cumulative=float(np.rad2deg(np.abs(np.diff(tangent_angles)).sum()))
    return BendingMeasurement(angle,tuple(v1),tuple(v2),windows[0],windows[1],rms[0],rms[1],length,cumulative)
