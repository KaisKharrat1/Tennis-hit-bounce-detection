import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline


def extract_xy(ball_data):
    frames = sorted(ball_data.keys(), key=lambda x: int(x))
    t = np.array([int(f) for f in frames])
    x = np.array([ball_data[f]["x"] if ball_data[f]["visible"] else np.nan for f in frames], dtype=float)
    y = np.array([ball_data[f]["y"] if ball_data[f]["visible"] else np.nan for f in frames], dtype=float)
    return t, x, y


def spline_interpolate(t, v, max_gap=10):
    v_interp = v.copy()

    valid = ~np.isnan(v)
    if valid.sum() < 4:
        return v_interp
    cs = CubicSpline(t[valid], v[valid])
    nan_idx = np.where(np.isnan(v))[0]
    for i in nan_idx:
        # distance au point valide le plus proche
        left = np.where(valid[:i])[0]
        right = np.where(valid[i+1:])[0]

        if len(left) == 0 or len(right) == 0:
            continue

        gap = (t[i + right[0]] - t[left[-1]])
        if gap <= max_gap:
            v_interp[i] = cs(t[i])

    return v_interp

def smooth(v, window=7, poly=2):
    valid = ~np.isnan(v)
    v_s = v.copy()
    v_s[valid] = savgol_filter(v[valid], window, poly)
    return v_s


def build_spline(t, v):
    valid = ~np.isnan(v)
    return CubicSpline(t[valid], v[valid])

def compute_kinematics(t, x, y):
    spline_x = build_spline(t, x)
    spline_y = build_spline(t, y)

    vx = spline_x(t, 1)   # 1ère dérivée
    vy = spline_y(t, 1)

    ax = spline_x(t, 2)   # 2ème dérivée
    ay = spline_y(t, 2)

    return vx, vy, ax, ay

def temporal_nms(probs, preds, cls, window=2):
    """
    Garde un seul événement (hit ou bounce)
    autour du maximum de probabilité.
    """
    final = preds.copy()
    idxs = np.where(preds == cls)[0]

    for i in idxs:
        left = max(0, i - window)
        right = min(len(preds), i + window + 1)

        if probs[i, cls] < np.max(probs[left:right, cls]):
            final[i] = 0  # air

    return final
