# stereonet_dash.py
import os
import numpy as np
import pandas as pd
from io import StringIO

import base64
import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------
# Layout constants
# -------------------------
FIG_HEIGHT = 700
H_SPACING = 0.02
PLOT_MARGIN = dict(l=20, r=20, t=80, b=80)
FIG_WIDTH = int(
    ((FIG_HEIGHT - (PLOT_MARGIN["t"] + PLOT_MARGIN["b"])) * 2) / (1 - H_SPACING)
    + (PLOT_MARGIN["l"] + PLOT_MARGIN["r"])
)
NET_WIDTH = FIG_WIDTH - (PLOT_MARGIN["l"] + PLOT_MARGIN["r"])
PANEL_WIDTH = int(NET_WIDTH * (1 - H_SPACING) / 2)
LEGEND_COLS = 4
LEGEND_ENTRY_WIDTH = 1 / LEGEND_COLS
LEGEND_ENTRY_WIDTH_MODE = 'fraction'
LEFT_AXIS_COLORS = {
    "P": "#1f77b4",
    "B": "#2ca02c",
    "T": "#ff7f0e",
}
RIGHT_AXIS_COLORS = ['#8ab7e0', '#7fd39a', '#f6c56f']  # 1, 2, 3 aligned to P, B, T
RIGHT_AXIS_RANKS = {1: 30, 2: 70, 3: 110}
NET_MODE_OPTIONS = [
    {"label": "Oriented stereonet", "value": "oriented"},
    {"label": "Classic (Equator View)", "value": "normal"},
]
DEFAULT_NET_MODE = "oriented"
BOTTOM_FIG_HEIGHT = 400
TENSOR_FIG_HEIGHT = BOTTOM_FIG_HEIGHT
PIE_FIG_HEIGHT = BOTTOM_FIG_HEIGHT
PIE_WIDTH = int(FIG_WIDTH * 0.4)
BOTTOM_GAP = 12
TENSOR_WIDTH = FIG_WIDTH - PIE_WIDTH - BOTTOM_GAP
IDENTITY_VIEW_MATRIX = np.eye(3)
DISPLAY_VIEW_MATRIX = IDENTITY_VIEW_MATRIX
ANGLE_CATEGORY_LABELS = ["None/Minor", "Moderate", "Significant", "Access not advisable"]
ANGLE_CATEGORY_COLORS = ["#76c893", "#ffd166", "#f77f00", "#d62828"]
ANGLE_CATEGORY_BINS = [15.0, 30.0, 45.0]

# -------------------------
# Helper geometry functions
# -------------------------
def dipdir_to_pole_vector(dip_dir_deg, dip_deg):
    """
    Given dip direction (azimuth, degrees, 0=N, clockwise) and dip (degrees),
    return a unit normal (pole) vector in (east, north, up) coordinates,
    forced to the lower hemisphere (nz <= 0).
    """
    D = np.radians(dip_dir_deg)
    delta = np.radians(dip_deg)

    # down-dip (unit) vector (points down into plane)
    dx = np.sin(D) * np.cos(delta)
    dy = np.cos(D) * np.cos(delta)
    dz = -np.sin(delta)  # negative downwards if z is up

    # strike azimuth (horizontal) = dip_direction - 90°
    S = D - np.pi/2.0
    sx = np.sin(S); sy = np.cos(S); sz = 0.0

    # pole = cross(strike, down-dip)
    nx = sy*dz - sz*dy
    ny = sz*dx - sx*dz
    nz = sx*dy - sy*dx

    # normalize
    norm = np.sqrt(nx*nx + ny*ny + nz*nz)
    if norm == 0:
        return 0.0, 0.0, -1.0  # fallback
    nx /= norm; ny /= norm; nz /= norm

    # force to lower hemisphere (nz <= 0)
    if nz > 0:
        nx, ny, nz = -nx, -ny, -nz

    return nx, ny, nz

def vector_to_trend_plunge(nx, ny, nz):
    # Trend (azimuth) measured clockwise from North:
    # use atan2(east, north)
    trend = (np.degrees(np.arctan2(nx, ny)) + 360.0) % 360.0
    # Plunge: angle below horizontal; nz is up (negative for lower hemisphere)
    plunge = np.degrees(np.arcsin(-nz))
    return trend, plunge

def equal_area_proj(trend_deg, plunge_deg, rotation_deg=0.0):
    """
    Schmidt equal-area projection of a line/pole with (trend, plunge) in degrees.
    rotation_deg adds a rotation to the trend (positive clockwise).
    Returns x,y coordinates where north=up (y positive).
    """
    alpha = np.radians((trend_deg + rotation_deg) % 360.0)
    p = plunge_deg
    r = np.sqrt(2.0) * np.sin(np.radians((90.0 - p) / 2.0))
    x = r * np.sin(alpha)  # east component -> x
    y = r * np.cos(alpha)  # north component -> y
    return x, y

def dipdir_array_to_xy(dip_dirs, dips, rotation_deg=0.0):
    nx, ny, nz = zip(*(dipdir_to_pole_vector(dd, dp) for dd, dp in zip(dip_dirs, dips)))
    nx = np.array(nx); ny = np.array(ny); nz = np.array(nz)
    trend, plunge = vector_to_trend_plunge(nx, ny, nz)
    x, y = equal_area_proj(trend, plunge, rotation_deg)
    return x, y

def trend_plunge_to_vector(trend_deg, plunge_deg):
    """
    Convert trend/plunge (degrees) to unit vectors (east, north, up).
    Plunge is positive down from horizontal.
    """
    t = np.radians(trend_deg)
    p = np.radians(plunge_deg)
    x = np.sin(t) * np.cos(p)
    y = np.cos(t) * np.cos(p)
    z = -np.sin(p)
    return x, y, z

def rotate_trend_plunge(trend_deg, plunge_deg, rot_matrix):
    x, y, z = trend_plunge_to_vector(trend_deg, plunge_deg)
    v = np.vstack([x, y, z])
    v = rot_matrix @ v
    upper = v[2] > 0
    v[:, upper] *= -1.0
    return vector_to_trend_plunge(v[0], v[1], v[2])

def rotate_vector(v, rot_matrix):
    if v is None:
        return None
    v_rot = rot_matrix @ v
    return force_lower_hemisphere(v_rot)

def axial_mean_direction(trend_deg, plunge_deg):
    """
    Axial mean (directionless) using the orientation matrix eigenvector.
    Returns a unit vector (east, north, up) forced to lower hemisphere (z <= 0).
    """
    x, y, z = trend_plunge_to_vector(trend_deg, plunge_deg)
    v = np.vstack([x, y, z]).T
    if v.size == 0:
        return None
    # orientation matrix
    S = v.T @ v
    vals, vecs = np.linalg.eigh(S)
    mean_vec = vecs[:, np.argmax(vals)]
    # force to lower hemisphere
    if mean_vec[2] > 0:
        mean_vec = -mean_vec
    # normalize
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        return None
    return mean_vec / norm

def directional_mean_direction(trend_deg, plunge_deg):
    """
    Directional mean (signed) using vector averaging.
    Returns a unit vector (east, north, up) forced to lower hemisphere (z <= 0).
    """
    x, y, z = trend_plunge_to_vector(trend_deg, plunge_deg)
    v = np.vstack([x, y, z]).T
    if v.size == 0:
        return None
    mean_vec = v.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        return None
    mean_vec = mean_vec / norm
    if mean_vec[2] > 0:
        mean_vec = -mean_vec
    return mean_vec

def orthonormalize_triad(v1, v2, v3):
    """
    Find the closest orthonormal triad to the three input vectors.
    Returns (v1o, v2o, v3o) as unit vectors.
    """
    M = np.stack([v1, v2, v3], axis=1)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R[:, 0], R[:, 1], R[:, 2]

def orthonormalize_axial_axes(v1, v2, v3):
    """
    Closest orthonormal axes for directionless poles.
    Handedness is irrelevant for axial data, so reflections are allowed.
    """
    M = np.stack([v1, v2, v3], axis=1)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    return R[:, 0], R[:, 1], R[:, 2]

def force_lower_hemisphere(v):
    if v is None:
        return None
    return -v if v[2] > 0 else v

def safe_normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return None
    return v / n

def angle_between(a, b):
    a_n = safe_normalize(a)
    b_n = safe_normalize(b)
    if a_n is None or b_n is None:
        return 0.0
    return float(np.arccos(np.clip(np.dot(a_n, b_n), -1.0, 1.0)))

def signed_angle_about_axis(u, v, axis):
    axis_n = safe_normalize(axis)
    if axis_n is None:
        return 0.0
    u_p = u - np.dot(u, axis_n) * axis_n
    v_p = v - np.dot(v, axis_n) * axis_n
    u_p = safe_normalize(u_p)
    v_p = safe_normalize(v_p)
    if u_p is None or v_p is None:
        return 0.0
    cross = np.cross(u_p, v_p)
    sin = np.dot(axis_n, cross)
    cos = np.dot(u_p, v_p)
    return float(np.arctan2(sin, cos))

def rotate_about_axis(v, axis, angle_rad):
    axis_n = safe_normalize(axis)
    if axis_n is None:
        return v
    K = np.array([[0, -axis_n[2], axis_n[1]],
                  [axis_n[2], 0, -axis_n[0]],
                  [-axis_n[1], axis_n[0], 0]])
    return v + np.sin(angle_rad) * (K @ v) + (1 - np.cos(angle_rad)) * (K @ (K @ v))

def rotation_between_vectors(a, b):
    a = safe_normalize(a)
    b = safe_normalize(b)
    if a is None or b is None:
        return np.eye(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # opposite vectors: pick any orthogonal axis
        ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = safe_normalize(np.cross(a, ref))
        if axis is None:
            return np.eye(3)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + (K @ K) * ((1 - c) / (s ** 2))

def rotation_from_pole_equator(pole_vec, equator_vec):
    """
    Build rotation matrix so that local down axis maps to pole_vec,
    and local x-axis aligns with equator_vec projected to the equator.
    Returns 3x3 matrix with columns = world coords of local (x, y, z).
    """
    p = safe_normalize(pole_vec)
    if p is None:
        return None
    z_axis = -p  # local +up maps to -pole, so local -up maps to pole
    e = equator_vec
    if e is None:
        return None
    e = e - np.dot(e, z_axis) * z_axis
    e = safe_normalize(e)
    if e is None:
        # fallback axis orthogonal to z
        ref = np.array([0.0, 1.0, 0.0]) if abs(z_axis[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        e = safe_normalize(np.cross(ref, z_axis))
        if e is None:
            return None
    x_axis = e
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)

DISPLAY_VIEW_BASIS = rotation_from_pole_equator(
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
)
DISPLAY_VIEW_MATRIX = DISPLAY_VIEW_BASIS.T if DISPLAY_VIEW_BASIS is not None else IDENTITY_VIEW_MATRIX

def radius_to_plunge(r):
    r = np.clip(r, 0.0, np.sqrt(2.0))
    return 90.0 - 2.0 * np.degrees(np.arcsin(r / np.sqrt(2.0)))

def vectors_to_projected_xy(v, rotation_deg=0.0, view_matrix=None):
    if view_matrix is None:
        view_matrix = IDENTITY_VIEW_MATRIX
    v_view = view_matrix @ v
    upper = v_view[2] > 0
    if np.any(upper):
        v_view[:, upper] *= -1.0
    tr, pl = vector_to_trend_plunge(v_view[0], v_view[1], v_view[2])
    return equal_area_proj(tr, pl, rotation_deg), tr, pl

def trend_plunge_to_projected_xy(trend_deg, plunge_deg, rotation_deg=0.0, view_matrix=None):
    x, y, z = trend_plunge_to_vector(trend_deg, plunge_deg)
    projected, tr, pl = vectors_to_projected_xy(np.vstack([x, y, z]), rotation_deg, view_matrix)
    return projected[0], projected[1], tr, pl

def project_rotated_grid(trend_deg, plunge_deg, rotation_deg, rot_matrix, view_matrix=None):
    x, y, z = trend_plunge_to_vector(trend_deg, plunge_deg)
    v = np.vstack([x, y, z])
    if rot_matrix is not None:
        v = rot_matrix @ v
    if view_matrix is None:
        view_matrix = IDENTITY_VIEW_MATRIX
    v_view = view_matrix @ v
    tr, pl = vector_to_trend_plunge(v_view[0], v_view[1], v_view[2])
    xg, yg = equal_area_proj(tr, pl, rotation_deg)

    visible = v_view[2] <= 0
    if len(visible) < 2:
        if visible.all():
            return xg, yg
        return np.array([np.nan]), np.array([np.nan])

    xs = []
    ys = []
    n = len(visible)
    for i in range(n - 1):
        zi = v_view[2, i]
        zj = v_view[2, i + 1]
        vi = visible[i]
        vj = visible[i + 1]
        if vi:
            xs.append(xg[i]); ys.append(yg[i])
        if vi != vj:
            t = zi / (zi - zj)
            v_int = v_view[:, i] + t * (v_view[:, i + 1] - v_view[:, i])
            tr_i, pl_i = vector_to_trend_plunge(v_int[0], v_int[1], v_int[2])
            x_int, y_int = equal_area_proj(tr_i, pl_i, rotation_deg)
            xs.append(x_int); ys.append(y_int)
            if vi:
                xs.append(np.nan); ys.append(np.nan)
    if visible[-1]:
        xs.append(xg[-1]); ys.append(yg[-1])
    return np.array(xs), np.array(ys)

def mean_triad_from_rows(trend_plunge_cols, ref_means=None):
    """
    Average triad from per-row axes using rotation averaging (SVD of summed rotations).
    trend_plunge_cols: list of (trend_array, plunge_array) for 3 axes (same length).
    ref_means: optional list of 3 reference vectors for sign consistency (axial).
    Returns a list of 3 unit vectors (east, north, up).
    """
    if len(trend_plunge_cols) != 3:
        return None

    trends = [np.asarray(tp[0]) for tp in trend_plunge_cols]
    plunges = [np.asarray(tp[1]) for tp in trend_plunge_cols]
    mask = np.ones_like(trends[0], dtype=bool)
    for t, p in zip(trends, plunges):
        mask &= np.isfinite(t) & np.isfinite(p)
    if not mask.any():
        return None

    mats = []
    idxs = np.where(mask)[0]
    for i in idxs:
        cols = []
        for k, (t, p) in enumerate(zip(trends, plunges)):
            vx, vy, vz = trend_plunge_to_vector(t[i], p[i])
            v = np.array([vx, vy, vz], dtype=float)
            if ref_means is not None and ref_means[k] is not None:
                if np.dot(v, ref_means[k]) < 0:
                    v = -v
            cols.append(v)
        v1, v2, v3 = orthonormalize_triad(cols[0], cols[1], cols[2])
        R = np.stack([v1, v2, v3], axis=1)
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        mats.append(R)

    if not mats:
        return None
    S = np.sum(mats, axis=0)
    U, _, Vt = np.linalg.svd(S)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    triad = [R_avg[:, 0], R_avg[:, 1], R_avg[:, 2]]
    triad = [force_lower_hemisphere(v) for v in triad]
    return triad

def best_fit_rotation_from_axis_pairs(left_pairs, right_pairs, max_iter=8):
    """
    Least-squares proper rotation from right axes to left axes.
    The vectors are axial, so signs are iteratively chosen to minimize mismatch.
    """
    if len(left_pairs) != 3 or len(right_pairs) != 3:
        return None

    left_trends = [np.asarray(tp[0], dtype=float) for tp in left_pairs]
    left_plunges = [np.asarray(tp[1], dtype=float) for tp in left_pairs]
    right_trends = [np.asarray(tp[0], dtype=float) for tp in right_pairs]
    right_plunges = [np.asarray(tp[1], dtype=float) for tp in right_pairs]
    row_mask = np.ones_like(left_trends[0], dtype=bool)
    for values in left_trends + left_plunges + right_trends + right_plunges:
        row_mask &= np.isfinite(values)
    if not row_mask.any():
        return None

    left_vectors = []
    right_vectors = []
    for left_trend, left_plunge, right_trend, right_plunge in zip(left_trends, left_plunges, right_trends, right_plunges):
        lx, ly, lz = trend_plunge_to_vector(left_trend[row_mask], left_plunge[row_mask])
        rx, ry, rz = trend_plunge_to_vector(right_trend[row_mask], right_plunge[row_mask])
        left_vectors.append(np.vstack([lx, ly, lz]).T)
        right_vectors.append(np.vstack([rx, ry, rz]).T)
    left = np.vstack(left_vectors)
    right = np.vstack(right_vectors)

    rot = np.eye(3)
    for _ in range(max_iter):
        right_rot = (rot @ right.T).T
        signs = np.where(np.sum(left * right_rot, axis=1) < 0.0, -1.0, 1.0)
        signed_right = right * signs[:, None]
        cov = signed_right.T @ left
        u, _, vt = np.linalg.svd(cov)
        next_rot = vt.T @ u.T
        if np.linalg.det(next_rot) < 0:
            vt[-1, :] *= -1.0
            next_rot = vt.T @ u.T
        if np.allclose(next_rot, rot, atol=1e-10):
            rot = next_rot
            break
        rot = next_rot
    return rot

def compute_alignment_defaults(df, right_prefix):
    p_trend = to_numeric_series(df, LEFT_COLS["p_trend"])
    p_plunge = to_numeric_series(df, LEFT_COLS["p_plunge"])
    t_trend = to_numeric_series(df, LEFT_COLS["t_trend"])
    t_plunge = to_numeric_series(df, LEFT_COLS["t_plunge"])
    b_trend = to_numeric_series(df, LEFT_COLS["b_trend"])
    b_plunge = to_numeric_series(df, LEFT_COLS["b_plunge"])

    right_cols = right_cols_for_prefix(df, right_prefix)
    if right_cols is None:
        return 0.0, 0.0, [1.0, 0.0, 0.0]

    e_trend_1 = to_numeric_series(df, right_cols[0][0])
    e_plunge_1 = to_numeric_series(df, right_cols[0][1])
    e_trend_2 = to_numeric_series(df, right_cols[1][0])
    e_plunge_2 = to_numeric_series(df, right_cols[1][1])
    e_trend_3 = to_numeric_series(df, right_cols[2][0])
    e_plunge_3 = to_numeric_series(df, right_cols[2][1])

    R_align = best_fit_rotation_from_axis_pairs(
        [
            (p_trend.to_numpy(), p_plunge.to_numpy()),
            (b_trend.to_numpy(), b_plunge.to_numpy()),
            (t_trend.to_numpy(), t_plunge.to_numpy()),
        ],
        [
            (e_trend_1.to_numpy(), e_plunge_1.to_numpy()),
            (e_trend_2.to_numpy(), e_plunge_2.to_numpy()),
            (e_trend_3.to_numpy(), e_plunge_3.to_numpy()),
        ],
    )
    if R_align is None:
        return 0.0, 0.0, [1.0, 0.0, 0.0]

    pole_vec = R_align @ np.array([0.0, 0.0, -1.0])
    equator_ref = R_align @ np.array([1.0, 0.0, 0.0])
    trend, plunge = vector_to_trend_plunge(pole_vec[0], pole_vec[1], pole_vec[2])
    return float(trend), float(plunge), equator_ref.tolist()

def compute_alignment_context(df, right_prefix):
    trend, plunge, equator_ref = compute_alignment_defaults(df, right_prefix)
    align_pole = np.array(trend_plunge_to_vector(trend, plunge))
    align_equator = np.array(equator_ref, dtype=float)
    base_pole = np.array([0.0, 0.0, -1.0])
    base_equator = np.array([1.0, 0.0, 0.0])
    R_min = rotation_between_vectors(base_pole, align_pole)
    equator_min = R_min @ base_equator
    twist = signed_angle_about_axis(equator_min, align_equator, align_pole)
    align_angle_rad = angle_between(base_pole, align_pole)
    align_angle_deg = float(np.degrees(align_angle_rad))
    return {
        "trend": float(trend),
        "plunge": float(plunge),
        "equator": equator_ref,
        "twist": twist,
        "angle_rad": align_angle_rad,
        "angle_deg": align_angle_deg,
    }

def acute_angle_difference_deg(vec_a, vec_b):
    a_n = safe_normalize(vec_a)
    b_n = safe_normalize(vec_b)
    if a_n is None or b_n is None:
        return None
    dot = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(dot)))
    return min(angle, 180.0 - angle)

def customdata_from_trend_plunge(trend_deg, plunge_deg, extra_deg=None):
    cols = [
        np.asarray(trend_deg, dtype=float),
        np.asarray(plunge_deg, dtype=float),
    ]
    if extra_deg is not None:
        cols.append(np.asarray(extra_deg, dtype=float))
    return np.column_stack(cols)

def numeric_filter_bounds(df, filter_column):
    if filter_column is None or filter_column not in df.columns:
        return None
    series = df[filter_column]
    if not pd.api.types.is_numeric_dtype(series):
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min()), float(values.max())

def is_numeric_filter_column(df, filter_column):
    return numeric_filter_bounds(df, filter_column) is not None

def slider_marks(min_value, max_value):
    if min_value == max_value:
        return {min_value: f"{min_value:g}"}
    midpoint = (min_value + max_value) / 2.0
    return {
        min_value: f"{min_value:g}",
        midpoint: f"{midpoint:g}",
        max_value: f"{max_value:g}",
    }

def slider_step(df, filter_column, min_value, max_value):
    if min_value == max_value:
        return 1
    values = pd.to_numeric(df[filter_column], errors="coerce").dropna()
    if not values.empty and np.allclose(values, np.round(values)):
        return 1
    return float((max_value - min_value) / 1000.0)

def apply_row_filter(df, filter_column, filter_values=None, filter_range=None):
    if df is None or filter_column is None or filter_column not in df.columns:
        return df
    if is_numeric_filter_column(df, filter_column):
        if not filter_range or len(filter_range) != 2:
            return df
        values = pd.to_numeric(df[filter_column], errors="coerce")
        lo, hi = sorted(float(v) for v in filter_range)
        return df.loc[values.between(lo, hi, inclusive="both")].copy()
    if not filter_values:
        return df
    allowed = {str(v) for v in filter_values}
    mask = df[filter_column].map(lambda v: pd.notna(v) and str(v) in allowed)
    return df.loc[mask].copy()

def available_filter_columns(df, max_unique=100):
    excluded = set(LEFT_COLS.values())
    for prefix in RIGHT_PREFIXES:
        cols = right_cols_for_prefix(df, prefix)
        if cols is None:
            continue
        for dipdir_col, dip_col in cols:
            excluded.add(dipdir_col)
            excluded.add(dip_col)
    options = []
    for col in df.columns:
        if col in excluded:
            continue
        bounds = numeric_filter_bounds(df, col)
        unique = pd.unique(df[col].dropna())
        if bounds is not None or 1 < len(unique) <= max_unique:
            options.append({"label": col, "value": col})
    return options

def filter_value_options(df, filter_column, max_values=200):
    if filter_column is None or filter_column not in df.columns:
        return []
    unique = pd.unique(df[filter_column].dropna())
    values = sorted(unique.tolist(), key=lambda v: str(v).lower())
    return [{"label": str(v), "value": str(v)} for v in values[:max_values]]

def empty_figure(title):
    fig = go.Figure()
    fig.update_layout(title=title)
    return fig

def build_angle_summary(left_vectors, right_vectors, right_label, filtered_rows, total_rows):
    axis_pairs = [
        ("P", "1", left_vectors[0], right_vectors[0]),
        ("B", "2", left_vectors[1], right_vectors[1]),
        ("T", "3", left_vectors[2], right_vectors[2]),
    ]
    chips = []
    angles = []
    for left_axis, right_idx, left_vec, right_vec in axis_pairs:
        angle = acute_angle_difference_deg(left_vec, right_vec)
        if angle is None:
            label = f"{left_axis} vs {right_label}{right_idx}: n/a"
        else:
            label = f"{left_axis} vs {right_label}{right_idx}: {angle:.1f}°"
            angles.append(angle)
        chips.append(
            html.Span(
                label,
                style={
                    "display": "inline-block",
                    "marginRight": "10px",
                    "marginBottom": "6px",
                    "padding": "6px 10px",
                    "border": "1px solid #c9c9c9",
                    "borderRadius": "999px",
                    "backgroundColor": "#f7f7f7",
                },
            )
        )
    mean_label = "Mean acute difference: n/a"
    if angles:
        mean_label = f"Mean acute difference: {float(np.mean(angles)):.1f}°"
    return html.Div(
        [
            html.Div(
                [
                    html.Strong("Angle difference"),
                    html.Span(f"  Filtered rows: {filtered_rows}/{total_rows}", style={"marginLeft": "10px", "color": "#555"}),
                ],
                style={"marginBottom": "8px"},
            ),
            html.Div(chips),
            html.Div(mean_label, style={"marginTop": "4px", "color": "#444"}),
        ],
        style={"padding": "10px", "border": "1px solid #d9d9d9", "borderRadius": "6px", "backgroundColor": "#fcfcfc"},
    )

def acute_angle_difference_array(trend_a, plunge_a, trend_b, plunge_b):
    ax, ay, az = trend_plunge_to_vector(trend_a, plunge_a)
    bx, by, bz = trend_plunge_to_vector(trend_b, plunge_b)
    dot = (ax * bx) + (ay * by) + (az * bz)
    angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return np.minimum(angle, 180.0 - angle)

def angle_category_counts(row_angles):
    counts = {label: 0 for label in ANGLE_CATEGORY_LABELS}
    for angle in row_angles:
        if not np.isfinite(angle):
            continue
        if angle <= ANGLE_CATEGORY_BINS[0]:
            label = ANGLE_CATEGORY_LABELS[0]
        elif angle <= ANGLE_CATEGORY_BINS[1]:
            label = ANGLE_CATEGORY_LABELS[1]
        elif angle <= ANGLE_CATEGORY_BINS[2]:
            label = ANGLE_CATEGORY_LABELS[2]
        else:
            label = ANGLE_CATEGORY_LABELS[3]
        counts[label] += 1
    return counts

def build_angle_pie_figure(row_angles):
    counts = angle_category_counts(row_angles)
    values = [counts[label] for label in ANGLE_CATEGORY_LABELS]
    fig = go.Figure(
        go.Pie(
            labels=ANGLE_CATEGORY_LABELS,
            values=values,
            marker=dict(colors=ANGLE_CATEGORY_COLORS, line=dict(color="white", width=1)),
            hole=0.42,
            sort=False,
            direction="clockwise",
            textinfo="percent",
            textposition="inside",
            hovertemplate="%{label}<br>Rows: %{value}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        height=PIE_FIG_HEIGHT,
        margin=dict(l=18, r=12, t=44, b=44),
        title=dict(text="Angle Difference Classes", x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="left", x=0),
    )
    return fig

def add_tensor_axis_trace(fig, vec, color, name, width, showlegend):
    if vec is None:
        return
    coords = np.column_stack((-vec, vec))
    fig.add_trace(
        go.Scatter3d(
            x=coords[0],
            y=coords[1],
            z=coords[2],
            mode="lines+markers+text",
            line=dict(color=color, width=width),
            marker=dict(size=3, color=color),
            text=["", name],
            textposition="top center",
            name=name,
            showlegend=showlegend,
            hovertemplate=(
                "Trend: %{customdata[0]:.1f}°<br>"
                "Plunge: %{customdata[1]:.1f}°<extra>%{fullData.name}</extra>"
            ),
            customdata=np.array([
                vector_to_trend_plunge(*(-vec)),
                vector_to_trend_plunge(*vec),
            ]),
        )
    )

def build_tensor_figure(left_vectors, right_vectors, right_label):
    fig = go.Figure()
    cube_edges = [
        ((-1, -1, -1), (1, -1, -1)), ((-1, 1, -1), (1, 1, -1)),
        ((-1, -1, 1), (1, -1, 1)), ((-1, 1, 1), (1, 1, 1)),
        ((-1, -1, -1), (-1, 1, -1)), ((1, -1, -1), (1, 1, -1)),
        ((-1, -1, 1), (-1, 1, 1)), ((1, -1, 1), (1, 1, 1)),
        ((-1, -1, -1), (-1, -1, 1)), ((1, -1, -1), (1, -1, 1)),
        ((-1, 1, -1), (-1, 1, 1)), ((1, 1, -1), (1, 1, 1)),
    ]
    for start, end in cube_edges:
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color="rgba(140,140,140,0.25)", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    add_tensor_axis_trace(fig, left_vectors[0], LEFT_AXIS_COLORS["P"], "Actual P", 8, True)
    add_tensor_axis_trace(fig, left_vectors[1], LEFT_AXIS_COLORS["B"], "Actual B", 8, True)
    add_tensor_axis_trace(fig, left_vectors[2], LEFT_AXIS_COLORS["T"], "Actual T", 8, True)
    add_tensor_axis_trace(fig, right_vectors[0], RIGHT_AXIS_COLORS[0], f"Model {right_label}1", 5, True)
    add_tensor_axis_trace(fig, right_vectors[1], RIGHT_AXIS_COLORS[1], f"Model {right_label}2", 5, True)
    add_tensor_axis_trace(fig, right_vectors[2], RIGHT_AXIS_COLORS[2], f"Model {right_label}3", 5, True)
    fig.update_layout(
        height=TENSOR_FIG_HEIGHT,
        margin=dict(l=6, r=10, t=44, b=44),
        title=dict(text="3D Average Axis View", x=0.02, xanchor="left"),
        scene=dict(
            xaxis=dict(visible=False, range=[-1.1, 1.1]),
            yaxis=dict(visible=False, range=[-1.1, 1.1]),
            zaxis=dict(visible=False, range=[-1.1, 1.1]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=1.25, z=1.0)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="left", x=0),
    )
    return fig

# -------------------------
# Data load
# -------------------------

LEFT_COLS = {
    "p_trend": "P-Axis Trend (°)",
    "p_plunge": "P-Axis Plunge (°)",
    "t_trend": "T-Axis Trend (°)",
    "t_plunge": "T-Axis Plunge (°)",
    "b_trend": "B-Axis Trend (°)",
    "b_plunge": "B-Axis Plunge (°)",
}

RIGHT_PREFIXES = ["E", "S"]
RIGHT_MODE_LABELS = {
    "E": "E (strain)",
    "S": "S (stress)",
    "E/S": "E/S (strain/stress)",
}

def right_mode_label(mode):
    if not mode:
        return RIGHT_MODE_LABELS["E/S"]
    return RIGHT_MODE_LABELS.get(mode, mode)

def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

def to_numeric_series(df, col):
    return pd.to_numeric(df[col], errors="coerce")

def _col_lookup(df):
    return {c.lower(): c for c in df.columns}

def right_cols_for_prefix(df, prefix):
    lookup = _col_lookup(df)
    prefix_l = prefix.lower()
    pairs = []
    for i in (1, 2, 3):
        dipdir_key = f"{prefix_l}dipdir{i}"
        dip_key = f"{prefix_l}dip{i}"
        dipdir_col = lookup.get(dipdir_key)
        dip_col = lookup.get(dip_key)
        if dipdir_col is None or dip_col is None:
            return None
        pairs.append((dipdir_col, dip_col))
    return pairs

def available_right_modes(df):
    modes = []
    for prefix in RIGHT_PREFIXES:
        if right_cols_for_prefix(df, prefix) is not None:
            modes.append(prefix)
    return modes

def validate_dataset(df):
    require_columns(df, list(LEFT_COLS.values()))
    if not available_right_modes(df):
        raise ValueError("Missing right-side columns (expected EDipDir*/EDip* or SDipDir*/SDip*)")
    return df

DEFAULT_RIGHT_MODE = None
RIGHT_MODE_OPTIONS = []
BASE_ALIGN_CTX = {
    "trend": 0.0,
    "plunge": 0.0,
    "equator": [1.0, 0.0, 0.0],
    "twist": 0.0,
    "angle_rad": 0.0,
    "angle_deg": 0.0,
}

# -------------------------
# Dash App
# -------------------------
app = dash.Dash(__name__)
server = app.server

def make_title(path):
    name = os.path.basename(path) if path else ""
    return f"SMTI stereonet comparison — {name}" if name else "SMTI stereonet comparison"

app.layout = html.Div([
    html.H3(make_title(None), id='page_title'),
    html.Div([
        dcc.Upload(
            id='upload_csv',
            children=html.Button('Browse…'),
            accept='.csv',
        ),
        html.Div(id='load_status', style={'marginTop':'6px', 'fontSize':'12px'}),
    ], style={'padding':'10px'}),
    html.Div([
        html.Div([
            html.Label("Stereonet Mode"),
            dcc.Dropdown(
                id='net_mode',
                options=NET_MODE_OPTIONS,
                value=DEFAULT_NET_MODE,
                clearable=False,
            ),
            html.Label("Right Dataset"),
            dcc.Dropdown(
                id='right_mode',
                options=RIGHT_MODE_OPTIONS,
                value=DEFAULT_RIGHT_MODE,
                clearable=True,
                placeholder="Upload a CSV to detect E (strain) / S (stress)",
            ),
            html.Label("Right Δ Trend (°)"),
            dcc.Slider(
                id='right_trend_delta',
                min=-180,
                max=180,
                step=1,
                value=BASE_ALIGN_CTX["trend"],
                marks={-180:'-180',-90:'-90',0:'0',90:'90',180:'180'}
            ),
            html.Label("Right Δ Plunge (°)", style={'marginTop':'8px', 'display':'block'}),
            dcc.Slider(
                id='right_plunge_delta',
                min=-90,
                max=90,
                step=1,
                value=BASE_ALIGN_CTX["angle_deg"],
                marks={-90:'-90',-45:'-45',0:'0',45:'45',90:'90'}
            ),
            html.Div([
                html.Button('No Rotation', id='btn_no_rotation', n_clicks=0),
                html.Button('Best Fit', id='btn_best_fit', n_clicks=0, style={'marginLeft':'8px'}),
            ], style={'marginTop':'10px'}),
        ], style={'width': f'{PANEL_WIDTH}px', 'padding':'10px'}),
        html.Div([
            html.Label("Filter Column"),
            dcc.Dropdown(
                id='filter_column',
                options=[],
                value=None,
                clearable=True,
                placeholder="Optional row filter",
            ),
            html.Div([
                html.Label("Filter Values", style={'marginTop':'8px', 'display':'block'}),
                dcc.Dropdown(
                    id='filter_values',
                    options=[],
                    value=[],
                    multi=True,
                    placeholder="Select values to keep",
                ),
            ], id='filter_values_wrap'),
            html.Div([
                html.Label("Filter Range", style={'marginTop':'8px', 'display':'block'}),
                dcc.RangeSlider(
                    id='filter_range',
                    min=0,
                    max=1,
                    step=0.01,
                    value=[0, 1],
                    marks={0: '0', 1: '1'},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], id='filter_range_wrap', style={'display':'none'}),
            html.Button('Remove All Filters', id='btn_clear_filters', n_clicks=0, style={'marginTop':'10px'}),
            html.Details([
                html.Summary("How to read the oriented stereonet"),
                html.P("Classic view uses an equator-facing display projection. Oriented stereonet keeps the plotted points fixed and rotates the net grid to the average axis frame."),
                html.P("The plotted points represent the same directions in both modes. What changes is the reference frame used by the grid and the 3D average-axis view."),
                html.P("Use Best Fit to rotate the model side toward the actual P/B/T frame, then read the angle-difference summary to see the remaining mismatch."),
            ], style={'marginTop':'12px', 'fontSize':'13px'}),
        ], style={'width': f'{PANEL_WIDTH}px', 'padding':'10px'}),
    ], style={'display':'flex', 'alignItems':'flex-start', 'width': f'{FIG_WIDTH}px'}),
    dcc.Store(id='data_store'),
    dcc.Store(id='right_align_base', data={
        "trend": BASE_ALIGN_CTX["trend"],
        "plunge": BASE_ALIGN_CTX["plunge"],
        "equator": BASE_ALIGN_CTX["equator"],
        "twist": BASE_ALIGN_CTX["twist"],
        "angle_rad": BASE_ALIGN_CTX["angle_rad"],
        "angle_deg": BASE_ALIGN_CTX["angle_deg"],
    }),
    html.Div(
        dcc.Graph(
            id='stereo_graph',
            style={'height': f'{FIG_HEIGHT}px', 'width': f'{FIG_WIDTH}px', 'minWidth': f'{FIG_WIDTH}px'}
        ),
        style={'overflowX': 'auto'}
    ),
    html.Div(id='angle_summary', style={'padding':'10px'}),
    html.Div([
        dcc.Graph(
            id='angle_pie_graph',
            style={'height': f'{PIE_FIG_HEIGHT}px', 'width': f'{PIE_WIDTH}px', 'minWidth': f'{PIE_WIDTH}px'}
        ),
        dcc.Graph(
            id='tensor_graph',
            style={'height': f'{TENSOR_FIG_HEIGHT}px', 'width': f'{TENSOR_WIDTH}px', 'minWidth': f'{TENSOR_WIDTH}px'}
        ),
    ], style={'display':'flex', 'alignItems':'flex-start', 'gap':'12px', 'width': f'{FIG_WIDTH}px', 'overflowX': 'auto'}),
    html.Div(id='debug', style={'display':'none'})  # for debug prints if needed
])

@app.callback(
    Output('data_store', 'data'),
    Output('load_status', 'children'),
    Output('right_trend_delta', 'value'),
    Output('right_plunge_delta', 'value'),
    Output('right_align_base', 'data'),
    Output('right_mode', 'options'),
    Output('right_mode', 'value'),
    Output('page_title', 'children'),
    Output('filter_column', 'options'),
    Output('filter_column', 'value'),
    Output('filter_values', 'options'),
    Output('filter_values', 'value'),
    Output('filter_range', 'value'),
    Input('upload_csv', 'contents'),
    State('upload_csv', 'filename'),
    prevent_initial_call=True,
)
def load_data(upload_contents, upload_filename):
    try:
        if not upload_contents:
            return no_update, "No file selected.", no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        header, b64 = upload_contents.split(',', 1)
        decoded = base64.b64decode(b64)
        df = pd.read_csv(StringIO(decoded.decode('utf-8', errors='replace')))
        df = validate_dataset(df)
        source_label = upload_filename or "uploaded file"
        title = make_title(upload_filename or "")
    except Exception as exc:
        return no_update, f"Load failed: {exc}", no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    modes = available_right_modes(df)
    right_mode = modes[0] if modes else None
    ctx = compute_alignment_context(df, right_mode) if right_mode else BASE_ALIGN_CTX
    filter_cols = available_filter_columns(df)
    return (
        df.to_json(date_format='iso', orient='split'),
        f"Loaded {source_label} ({len(df)} rows)",
        ctx["trend"],
        ctx["angle_deg"],
        ctx,
        [{"label": right_mode_label(m), "value": m} for m in modes],
        right_mode,
        title,
        filter_cols,
        None,
        [],
        [],
        [0, 1],
    )

@app.callback(
    Output('filter_values', 'options', allow_duplicate=True),
    Output('filter_values', 'value', allow_duplicate=True),
    Output('filter_range', 'min'),
    Output('filter_range', 'max'),
    Output('filter_range', 'value', allow_duplicate=True),
    Output('filter_range', 'marks'),
    Output('filter_range', 'step'),
    Output('filter_values_wrap', 'style'),
    Output('filter_range_wrap', 'style'),
    Input('filter_column', 'value'),
    Input('data_store', 'data'),
    prevent_initial_call=True,
)
def on_filter_column_change(filter_column, data_store):
    if not data_store:
        return [], [], 0, 1, [0, 1], {0: '0', 1: '1'}, 0.01, {}, {'display':'none'}
    try:
        df = pd.read_json(StringIO(data_store), orient='split')
    except Exception:
        return [], [], 0, 1, [0, 1], {0: '0', 1: '1'}, 0.01, {}, {'display':'none'}
    bounds = numeric_filter_bounds(df, filter_column)
    if bounds is None:
        return filter_value_options(df, filter_column), [], 0, 1, [0, 1], {0: '0', 1: '1'}, 0.01, {}, {'display':'none'}
    min_value, max_value = bounds
    return [], [], min_value, max_value, [min_value, max_value], slider_marks(min_value, max_value), slider_step(df, filter_column, min_value, max_value), {'display':'none'}, {}

@app.callback(
    Output('filter_column', 'value', allow_duplicate=True),
    Input('btn_clear_filters', 'n_clicks'),
    prevent_initial_call=True,
)
def clear_filters(n_clicks):
    if not n_clicks:
        return no_update
    return None

@app.callback(
    Output('right_trend_delta', 'value', allow_duplicate=True),
    Output('right_plunge_delta', 'value', allow_duplicate=True),
    Output('right_align_base', 'data', allow_duplicate=True),
    Input('btn_no_rotation', 'n_clicks'),
    Input('btn_best_fit', 'n_clicks'),
    State('right_align_base', 'data'),
    State('right_mode', 'value'),
    State('filter_column', 'value'),
    State('filter_values', 'value'),
    State('filter_range', 'value'),
    State('data_store', 'data'),
    prevent_initial_call=True,
)
def set_rotation_buttons(n_no, n_best, right_align_base, right_mode, filter_column, filter_values, filter_range, data_store):
    if not right_align_base:
        right_align_base = {"trend": 0.0, "angle_deg": 0.0}
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    if triggered == "btn_no_rotation":
        return 0.0, 0.0, right_align_base
    if triggered != "btn_best_fit":
        return no_update, no_update, no_update

    ctx = right_align_base
    if right_mode and data_store:
        try:
            df = pd.read_json(StringIO(data_store), orient='split')
            filtered = apply_row_filter(df, filter_column, filter_values, filter_range)
            if filtered is not None and not filtered.empty:
                ctx = compute_alignment_context(filtered, right_mode)
        except Exception:
            ctx = right_align_base
    return float(ctx.get("trend", 0.0)), float(ctx.get("angle_deg", 0.0)), ctx

@app.callback(
    Output('right_trend_delta', 'value', allow_duplicate=True),
    Output('right_plunge_delta', 'value', allow_duplicate=True),
    Output('right_align_base', 'data', allow_duplicate=True),
    Input('right_mode', 'value'),
    Input('filter_column', 'value'),
    Input('filter_values', 'value'),
    Input('filter_range', 'value'),
    Input('data_store', 'data'),
    prevent_initial_call=True,
)
def on_right_mode_change(right_mode, filter_column, filter_values, filter_range, data_store):
    if not right_mode:
        return no_update, no_update, no_update
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    try:
        if data_store:
            df = pd.read_json(StringIO(data_store), orient='split')
        else:
            return no_update, no_update, no_update
    except Exception:
        return no_update, no_update, no_update
    filtered = apply_row_filter(df, filter_column, filter_values, filter_range)
    if filtered is None or filtered.empty:
        return no_update, no_update, BASE_ALIGN_CTX
    ctx = compute_alignment_context(filtered, right_mode)
    if triggered in {"filter_column", "filter_values", "filter_range"}:
        return no_update, no_update, ctx
    return ctx["trend"], ctx["angle_deg"], ctx

@app.callback(
    Output('stereo_graph', 'figure'),
    Output('angle_pie_graph', 'figure'),
    Output('tensor_graph', 'figure'),
    Output('angle_summary', 'children'),
    Input('net_mode', 'value'),
    Input('right_trend_delta', 'value'),
    Input('right_plunge_delta', 'value'),
    Input('right_align_base', 'data'),
    Input('right_mode', 'value'),
    Input('filter_column', 'value'),
    Input('filter_values', 'value'),
    Input('filter_range', 'value'),
    Input('data_store', 'data'),
)
def update_figure(net_mode, right_trend_delta, right_plunge_delta, right_align_base, right_mode, filter_column, filter_values, filter_range, data_store):
    try:
        if data_store:
            df = pd.read_json(StringIO(data_store), orient='split')
        else:
            empty = empty_figure("Upload a CSV to begin")
            return empty, empty, empty, ""
    except Exception as exc:
        message = f"Data load error: {exc}"
        empty = empty_figure(message)
        return empty, empty, empty, ""

    total_rows = len(df)
    filtered_df = apply_row_filter(df, filter_column, filter_values, filter_range)
    if filtered_df is None or filtered_df.empty:
        message = "No rows match the current filter"
        empty = empty_figure(message)
        return empty, empty, empty, html.Div(message, style={"padding": "10px"})
    df = filtered_df

    avg_method = "axial_ortho"
    rotation = 0.0

    if right_trend_delta is None:
        right_trend_delta = 0.0
    if right_plunge_delta is None:
        right_plunge_delta = 0.0
    if not right_align_base:
        right_align_base = {
            "trend": 0.0,
            "plunge": 0.0,
            "equator": [1.0, 0.0, 0.0],
            "twist": 0.0,
            "angle_rad": 0.0,
            "angle_deg": 0.0,
        }

    align_angle_rad = float(right_align_base.get("angle_rad", 0.0))
    align_twist = float(right_align_base.get("twist", 0.0))

    base_pole_vec = np.array([0.0, 0.0, -1.0])
    base_equator = np.array([1.0, 0.0, 0.0])

    target_trend = right_trend_delta % 360.0
    tilt_deg = float(right_plunge_delta)
    if tilt_deg < 0:
        tilt_deg = -tilt_deg
        target_trend = (target_trend + 180.0) % 360.0
    tilt_deg = np.clip(tilt_deg, 0.0, 90.0)
    target_plunge = 90.0 - tilt_deg
    target_pole_vec = np.array(trend_plunge_to_vector(target_trend, target_plunge))

    R_min = rotation_between_vectors(base_pole_vec, target_pole_vec)
    target_equator = R_min @ base_equator

    frac = 0.0
    if align_angle_rad > 1e-6:
        frac = np.clip(angle_between(base_pole_vec, target_pole_vec) / align_angle_rad, 0.0, 1.0)
    twist = align_twist * frac
    target_equator = rotate_about_axis(target_equator, target_pole_vec, twist)

    right_rot = rotation_from_pole_equator(target_pole_vec, target_equator)
    if right_rot is None:
        right_rot = np.eye(3)
    view_matrix = DISPLAY_VIEW_MATRIX

    # Prepare left dataset (P, B, T axes as lines)
    p_trend = to_numeric_series(df, LEFT_COLS["p_trend"])
    p_plunge = to_numeric_series(df, LEFT_COLS["p_plunge"])
    t_trend = to_numeric_series(df, LEFT_COLS["t_trend"])
    t_plunge = to_numeric_series(df, LEFT_COLS["t_plunge"])
    b_trend = to_numeric_series(df, LEFT_COLS["b_trend"])
    b_plunge = to_numeric_series(df, LEFT_COLS["b_plunge"])

    p_mask = p_trend.notna() & p_plunge.notna()
    t_mask = t_trend.notna() & t_plunge.notna()
    b_mask = b_trend.notna() & b_plunge.notna()

    p_trend_vals = p_trend[p_mask].to_numpy()
    p_plunge_vals = p_plunge[p_mask].to_numpy()
    t_trend_vals = t_trend[t_mask].to_numpy()
    t_plunge_vals = t_plunge[t_mask].to_numpy()
    b_trend_vals = b_trend[b_mask].to_numpy()
    b_plunge_vals = b_plunge[b_mask].to_numpy()

    x_p, y_p, _, _ = trend_plunge_to_projected_xy(p_trend_vals, p_plunge_vals, rotation_deg=rotation, view_matrix=view_matrix)
    x_t, y_t, _, _ = trend_plunge_to_projected_xy(t_trend_vals, t_plunge_vals, rotation_deg=rotation, view_matrix=view_matrix)
    x_b, y_b, _, _ = trend_plunge_to_projected_xy(b_trend_vals, b_plunge_vals, rotation_deg=rotation, view_matrix=view_matrix)

    # Precompute means for labeling and options
    p_axial = axial_mean_direction(p_trend[p_mask].to_numpy(), p_plunge[p_mask].to_numpy())
    t_axial = axial_mean_direction(t_trend[t_mask].to_numpy(), t_plunge[t_mask].to_numpy())
    b_axial = axial_mean_direction(b_trend[b_mask].to_numpy(), b_plunge[b_mask].to_numpy())
    p_dir = directional_mean_direction(p_trend[p_mask].to_numpy(), p_plunge[p_mask].to_numpy())
    t_dir = directional_mean_direction(t_trend[t_mask].to_numpy(), t_plunge[t_mask].to_numpy())
    b_dir = directional_mean_direction(b_trend[b_mask].to_numpy(), b_plunge[b_mask].to_numpy())

    p_avg = t_avg = b_avg = None
    if avg_method == "axis_axial":
        p_avg, t_avg, b_avg = p_axial, t_axial, b_axial
    elif avg_method == "axial_ortho":
        p_mean, t_mean, b_mean = p_axial, t_axial, b_axial
    elif avg_method == "dir_ortho":
        p_mean, t_mean, b_mean = p_dir, t_dir, b_dir
    else:
        p_mean = t_mean = b_mean = None
    if avg_method in ("axial_ortho", "dir_ortho") and p_mean is not None and t_mean is not None and b_mean is not None:
        p_avg, t_avg, b_avg = orthonormalize_axial_axes(p_mean, t_mean, b_mean)
        p_avg = force_lower_hemisphere(p_avg)
        t_avg = force_lower_hemisphere(t_avg)
        b_avg = force_lower_hemisphere(b_avg)
    elif avg_method == "joint_eigen":
        triad = mean_triad_from_rows(
            [
                (p_trend.to_numpy(), p_plunge.to_numpy()),
                (b_trend.to_numpy(), b_plunge.to_numpy()),
                (t_trend.to_numpy(), t_plunge.to_numpy()),
            ],
            ref_means=[p_axial, b_axial, t_axial],
        )
        if triad is not None:
            p_avg, b_avg, t_avg = triad

    # Prepare right dataset (three trend/plunge pairs per row)
    right_cols = right_cols_for_prefix(df, right_mode or DEFAULT_RIGHT_MODE)
    right_sets = []
    row_angle_parts = []
    if right_cols is None:
        right_cols = []
    left_pairs = [
        (p_trend, p_plunge),
        (b_trend, b_plunge),
        (t_trend, t_plunge),
    ]
    for idx, (dipdir_col, dip_col) in enumerate(right_cols):
        trend = to_numeric_series(df, dipdir_col)
        plunge = to_numeric_series(df, dip_col)
        mask = trend.notna() & plunge.notna()
        if mask.any():
            tr_rot, pl_rot = rotate_trend_plunge(
                trend[mask].to_numpy(),
                plunge[mask].to_numpy(),
                right_rot,
            )
            x, y, _, _ = trend_plunge_to_projected_xy(tr_rot, pl_rot, rotation_deg=rotation, view_matrix=view_matrix)
        else:
            tr_rot, pl_rot = np.array([]), np.array([])
            x, y = np.array([]), np.array([])
        right_sets.append((dipdir_col, dip_col, trend, plunge, mask, tr_rot, pl_rot, x, y))
        if idx < len(left_pairs):
            left_trend, left_plunge = left_pairs[idx]
            pair_mask = left_trend.notna() & left_plunge.notna() & mask
            pair_angles = pd.Series(np.nan, index=df.index, dtype=float)
            if pair_mask.any():
                pair_tr_rot, pair_pl_rot = rotate_trend_plunge(
                    trend[pair_mask].to_numpy(),
                    plunge[pair_mask].to_numpy(),
                    right_rot,
                )
                pair_angles.loc[pair_mask] = acute_angle_difference_array(
                    left_trend[pair_mask].to_numpy(),
                    left_plunge[pair_mask].to_numpy(),
                    pair_tr_rot,
                    pair_pl_rot,
                )
            row_angle_parts.append(pair_angles)

    if row_angle_parts:
        row_angles = pd.concat(row_angle_parts, axis=1).mean(axis=1, skipna=True).dropna().to_numpy()
    else:
        row_angles = np.array([])

    # Average E1/E2/E3 directions
    e_means = []
    for dipdir_col, dip_col, trend, plunge, mask, _, _, _, _ in right_sets:
        e_means.append(axial_mean_direction(trend[mask].to_numpy(), plunge[mask].to_numpy()))
    e_dir_means = []
    for dipdir_col, dip_col, trend, plunge, mask, _, _, _, _ in right_sets:
        e_dir_means.append(directional_mean_direction(trend[mask].to_numpy(), plunge[mask].to_numpy()))
    e_avg = [None, None, None]
    if avg_method == "axis_axial":
        e_avg = e_means
    elif avg_method in ("axial_ortho", "dir_ortho"):
        means = e_means if avg_method == "axial_ortho" else e_dir_means
        if all(v is not None for v in means):
            e_avg[0], e_avg[1], e_avg[2] = orthonormalize_axial_axes(means[0], means[1], means[2])
            e_avg = [force_lower_hemisphere(v) for v in e_avg]
    elif avg_method == "joint_eigen":
        if len(right_sets) == 3:
            triad = mean_triad_from_rows(
                [
                    (right_sets[0][2].to_numpy(), right_sets[0][3].to_numpy()),
                    (right_sets[1][2].to_numpy(), right_sets[1][3].to_numpy()),
                    (right_sets[2][2].to_numpy(), right_sets[2][3].to_numpy()),
                ],
                ref_means=e_means,
            )
        else:
            triad = None
        if triad is not None:
            e_avg = triad

    e_avg_rot = [None, None, None]
    if all(v is not None for v in e_avg):
        e_avg_rot = [rotate_vector(v, right_rot) for v in e_avg]

    # build two-panel figure
    right_title = right_mode_label(right_mode)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Actual P/B/T (Trend/Plunge)", f"Model {right_title} (Trend/Plunge)"),
        horizontal_spacing=H_SPACING
    )
    # common background circle for stereonet
    circle_theta = np.linspace(0, 2*np.pi, 200)
    circle_x = np.sin(circle_theta)
    circle_y = np.cos(circle_theta)

    # Grid rotations: align with averages if available
    if net_mode == "oriented":
        grid_rot_left = rotation_from_pole_equator(p_avg, t_avg) if (p_avg is not None and t_avg is not None) else None
        grid_rot_right = rotation_from_pole_equator(e_avg_rot[0], e_avg_rot[1]) if (e_avg_rot[0] is not None and e_avg_rot[1] is not None) else None
    else:
        grid_rot_left = None
        grid_rot_right = None

    # Add grid lines (rotated Schmidt net) every 10°
    grid_angles = np.arange(0, 360, 10)
    plunge_circles = list(range(-80, 81, 10))

    grid_color = '#b8b8b8'
    equator_color = '#9a9a9a'

    def add_rotated_grid(rot_matrix, row, col):
        # constant plunge circles
        trend_samples = np.linspace(0, 360, 181)
        for plunge in plunge_circles:
            t = np.full_like(trend_samples, plunge)
            xg, yg = project_rotated_grid(trend_samples, t, rotation, rot_matrix, view_matrix=view_matrix)
            is_equator = abs(plunge) < 1e-6
            fig.add_trace(
                go.Scatter(
                    x=xg, y=yg, mode='lines',
                    line=dict(
                        color=equator_color if is_equator else grid_color,
                        width=1.4 if is_equator else 1.0,
                        dash=None if is_equator else 'dot'
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=row, col=col
            )
        # constant trend lines
        plunge_samples = np.linspace(-90, 90, 121)
        for trend in grid_angles:
            tr = np.full_like(plunge_samples, trend)
            xg, yg = project_rotated_grid(tr, plunge_samples, rotation, rot_matrix, view_matrix=view_matrix)
            fig.add_trace(
                go.Scatter(x=xg, y=yg, mode='lines',
                           line=dict(color=grid_color, width=1, dash='dot'),
                           showlegend=False,
                           hoverinfo='skip'),
                row=row, col=col
            )

    add_rotated_grid(grid_rot_left, row=1, col=1)
    add_rotated_grid(grid_rot_right, row=1, col=2)

    # Left: P/B/T axes
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', line=dict(color='black'), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_p, y=y_p, mode='markers',
        marker=dict(size=6, color=LEFT_AXIS_COLORS["P"], opacity=0.7),
        name='P-Axis',
        legendrank=10,
        customdata=customdata_from_trend_plunge(p_trend_vals, p_plunge_vals),
        hovertemplate="Trend: %{customdata[0]:.1f}°<br>Plunge: %{customdata[1]:.1f}°<extra>P-Axis</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_b, y=y_b, mode='markers',
        marker=dict(size=6, color=LEFT_AXIS_COLORS["B"], opacity=0.7),
        name='B-Axis',
        legendrank=50,
        customdata=customdata_from_trend_plunge(b_trend_vals, b_plunge_vals),
        hovertemplate="Trend: %{customdata[0]:.1f}°<br>Plunge: %{customdata[1]:.1f}°<extra>B-Axis</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_t, y=y_t, mode='markers',
        marker=dict(size=6, color=LEFT_AXIS_COLORS["T"], opacity=0.7),
        name='T-Axis',
        legendrank=90,
        customdata=customdata_from_trend_plunge(t_trend_vals, t_plunge_vals),
        hovertemplate="Trend: %{customdata[0]:.1f}°<br>Plunge: %{customdata[1]:.1f}°<extra>T-Axis</extra>",
    ), row=1, col=1)

    right_label = right_mode or "R"
    # Right: EDip/SDip points
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', line=dict(color='black'), showlegend=False, hoverinfo='skip'), row=1, col=2)
    colors = RIGHT_AXIS_COLORS
    for idx, (dipdir_col, dip_col, _, _, _, tr_rot, pl_rot, x, y) in enumerate(right_sets, start=1):
        label = f"{right_label}{idx}"
        rank = RIGHT_AXIS_RANKS.get(idx, 200 + idx)
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(size=6, color=colors[idx-1], opacity=0.7),
                                 name=label,
                                 legendrank=rank,
                                 customdata=customdata_from_trend_plunge(tr_rot, pl_rot),
                                 hovertemplate="Trend: %{customdata[0]:.1f}°<br>Plunge: %{customdata[1]:.1f}°<extra>" + label + "</extra>"), row=1, col=2)

    # Average markers (orthonormal triads)
    avg_marker_size = 17
    avg_marker_line = dict(color='black', width=1)
    if p_avg is not None and t_avg is not None and b_avg is not None:
        p_tr, p_pl = vector_to_trend_plunge(*p_avg)
        t_tr, t_pl = vector_to_trend_plunge(*t_avg)
        b_tr, b_pl = vector_to_trend_plunge(*b_avg)
        xp, yp, _, _ = trend_plunge_to_projected_xy([p_tr], [p_pl], rotation_deg=rotation, view_matrix=view_matrix)
        xt, yt, _, _ = trend_plunge_to_projected_xy([t_tr], [t_pl], rotation_deg=rotation, view_matrix=view_matrix)
        xb, yb, _, _ = trend_plunge_to_projected_xy([b_tr], [b_pl], rotation_deg=rotation, view_matrix=view_matrix)
        xp, yp = xp[0], yp[0]
        xt, yt = xt[0], yt[0]
        xb, yb = xb[0], yb[0]
        fig.add_trace(go.Scatter(x=[xp], y=[yp], mode='markers',
                                 marker=dict(size=avg_marker_size, color=LEFT_AXIS_COLORS["P"], symbol='circle', line=avg_marker_line),
                                 name='P-Axis Avg', legendrank=20,
                                 customdata=customdata_from_trend_plunge([p_tr], [p_pl]),
                                 hovertemplate="Trend: %{customdata[0]:.1f}°<br>Plunge: %{customdata[1]:.1f}°<extra>P-Axis Avg</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[xb], y=[yb], mode='markers',
                                 marker=dict(size=avg_marker_size, color=LEFT_AXIS_COLORS["B"], symbol='circle', line=avg_marker_line),
                                 name='B-Axis Avg', legendrank=60,
                                 customdata=customdata_from_trend_plunge([b_tr], [b_pl]),
                                 hovertemplate="Trend: %{customdata[0]:.1f}°<br>Plunge: %{customdata[1]:.1f}°<extra>B-Axis Avg</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[xt], y=[yt], mode='markers',
                                 marker=dict(size=avg_marker_size, color=LEFT_AXIS_COLORS["T"], symbol='circle', line=avg_marker_line),
                                 name='T-Axis Avg', legendrank=100,
                                 customdata=customdata_from_trend_plunge([t_tr], [t_pl]),
                                 hovertemplate="Trend: %{customdata[0]:.1f}°<br>Plunge: %{customdata[1]:.1f}°<extra>T-Axis Avg</extra>"), row=1, col=1)

    if all(v is not None for v in e_avg_rot):
        for idx, v in enumerate(e_avg_rot, start=1):
            tr, pl = vector_to_trend_plunge(*v)
            x, y, _, _ = trend_plunge_to_projected_xy([tr], [pl], rotation_deg=rotation, view_matrix=view_matrix)
            x, y = x[0], y[0]
            rank = (RIGHT_AXIS_RANKS.get(idx, 200 + idx) + 10)
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                     marker=dict(size=avg_marker_size, color=colors[idx-1], symbol='circle', line=avg_marker_line),
                                     name=f"{right_label}{idx} Avg",
                                     legendrank=rank,
                                     customdata=customdata_from_trend_plunge([tr], [pl]),
                                     hovertemplate="Trend: %{customdata[0]:.1f}°<br>Plunge: %{customdata[1]:.1f}°<extra>" + f"{right_label}{idx} Avg" + "</extra>"), row=1, col=2)

    # layout cosmetics
    fig.update_xaxes(range=[-1.05,1.05], zeroline=False, showticklabels=False, row=1, col=1, constrain='domain')
    fig.update_yaxes(range=[-1.05,1.05], zeroline=False, showticklabels=False, row=1, col=1,
                     scaleanchor='x', scaleratio=1)
    fig.update_xaxes(range=[-1.05,1.05], zeroline=False, showticklabels=False, row=1, col=2, constrain='domain')
    fig.update_yaxes(range=[-1.05,1.05], zeroline=False, showticklabels=False, row=1, col=2,
                     scaleanchor='x2', scaleratio=1)
    fig.update_layout(
        height=FIG_HEIGHT,
        width=FIG_WIDTH,
        margin=PLOT_MARGIN,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            title=dict(text='Legend', side='top'),
            orientation='h',
            yanchor='top',
            y=-0.12,
            xanchor='left',
            x=0,
            entrywidth=LEGEND_ENTRY_WIDTH,
            entrywidthmode=LEGEND_ENTRY_WIDTH_MODE,
            traceorder='normal'
        )
    )

    left_vectors = [p_avg, b_avg, t_avg]
    right_vectors = [e_avg_rot[0], e_avg_rot[1], e_avg_rot[2]]
    angle_summary = build_angle_summary(left_vectors, right_vectors, right_label, len(df), total_rows)
    angle_pie_fig = build_angle_pie_figure(row_angles)
    tensor_fig = build_tensor_figure(left_vectors, right_vectors, right_label)

    return fig, angle_pie_fig, tensor_fig, angle_summary

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
