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
RIGHT_AXIS_COLORS = ['#8ab7e0', '#7fd39a', '#f6c56f']  # P, B, T
RIGHT_AXIS_RANKS = {1: 30, 2: 70, 3: 110}

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

def radius_to_plunge(r):
    r = np.clip(r, 0.0, np.sqrt(2.0))
    return 90.0 - 2.0 * np.degrees(np.arcsin(r / np.sqrt(2.0)))

def project_rotated_grid(trend_deg, plunge_deg, rotation_deg, rot_matrix):
    x, y, z = trend_plunge_to_vector(trend_deg, plunge_deg)
    v = np.vstack([x, y, z])
    if rot_matrix is not None:
        v = rot_matrix @ v
    tr, pl = vector_to_trend_plunge(v[0], v[1], v[2])
    xg, yg = equal_area_proj(tr, pl, rotation_deg)

    visible = v[2] <= 0
    if len(visible) < 2:
        if visible.all():
            return xg, yg
        return np.array([np.nan]), np.array([np.nan])

    xs = []
    ys = []
    n = len(visible)
    for i in range(n - 1):
        zi = v[2, i]
        zj = v[2, i + 1]
        vi = visible[i]
        vj = visible[i + 1]
        if vi:
            xs.append(xg[i]); ys.append(yg[i])
        if vi != vj:
            t = zi / (zi - zj)
            v_int = v[:, i] + t * (v[:, i + 1] - v[:, i])
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

def compute_alignment_defaults(df, right_prefix):
    p_trend = to_numeric_series(df, LEFT_COLS["p_trend"])
    p_plunge = to_numeric_series(df, LEFT_COLS["p_plunge"])
    t_trend = to_numeric_series(df, LEFT_COLS["t_trend"])
    t_plunge = to_numeric_series(df, LEFT_COLS["t_plunge"])
    b_trend = to_numeric_series(df, LEFT_COLS["b_trend"])
    b_plunge = to_numeric_series(df, LEFT_COLS["b_plunge"])

    p_axial = axial_mean_direction(p_trend.to_numpy(), p_plunge.to_numpy())
    t_axial = axial_mean_direction(t_trend.to_numpy(), t_plunge.to_numpy())
    b_axial = axial_mean_direction(b_trend.to_numpy(), b_plunge.to_numpy())

    left_triad = mean_triad_from_rows(
        [
            (p_trend.to_numpy(), p_plunge.to_numpy()),
            (t_trend.to_numpy(), t_plunge.to_numpy()),
            (b_trend.to_numpy(), b_plunge.to_numpy()),
        ],
        ref_means=[p_axial, t_axial, b_axial],
    )

    right_cols = right_cols_for_prefix(df, right_prefix)
    if right_cols is None:
        return 0.0, 0.0, [1.0, 0.0, 0.0]

    e_trend_1 = to_numeric_series(df, right_cols[0][0])
    e_plunge_1 = to_numeric_series(df, right_cols[0][1])
    e_trend_2 = to_numeric_series(df, right_cols[1][0])
    e_plunge_2 = to_numeric_series(df, right_cols[1][1])
    e_trend_3 = to_numeric_series(df, right_cols[2][0])
    e_plunge_3 = to_numeric_series(df, right_cols[2][1])

    e_axial = [
        axial_mean_direction(e_trend_1.to_numpy(), e_plunge_1.to_numpy()),
        axial_mean_direction(e_trend_2.to_numpy(), e_plunge_2.to_numpy()),
        axial_mean_direction(e_trend_3.to_numpy(), e_plunge_3.to_numpy()),
    ]

    right_triad = mean_triad_from_rows(
        [
            (e_trend_1.to_numpy(), e_plunge_1.to_numpy()),
            (e_trend_2.to_numpy(), e_plunge_2.to_numpy()),
            (e_trend_3.to_numpy(), e_plunge_3.to_numpy()),
        ],
        ref_means=e_axial,
    )

    if left_triad is None or right_triad is None:
        return 0.0, 0.0, [1.0, 0.0, 0.0]

    R_left = np.stack(left_triad, axis=1)
    R_right = np.stack(right_triad, axis=1)
    R_align = R_left @ R_right.T

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
        html.Div(style={'width': f'{PANEL_WIDTH}px'}),
        html.Div([
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
    ], style={'display':'flex', 'width': f'{FIG_WIDTH}px'}),
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
    Input('upload_csv', 'contents'),
    State('upload_csv', 'filename'),
    prevent_initial_call=True,
)
def load_data(upload_contents, upload_filename):
    try:
        if not upload_contents:
            return no_update, "No file selected.", no_update, no_update, no_update, no_update, no_update, no_update
        header, b64 = upload_contents.split(',', 1)
        decoded = base64.b64decode(b64)
        df = pd.read_csv(StringIO(decoded.decode('utf-8', errors='replace')))
        df = validate_dataset(df)
        source_label = upload_filename or "uploaded file"
        title = make_title(upload_filename or "")
    except Exception as exc:
        return no_update, f"Load failed: {exc}", no_update, no_update, no_update, no_update, no_update, no_update
    modes = available_right_modes(df)
    right_mode = modes[0] if modes else None
    ctx = compute_alignment_context(df, right_mode) if right_mode else BASE_ALIGN_CTX
    return (
        df.to_json(date_format='iso', orient='split'),
        f"Loaded {source_label} ({len(df)} rows)",
        ctx["trend"],
        ctx["angle_deg"],
        ctx,
        [{"label": right_mode_label(m), "value": m} for m in modes],
        right_mode,
        title,
    )

@app.callback(
    Output('right_trend_delta', 'value', allow_duplicate=True),
    Output('right_plunge_delta', 'value', allow_duplicate=True),
    Input('btn_no_rotation', 'n_clicks'),
    Input('btn_best_fit', 'n_clicks'),
    State('right_align_base', 'data'),
    prevent_initial_call=True,
)
def set_rotation_buttons(n_no, n_best, right_align_base):
    if not right_align_base:
        right_align_base = {"trend": 0.0, "angle_deg": 0.0}
    if n_no is None:
        n_no = 0
    if n_best is None:
        n_best = 0
    if n_no == 0 and n_best == 0:
        return no_update, no_update
    if n_no >= n_best:
        return 0.0, 0.0
    return float(right_align_base.get("trend", 0.0)), float(right_align_base.get("angle_deg", 0.0))

@app.callback(
    Output('right_trend_delta', 'value', allow_duplicate=True),
    Output('right_plunge_delta', 'value', allow_duplicate=True),
    Output('right_align_base', 'data', allow_duplicate=True),
    Input('right_mode', 'value'),
    Input('data_store', 'data'),
    prevent_initial_call=True,
)
def on_right_mode_change(right_mode, data_store):
    if not right_mode:
        return no_update, no_update, no_update
    try:
        if data_store:
            df = pd.read_json(StringIO(data_store), orient='split')
        else:
            return no_update, no_update, no_update
    except Exception:
        return no_update, no_update, no_update
    ctx = compute_alignment_context(df, right_mode)
    return ctx["trend"], ctx["angle_deg"], ctx

@app.callback(
    Output('stereo_graph', 'figure'),
    Input('right_trend_delta', 'value'),
    Input('right_plunge_delta', 'value'),
    Input('right_align_base', 'data'),
    Input('right_mode', 'value'),
    Input('data_store', 'data'),
)
def update_figure(right_trend_delta, right_plunge_delta, right_align_base, right_mode, data_store):
    try:
        if data_store:
            df = pd.read_json(StringIO(data_store), orient='split')
        else:
            fig = go.Figure()
            fig.update_layout(title="Upload a CSV to begin")
            return fig
    except Exception as exc:
        fig = go.Figure()
        fig.update_layout(title=f"Data load error: {exc}")
        return fig

    avg_method = "joint_eigen"
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

    x_p, y_p = equal_area_proj(p_trend[p_mask].to_numpy(), p_plunge[p_mask].to_numpy(), rotation_deg=rotation)
    x_t, y_t = equal_area_proj(t_trend[t_mask].to_numpy(), t_plunge[t_mask].to_numpy(), rotation_deg=rotation)
    x_b, y_b = equal_area_proj(b_trend[b_mask].to_numpy(), b_plunge[b_mask].to_numpy(), rotation_deg=rotation)

    # Precompute means for labeling and options
    p_axial = axial_mean_direction(p_trend[p_mask].to_numpy(), p_plunge[p_mask].to_numpy())
    t_axial = axial_mean_direction(t_trend[t_mask].to_numpy(), t_plunge[t_mask].to_numpy())
    b_axial = axial_mean_direction(b_trend[b_mask].to_numpy(), b_plunge[b_mask].to_numpy())
    p_dir = directional_mean_direction(p_trend[p_mask].to_numpy(), p_plunge[p_mask].to_numpy())
    t_dir = directional_mean_direction(t_trend[t_mask].to_numpy(), t_plunge[t_mask].to_numpy())
    b_dir = directional_mean_direction(b_trend[b_mask].to_numpy(), b_plunge[b_mask].to_numpy())

    if avg_method == "axial_ortho":
        p_mean, t_mean, b_mean = p_axial, t_axial, b_axial
    elif avg_method == "dir_ortho":
        p_mean, t_mean, b_mean = p_dir, t_dir, b_dir
    else:
        p_mean = t_mean = b_mean = None
    p_avg = t_avg = b_avg = None
    if avg_method in ("axial_ortho", "dir_ortho") and p_mean is not None and t_mean is not None and b_mean is not None:
        p_avg, t_avg, b_avg = orthonormalize_triad(p_mean, t_mean, b_mean)
        p_avg = force_lower_hemisphere(p_avg)
        t_avg = force_lower_hemisphere(t_avg)
        b_avg = force_lower_hemisphere(b_avg)
    elif avg_method == "joint_eigen":
        triad = mean_triad_from_rows(
            [
                (p_trend.to_numpy(), p_plunge.to_numpy()),
                (t_trend.to_numpy(), t_plunge.to_numpy()),
                (b_trend.to_numpy(), b_plunge.to_numpy()),
            ],
            ref_means=[p_axial, t_axial, b_axial],
        )
        if triad is not None:
            p_avg, t_avg, b_avg = triad

    # Prepare right dataset (three trend/plunge pairs per row)
    right_cols = right_cols_for_prefix(df, right_mode or DEFAULT_RIGHT_MODE)
    right_sets = []
    if right_cols is None:
        right_cols = []
    for dipdir_col, dip_col in right_cols:
        trend = to_numeric_series(df, dipdir_col)
        plunge = to_numeric_series(df, dip_col)
        mask = trend.notna() & plunge.notna()
        if mask.any():
            tr_rot, pl_rot = rotate_trend_plunge(
                trend[mask].to_numpy(),
                plunge[mask].to_numpy(),
                right_rot,
            )
            x, y = equal_area_proj(tr_rot, pl_rot, rotation_deg=rotation)
        else:
            x, y = np.array([]), np.array([])
        right_sets.append((dipdir_col, dip_col, trend, plunge, mask, x, y))

    # Average E1/E2/E3 directions
    e_means = []
    for dipdir_col, dip_col, trend, plunge, mask, _, _ in right_sets:
        e_means.append(axial_mean_direction(trend[mask].to_numpy(), plunge[mask].to_numpy()))
    e_dir_means = []
    for dipdir_col, dip_col, trend, plunge, mask, _, _ in right_sets:
        e_dir_means.append(directional_mean_direction(trend[mask].to_numpy(), plunge[mask].to_numpy()))
    e_avg = [None, None, None]
    if avg_method in ("axial_ortho", "dir_ortho"):
        means = e_means if avg_method == "axial_ortho" else e_dir_means
        if all(v is not None for v in means):
            e_avg[0], e_avg[1], e_avg[2] = orthonormalize_triad(means[0], means[1], means[2])
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
        subplot_titles=("P/B/T Axes (Trends/Plunges)", f"{right_title} (Trends/Plunges)"),
        horizontal_spacing=H_SPACING
    )
    # common background circle for stereonet
    circle_theta = np.linspace(0, 2*np.pi, 200)
    circle_x = np.sin(circle_theta)
    circle_y = np.cos(circle_theta)

    # Grid rotations: align with averages if available
    grid_rot_left = rotation_from_pole_equator(p_avg, t_avg) if (p_avg is not None and t_avg is not None) else None
    grid_rot_right = rotation_from_pole_equator(e_avg_rot[0], e_avg_rot[1]) if (e_avg_rot[0] is not None and e_avg_rot[1] is not None) else None

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
            xg, yg = project_rotated_grid(trend_samples, t, rotation, rot_matrix)
            is_equator = abs(plunge) < 1e-6
            fig.add_trace(
                go.Scatter(
                    x=xg, y=yg, mode='lines',
                    line=dict(
                        color=equator_color if is_equator else grid_color,
                        width=1.4 if is_equator else 1.0,
                        dash=None if is_equator else 'dot'
                    ),
                    showlegend=False
                ),
                row=row, col=col
            )
        # constant trend lines
        plunge_samples = np.linspace(-90, 90, 121)
        for trend in grid_angles:
            tr = np.full_like(plunge_samples, trend)
            xg, yg = project_rotated_grid(tr, plunge_samples, rotation, rot_matrix)
            fig.add_trace(
                go.Scatter(x=xg, y=yg, mode='lines',
                           line=dict(color=grid_color, width=1, dash='dot'),
                           showlegend=False),
                row=row, col=col
            )

    add_rotated_grid(grid_rot_left, row=1, col=1)
    add_rotated_grid(grid_rot_right, row=1, col=2)

    # Left: P/B/T axes
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', line=dict(color='black'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_p, y=y_p, mode='markers', marker=dict(size=6, color='#1f77b4', opacity=0.7),
                             name='P-Axis', legendrank=10), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_b, y=y_b, mode='markers', marker=dict(size=6, color='#2ca02c', opacity=0.7),
                             name='B-Axis', legendrank=50), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_t, y=y_t, mode='markers', marker=dict(size=6, color='#ff7f0e', opacity=0.7),
                             name='T-Axis', legendrank=90), row=1, col=1)

    right_label = right_mode or DEFAULT_RIGHT_MODE
    # Right: EDip/SDip points
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', line=dict(color='black'), showlegend=False), row=1, col=2)
    colors = RIGHT_AXIS_COLORS
    for idx, (dipdir_col, dip_col, _, _, _, x, y) in enumerate(right_sets, start=1):
        label = f"{right_label}{idx}"
        rank = RIGHT_AXIS_RANKS.get(idx, 200 + idx)
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(size=6, color=colors[idx-1], opacity=0.7),
                                 name=label,
                                 legendrank=rank), row=1, col=2)

    # Average markers (orthonormal triads)
    avg_marker_size = 17
    avg_marker_line = dict(color='black', width=1)
    if p_avg is not None and t_avg is not None and b_avg is not None:
        p_tr, p_pl = vector_to_trend_plunge(*p_avg)
        t_tr, t_pl = vector_to_trend_plunge(*t_avg)
        b_tr, b_pl = vector_to_trend_plunge(*b_avg)
        xp, yp = equal_area_proj(p_tr, p_pl, rotation_deg=rotation)
        xt, yt = equal_area_proj(t_tr, t_pl, rotation_deg=rotation)
        xb, yb = equal_area_proj(b_tr, b_pl, rotation_deg=rotation)
        fig.add_trace(go.Scatter(x=[xp], y=[yp], mode='markers',
                                 marker=dict(size=avg_marker_size, color='#1f77b4', symbol='circle', line=avg_marker_line),
                                 name='P-Axis Avg', legendrank=20), row=1, col=1)
        fig.add_trace(go.Scatter(x=[xb], y=[yb], mode='markers',
                                 marker=dict(size=avg_marker_size, color='#2ca02c', symbol='circle', line=avg_marker_line),
                                 name='B-Axis Avg', legendrank=60), row=1, col=1)
        fig.add_trace(go.Scatter(x=[xt], y=[yt], mode='markers',
                                 marker=dict(size=avg_marker_size, color='#ff7f0e', symbol='circle', line=avg_marker_line),
                                 name='T-Axis Avg', legendrank=100), row=1, col=1)

    if all(v is not None for v in e_avg_rot):
        for idx, v in enumerate(e_avg_rot, start=1):
            tr, pl = vector_to_trend_plunge(*v)
            x, y = equal_area_proj(tr, pl, rotation_deg=rotation)
            rank = (RIGHT_AXIS_RANKS.get(idx, 200 + idx) + 10)
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                     marker=dict(size=avg_marker_size, color=colors[idx-1], symbol='circle', line=avg_marker_line),
                                     name=f"{right_label}{idx} Avg",
                                     legendrank=rank), row=1, col=2)

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

    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)
