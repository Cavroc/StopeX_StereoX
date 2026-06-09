# Stereonet Dash (SMTI Comparison)

Interactive Dash app to compare left-side P/B/T axes against right-side E (strain) or S (stress) axes on dual stereonets.

This README is written for:
1. Power users and programmers who will run or extend the app.
2. AI agents who may pick this up later.

---

**What This App Does**
- Loads a CSV and plots two stereonets side by side.
- Left panel: P/B/T axes (trend/plunge).
- Right panel: E or S axes (trend/plunge), selectable if both exist.
- Computes an orthonormal average triad via joint eigen/SVD.
- Provides rotation sliders and buttons to align right axes to left.
- Uses lower-hemisphere Schmidt equal-area projection.
- Grid is rotated to match average axes (pole and equator).

---

**Quick Start**
1. Create a local virtual environment and install dependencies:
```bash
bash scripts/bootstrap.sh
```
2. Activate the environment:
```bash
source .venv/bin/activate
```
3. Run the app:
```bash
python stereonet_app.py
```
4. Open the URL printed in the console (default `http://127.0.0.1:8050`).
5. Use the **Browse...** button to upload a CSV. There is no default file.

If your system does not provide `python`, use `.venv/bin/python stereonet_app.py` or `python3 stereonet_app.py`.

The repository includes `2025_Stereo.csv` as a sample input file you can upload after launch.

---

**VS Code**
- The repo includes `.vscode/` config for the local `.venv`.
- Run **Bootstrap environment** once from the command palette or terminal.
- Then use the **Run stereonet app** launch configuration or task.

---

**Docker**
Build and run without a local Python environment:

```bash
docker build -t stereonet-app .
docker run --rm -p 8050:8050 stereonet-app
```

---

**Input Data Requirements**

Left-side columns (required):
- `P-Axis Trend (°)`
- `P-Axis Plunge (°)`
- `T-Axis Trend (°)`
- `T-Axis Plunge (°)`
- `B-Axis Trend (°)`
- `B-Axis Plunge (°)`

Right-side columns (one set required, both supported):
- **E (strain)**: `EDipDir1`, `EDip1`, `EDipDir2`, `EDip2`, `EDipDir3`, `EDip3`
- **S (stress)**: `SDipDir1`, `SDip1`, `SDipDir2`, `SDip2`, `SDipDir3`, `SDip3`

Notes:
- Column lookup is case-insensitive.
- If both E and S exist, a dropdown appears to choose the right dataset.

---

**Core Concepts**

**1) Average Triad (Joint Eigen)**
- Each row provides three trend/plunge pairs.
- These are converted to unit vectors.
- A mean triad is computed via SVD to yield orthonormal axes.
- This is the only averaging method exposed in the UI (others are retained in code but not selectable).

**2) Rotation Controls**
- Right-hand sliders represent **delta rotations**.
- `0, 0` means **no rotation** (original right data).
- Default slider values are set to the **best-fit alignment** that moves E/S axes into the P/B/T frame.
- **No Rotation** resets to `0, 0`. **Best Fit** resets to the computed best-fit deltas.

**3) Projection**
- Equal-area Schmidt projection.
- Lower hemisphere enforced for all vectors.
- Grid lines are clipped to visible hemisphere and the equator is solid.

---

**How to Use**
1. Upload a CSV.
2. Choose **Right Dataset** if both E and S are present.
3. Use sliders to adjust **Right Delta Trend** and **Right Delta Plunge**.
4. Use **Best Fit** to align right averages to left averages.
5. Use **No Rotation** to return to the original right orientation.

---

**UI Details**
- Legends are forced to 4 columns x 3 rows using fractional `entrywidth`.
- Star markers are now circles with black outlines for visibility.
- Right-side colors are derived from left (lighter shades).
- Plot dimensions are fixed to maintain a 1:1 stereonet ratio; horizontal scroll is enabled when needed.

---

**Render.com Deployment**

A `render.yaml` is included for one-click deployment on Render.com.
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn --bind 0.0.0.0:$PORT stereonet_app:server`

The app reads the `PORT` environment variable automatically; Render sets this at runtime.

---

**Key Files**
- `stereonet_app.py`: Dash app and all logic.
- `requirements.txt`: Python dependencies.
- `render.yaml`: Render.com service configuration.

---

**Important Functions (for maintainers or AI agents)**
- `mean_triad_from_rows(...)`: Builds the joint eigen/SVD average triad.
- `compute_alignment_context(...)`: Calculates best-fit alignment between right and left.
- `rotation_from_pole_equator(...)`: Builds rotation from a pole/equator pair.
- `rotate_trend_plunge(...)`: Rotates a trend/plunge by a 3D rotation.
- `right_mode_label(...)`: Maps right dataset values to human-readable labels.

---

**Design Invariants (Do Not Break)**
- Sliders represent **delta rotation** only.
- `0,0` must always show the **original right dataset**.
- Default slider values must align E/S averages to P/B/T averages.
- Use lower-hemisphere points only.
- Keep 1:1 stereonet aspect ratio.
- No default CSV; user must upload.
- Right dataset labels are `E (strain)` and `S (stress)`.

---

**Troubleshooting**
- **No dropdown options**: CSV lacks required E/S columns.
- **Sliders default to zero**: alignment context failed; check for NaNs in right data.
- **Points outside the globe**: ensure projection uses equal-area and lower hemisphere enforcement.

---

**Extending the App**
Ideas:
- Add export of aligned data.
- Add statistics panel (mean, dispersion, eigenvalues).
- Add saved presets for rotations.
- Provide a persistent upload history for multiple datasets.

---

**For AI Agents**
If you continue this work later:
- Start by reading `stereonet_app.py` and search for `mean_triad_from_rows`, `compute_alignment_context`, and `update_figure`.
- Keep the delta rotation semantics intact.
- Don't reintroduce a "raw average" or non-orthogonal options in the UI.
- Ensure labels and legend ordering remain stable.
