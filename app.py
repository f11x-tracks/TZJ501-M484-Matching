"""
TZJ501 vs M484 POR – Thickness Matching Dashboard
===================================================
Tabs
  1. POR Data         – scatter wafer map + mean/std bar chart per tool (ENTITY)
  2. TEST Data        – scatter wafer map + mean/std bar chart per test condition
  3. Normalized Match – side-by-side normalized thickness at matched XY sites (±1 mm)
  4. Spline Profiles  – radial spline from 0 → 150 mm for every wafer in POR & TEST
"""

import os
import glob

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, ctx, dash_table

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
POR_DIR   = os.path.join(BASE_DIR, "POR")
TEST_DIR  = os.path.join(BASE_DIR, "TEST")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_por() -> pd.DataFrame:
    frames = []
    for fp in glob.glob(os.path.join(POR_DIR, "*.csv")):
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    por = pd.concat(frames, ignore_index=True)

    # Keep only necessary columns and rename for uniformity
    por = por.rename(columns={
        "CR_VALUE":     "Thickness_A",
        "X_COORDINATE": "X_mm",
        "Y_COORDINATE": "Y_mm",
        "RAW_WAFER":    "Wafer_ID",
    })
    por["Thickness_A"] = pd.to_numeric(por["Thickness_A"], errors="coerce")
    por["X_mm"]        = pd.to_numeric(por["X_mm"],        errors="coerce")
    por["Y_mm"]        = pd.to_numeric(por["Y_mm"],        errors="coerce")
    por = por.dropna(subset=["Thickness_A", "X_mm", "Y_mm"])

    # Normalized thickness per wafer  (value / wafer mean)
    por["Norm_Thickness"] = por.groupby("Wafer_ID")["Thickness_A"].transform(
        lambda s: s / s.mean()
    )
    # Radial distance from centre
    por["Radius_mm"] = np.sqrt(por["X_mm"]**2 + por["Y_mm"]**2)
    por["Source"] = "POR"
    return por


def load_test() -> pd.DataFrame:
    """
    Each CSV is clean: header row then 32-row blocks per wafer.
    Wafers are identified by sequential Point No 1-32 repeating.
    Test conditions are constant within each 32-row block (DispSS differs across files/blocks).
    """
    frames = []
    for fp in glob.glob(os.path.join(TEST_DIR, "*.csv")):
        fname = os.path.splitext(os.path.basename(fp))[0]
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip()

        # Rename columns to standard names
        col_map = {}
        for c in df.columns:
            cl = c.strip()
            if cl == "Film Thickness":
                col_map[c] = "Film_Thickness_nm"
            elif cl == "X[mm]":
                col_map[c] = "X_mm"
            elif cl == "Y[mm]":
                col_map[c] = "Y_mm"
            elif cl == "Point No":
                col_map[c] = "Point_No"
            elif cl == "Fit Rate":
                col_map[c] = "Fit_Rate"
        df = df.rename(columns=col_map)

        # Only require essential measurement columns
        required = ["Point_No", "Film_Thickness_nm", "X_mm", "Y_mm"]
        missing = [r for r in required if r not in df.columns]
        if missing:
            print(f"WARNING: {fp} missing columns {missing} – skipping")
            continue

        df["Point_No"]         = pd.to_numeric(df["Point_No"],         errors="coerce")
        df["Film_Thickness_nm"]= pd.to_numeric(df["Film_Thickness_nm"],errors="coerce")
        df["X_mm"]             = pd.to_numeric(df["X_mm"],             errors="coerce")
        df["Y_mm"]             = pd.to_numeric(df["Y_mm"],             errors="coerce")
        df = df.dropna(subset=["Point_No", "Film_Thickness_nm", "X_mm", "Y_mm"])

        # Convert nm → Å (×10)
        df["Thickness_A"] = df["Film_Thickness_nm"] * 10.0

        # Assign Wafer_ID: each time Point_No resets to 1 → new wafer
        wafer_num = 0
        wafer_ids = []
        for pn in df["Point_No"].tolist():
            if int(pn) == 1:
                wafer_num += 1
            wafer_ids.append(f"{fname}_W{wafer_num:02d}")
        df["Wafer_ID"] = wafer_ids

        # Test condition label (build from available condition columns)
        condition_parts = []
        condition_cols = ["DispT", "PumpT", "DispSS", "RlxT", "RlxSS", "Cast", "Other"]
        for col in condition_cols:
            if col in df.columns:
                value = df[col].astype(str) if col != "Other" else df[col].fillna("").astype(str)
                condition_parts.append(f"{col}=" + value)
        
        if condition_parts:
            df["Condition"] = " | ".join(condition_parts)
        else:
            # No condition columns available - use filename as identifier
            df["Condition"] = f"File: {fname}"
        df["ENTITY"] = "TZJ501"
        df["Source"] = f"TEST ({fname})"
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    test = pd.concat(frames, ignore_index=True)

    # Normalized thickness per wafer
    test["Norm_Thickness"] = test.groupby("Wafer_ID")["Thickness_A"].transform(
        lambda s: s / s.mean()
    )
    test["Radius_mm"] = np.sqrt(test["X_mm"]**2 + test["Y_mm"]**2)
    return test


# ── load once at startup ───────────────────────────────────────────────────────
por_df  = load_por()
test_df = load_test()

# ══════════════════════════════════════════════════════════════════════════════
# HELPER – wafer-map scatter
# ══════════════════════════════════════════════════════════════════════════════

def wafer_map_figure(df: pd.DataFrame, color_col: str, title: str,
                     hover_cols: list) -> go.Figure:
    fig = px.scatter(
        df, x="X_mm", y="Y_mm", color=color_col,
        hover_data=hover_cols,
        title=title,
        labels={"X_mm": "X (mm)", "Y_mm": "Y (mm)"},
        color_continuous_scale="RdYlGn",
    )
    # Wafer edge circle
    theta = np.linspace(0, 2 * np.pi, 300)
    fig.add_shape(type="circle", x0=-150, y0=-150, x1=150, y1=150,
                  line_color="gray", line_dash="dash", line_width=1)
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#1a1a2e",
        height=520,
        yaxis_scaleanchor="x",
    )
    return fig


def por_contour_figure(df: pd.DataFrame, color_col: str, title: str) -> go.Figure:
    """Contour map of POR thickness interpolated onto a regular grid."""
    from scipy.interpolate import griddata

    # Average at each unique XY (multiple wafers → mean)
    avg = df.groupby(["X_mm", "Y_mm"])[color_col].mean().reset_index()
    x = avg["X_mm"].values
    y = avg["Y_mm"].values
    z = avg[color_col].values

    # Regular grid inside wafer boundary
    grid_r = np.linspace(-150, 150, 200)
    xi, yi = np.meshgrid(grid_r, grid_r)
    zi = griddata((x, y), z, (xi, yi), method="cubic")

    # Mask outside wafer circle
    zi[xi**2 + yi**2 > 150**2] = np.nan

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=grid_r, y=grid_r, z=zi,
        colorscale="RdYlGn",
        contours=dict(coloring="heatmap", showlabels=True,
                      labelfont=dict(size=9, color="#1a1a2e")),
        colorbar=dict(title=dict(text=color_col, font=dict(color="#1a1a2e")),
                      tickfont=dict(color="#1a1a2e")),
        hovertemplate="X: %{x:.0f} mm<br>Y: %{y:.0f} mm<br>Value: %{z:.1f}<extra></extra>",
    ))
    # Overlay raw measurement points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=5, color="#888", opacity=0.4),
        hoverinfo="skip",
        showlegend=False,
    ))
    # Wafer edge
    theta = np.linspace(0, 2 * np.pi, 300)
    fig.add_shape(type="circle", x0=-150, y0=-150, x1=150, y1=150,
                  line_color="gray", line_dash="dash", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        xaxis=dict(range=[-155, 155], constrain="domain"),
        yaxis=dict(range=[-155, 155], scaleanchor="x", scaleratio=1, constrain="domain"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#1a1a2e",
        height=520,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPER – mean / std bar chart
# ══════════════════════════════════════════════════════════════════════════════

def mean_std_figure(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    grp = df.groupby(group_col)["Thickness_A"].agg(["mean", "std"]).reset_index()
    grp.columns = [group_col, "Mean_A", "Std_A"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp[group_col], y=grp["Mean_A"],
        error_y=dict(type="data", array=grp["Std_A"].tolist(), visible=True),
        marker_color="#1d6fa4",
        name="Mean ± σ",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=group_col,
        yaxis_title="Thickness (Å)",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#1a1a2e",
        height=380,
    )
    return fig


def std_figure(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    grp = df.groupby(group_col)["Thickness_A"].std().reset_index()
    grp.columns = [group_col, "Std_A"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp[group_col], y=grp["Std_A"],
        marker_color="#f97316",
        name="σ Thickness",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=group_col,
        xaxis=dict(tickangle=-45),
        yaxis_title="Std Dev Thickness (Å)",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#1a1a2e",
        height=480,
        margin=dict(b=220),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPER – spline radial profile for all wafers
# ══════════════════════════════════════════════════════════════════════════════

def spline_figure(df: pd.DataFrame, title: str, color_col: str = "Wafer_ID") -> go.Figure:
    fig = go.Figure()
    r_dense = np.linspace(0, 150, 500)

    unique_ids = df[color_col].unique()
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    color_map = {uid: palette[i % len(palette)] for i, uid in enumerate(unique_ids)}

    for uid, grp in df.groupby(color_col):
        grp_sorted = grp.sort_values("Radius_mm")
        r = grp_sorted["Radius_mm"].values
        t = grp_sorted["Thickness_A"].values
        if len(r) < 4:
            continue
        try:
            spl = UnivariateSpline(r, t, s=len(r) * 50, ext=3)
            t_fit = spl(r_dense)
        except Exception:
            continue
        fig.add_trace(go.Scatter(
            x=r_dense, y=t_fit,
            mode="lines",
            name=str(uid),
            line=dict(color=color_map[uid], width=1.5),
            opacity=0.85,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Radius (mm)",
        yaxis_title="Thickness (Å)",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#1a1a2e",
        height=480,
        legend=dict(font_size=9),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HELPER – delta thickness by measurement site (TEST − POR)
# ══════════════════════════════════════════════════════════════════════════════

def delta_wafer_map_figure(condition: str = "All", entity: str = "All") -> go.Figure:
    """Wafer map coloured by Δ Thickness (TEST − POR) at each matched XY site."""
    if por_df.empty or test_df.empty:
        return go.Figure().update_layout(title="No data loaded",
                                         paper_bgcolor="#ffffff", font_color="#1a1a2e")

    p = por_df.copy()
    if entity != "All":
        p = p[p["ENTITY"] == entity]
    t = test_df.copy()
    if condition != "All":
        t = t[t["Condition"] == condition]

    if p.empty or t.empty:
        return go.Figure().update_layout(title="No data after filtering",
                                         paper_bgcolor="#ffffff", font_color="#1a1a2e")

    p["Xr"] = p["X_mm"].round(0).astype(int)
    p["Yr"] = p["Y_mm"].round(0).astype(int)
    t["Xr"] = t["X_mm"].round(0).astype(int)
    t["Yr"] = t["Y_mm"].round(0).astype(int)

    p_avg = p.groupby(["Xr", "Yr"])["Thickness_A"].mean().reset_index()
    p_avg.columns = ["Xr", "Yr", "POR_mean"]
    t_avg = t.groupby(["Xr", "Yr"])["Thickness_A"].mean().reset_index()
    t_avg.columns = ["Xr", "Yr", "TEST_mean"]

    merged = pd.merge(t_avg, p_avg, on=["Xr", "Yr"])
    if merged.empty:
        return go.Figure().update_layout(title="No matching XY sites found",
                                         paper_bgcolor="#ffffff", font_color="#1a1a2e")

    merged["Delta_A"] = merged["TEST_mean"] - merged["POR_mean"]
    abs_max = merged["Delta_A"].abs().max()
    clr_range = [-abs_max, abs_max] if abs_max > 0 else [-1, 1]

    cond_label   = condition if condition != "All" else "All Conditions"
    entity_label = entity   if entity   != "All" else "All Entities"

    fig = px.scatter(
        merged, x="Xr", y="Yr",
        color="Delta_A",
        color_continuous_scale="RdBu_r",
        range_color=clr_range,
        custom_data=["TEST_mean", "POR_mean", "Delta_A"],
        title=(f"Δ Thickness Wafer Map (TEST − POR)  "
               f"[TEST: {cond_label} | POR: {entity_label}]"),
        labels={"Xr": "X (mm)", "Yr": "Y (mm)", "Delta_A": "Δ (Å)"},
    )
    fig.update_traces(
        marker=dict(size=14),
        hovertemplate=(
            "X: %{x} mm, Y: %{y} mm<br>"
            "TEST mean: %{customdata[0]:.1f} Å<br>"
            "POR mean: %{customdata[1]:.1f} Å<br>"
            "Δ: %{customdata[2]:+.1f} Å<extra></extra>"
        ),
    )
    fig.add_shape(type="circle", x0=-150, y0=-150, x1=150, y1=150,
                  line_color="gray", line_dash="dash", line_width=1)
    fig.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font_color="#1a1a2e",
        height=520, yaxis_scaleanchor="x")
    return fig


def delta_by_site_figure(condition: str = "All", entity: str = "All") -> go.Figure:
    """Bar chart of Δ Thickness (TEST − POR) at each of the 32 XY measurement sites."""
    if por_df.empty or test_df.empty:
        return go.Figure().update_layout(title="No data loaded",
                                         paper_bgcolor="#ffffff", font_color="#1a1a2e")

    p = por_df.copy()
    if entity != "All":
        p = p[p["ENTITY"] == entity]

    t = test_df.copy()
    if condition != "All":
        t = t[t["Condition"] == condition]

    if p.empty or t.empty:
        return go.Figure().update_layout(title="No data after filtering",
                                         paper_bgcolor="#ffffff", font_color="#1a1a2e")

    p["Xr"] = p["X_mm"].round(0).astype(int)
    p["Yr"] = p["Y_mm"].round(0).astype(int)
    t["Xr"] = t["X_mm"].round(0).astype(int)
    t["Yr"] = t["Y_mm"].round(0).astype(int)

    p_avg = p.groupby(["Xr", "Yr"])["Thickness_A"].mean().reset_index()
    p_avg.columns = ["Xr", "Yr", "POR_mean"]

    t_avg = t.groupby(["Xr", "Yr"])["Thickness_A"].mean().reset_index()
    t_avg.columns = ["Xr", "Yr", "TEST_mean"]

    merged = pd.merge(t_avg, p_avg, on=["Xr", "Yr"])
    if merged.empty:
        return go.Figure().update_layout(title="No matching XY sites found",
                                         paper_bgcolor="#ffffff", font_color="#1a1a2e")

    merged["Delta_A"] = merged["TEST_mean"] - merged["POR_mean"]
    merged["Radius_mm"] = np.sqrt(merged["Xr"]**2 + merged["Yr"]**2)
    merged = merged.sort_values(["Radius_mm", "Xr", "Yr"]).reset_index(drop=True)
    merged["Site"] = [f"({r.Xr},{r.Yr})" for r in merged.itertuples()]

    colors = ["#ef4444" if d < 0 else "#1d6fa4" for d in merged["Delta_A"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=merged["Site"],
        y=merged["Delta_A"],
        marker_color=colors,
        customdata=merged[["Xr", "Yr", "TEST_mean", "POR_mean", "Radius_mm"]].values,
        hovertemplate=(
            "Site: (%{customdata[0]}, %{customdata[1]}) mm<br>"
            "TEST mean: %{customdata[2]:.1f} Å<br>"
            "POR mean: %{customdata[3]:.1f} Å<br>"
            "Δ (TEST−POR): %{y:.1f} Å<br>"
            "Radius: %{customdata[4]:.1f} mm<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_color="gray", line_dash="dash", line_width=1)

    cond_label   = condition if condition != "All" else "All Conditions"
    entity_label = entity   if entity   != "All" else "All Entities"
    fig.update_layout(
        title=(f"Δ Thickness (TEST − POR) by Measurement Site  "
               f"[TEST: {cond_label} | POR: {entity_label}]"),
        xaxis_title="Measurement Site (X, Y) mm  [sorted by radius]",
        yaxis_title="Δ Thickness (Å)",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#1a1a2e",
        height=540,
        xaxis=dict(tickangle=-45),
        bargap=0.25,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DROPDOWN OPTIONS
# ══════════════════════════════════════════════════════════════════════════════

por_entities   = ["All"] + (sorted(por_df["ENTITY"].unique().tolist())  if not por_df.empty  else [])
por_wafers     = ["All"] + (sorted(por_df["Wafer_ID"].unique().tolist()) if not por_df.empty  else [])
test_sources   = ["All"] + (sorted(test_df["Source"].unique().tolist())  if not test_df.empty else [])
test_wafers    = ["All"] + (sorted(test_df["Wafer_ID"].unique().tolist())if not test_df.empty else [])
test_conditions= ["All"] + (sorted(test_df["Condition"].dropna().unique().astype(str).tolist()) if not test_df.empty else [])
delta_cond_default = test_conditions[1] if len(test_conditions) > 1 else "All"

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__, title="TZJ501 – M484 Matching")

DARK = "#f0f2f5"
CARD = "#ffffff"
ACCENT = "#1d6fa4"

def dd(id_, options, value="All", w="220px"):
    return dcc.Dropdown(
        id=id_,
        options=[{"label": o, "value": o} for o in options],
        value=value,
        clearable=False,
        style={"width": w, "color": "#000", "fontSize": "13px"},
    )


app.layout = html.Div(style={"backgroundColor": DARK, "minHeight": "100vh",
                              "fontFamily": "Arial, sans-serif", "color": "#1a1a2e",
                              "padding": "12px"}, children=[

    html.H2("TZJ501 – M484 Thickness Matching Dashboard",
            style={"textAlign": "center", "color": ACCENT, "marginBottom": "4px"}),

    dcc.Tabs(id="tabs", value="por", style={"marginBottom": "10px"},
             colors={"border": "#444", "primary": ACCENT, "background": CARD},
             children=[

        # ── TAB 1 : POR ────────────────────────────────────────────────────
        dcc.Tab(label="POR Data", value="por",
                style={"color": "#1a1a2e", "backgroundColor": CARD},
                selected_style={"color": "black", "backgroundColor": ACCENT}, children=[

            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                            "margin": "8px 0"}, children=[
                html.Div([html.Label("Filter by Tool (ENTITY)"),
                          dd("por-entity-dd", por_entities)]),
                html.Div([html.Label("Filter by Wafer"),
                          dd("por-wafer-dd", por_wafers)]),
                html.Div([html.Label("Color map by"),
                          dcc.RadioItems(
                              id="por-color-radio",
                              options=[{"label": "  Thickness (Å)", "value": "Thickness_A"},
                                       {"label": "  Norm. Thickness",  "value": "Norm_Thickness"}],
                              value="Thickness_A",
                              inline=True,
                              style={"color": "#1a1a2e"},
                          )]),
            ]),

            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
                html.Div(dcc.Graph(id="por-wafer-map"),  style={"flex": "1 1 500px"}),
                html.Div(dcc.Graph(id="por-mean-std"),   style={"flex": "1 1 460px"}),
            ]),
        ]),

        # ── TAB 2 : TEST ───────────────────────────────────────────────────
        dcc.Tab(label="TEST Data", value="test",
                style={"color": "#1a1a2e", "backgroundColor": CARD},
                selected_style={"color": "black", "backgroundColor": ACCENT}, children=[

            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                            "margin": "8px 0"}, children=[
                html.Div([html.Label("Filter by File / Source"),
                          dd("test-source-dd", test_sources)]),
                html.Div([html.Label("Filter by Test Condition"),
                          dcc.Dropdown(
                              id="test-cond-dd",
                              options=[{"label": o, "value": o} for o in test_conditions if o != "All"],
                              value=None,
                              multi=True,
                              placeholder="All conditions",
                              style={"width": "520px", "color": "#000", "fontSize": "13px"},
                          )]),
                html.Div([html.Label("Filter by Wafer"),
                          dd("test-wafer-dd", test_wafers)]),
                html.Div([html.Label("Color map by"),
                          dcc.RadioItems(
                              id="test-color-radio",
                              options=[{"label": "  Thickness (Å)", "value": "Thickness_A"},
                                       {"label": "  Norm. Thickness",  "value": "Norm_Thickness"}],
                              value="Thickness_A",
                              inline=True,
                              style={"color": "#1a1a2e"},
                          )]),
            ]),

            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
                html.Div(style={"flex": "0 0 900px", "maxWidth": "900px"}, children=[
                    dcc.Graph(id="test-wafer-map"),
                    dcc.Graph(id="test-spline",
                              style={"width": "900px", "marginTop": "8px"}),
                ]),
                html.Div(style={"flex": "1 1 500px", "overflowX": "auto"}, children=[
                    html.H4("Thickness Statistics by Condition",
                            style={"color": ACCENT, "marginBottom": "6px", "fontSize": "14px"}),
                    dash_table.DataTable(
                        id="test-mean-std",
                        columns=[
                            {"name": "Condition",          "id": "Condition",  "type": "text"},
                            {"name": "Mean (Å)",            "id": "Mean_A",    "type": "numeric",
                             "format": {"specifier": ".1f"}},
                            {"name": "σ Overall (Å)",       "id": "Std_A",     "type": "numeric",
                             "format": {"specifier": ".2f"}},
                            {"name": "σ Z1 0-50mm (Å)",    "id": "Std_Z1",    "type": "numeric",
                             "format": {"specifier": ".2f"}},
                            {"name": "σ Z2 50-100mm (Å)",  "id": "Std_Z2",    "type": "numeric",
                             "format": {"specifier": ".2f"}},
                            {"name": "σ Z3 100-148mm (Å)", "id": "Std_Z3",    "type": "numeric",
                             "format": {"specifier": ".2f"}},
                        ],
                        data=[],
                        sort_action="native",
                        sort_mode="single",
                        filter_action="native",
                        page_action="native",
                        page_size=50,
                        style_table={"overflowX": "auto"},
                        style_header={"backgroundColor": "#d0d4db", "color": "#1a1a2e",
                                      "fontWeight": "bold", "fontSize": "13px"},
                        style_cell={"backgroundColor": CARD, "color": "#1a1a2e",
                                    "fontSize": "12px", "padding": "6px 10px",
                                    "maxWidth": "380px", "overflow": "hidden",
                                    "textOverflow": "ellipsis"},
                        style_data_conditional=[
                            {"if": {"row_index": "odd"},
                             "backgroundColor": "#e8eaee"},
                        ],
                        tooltip_data=[],
                        tooltip_delay=0,
                        tooltip_duration=None,
                    ),
                ]),
            ]),
        ]),

        # ── TAB 3 : DELTA BY SITE ──────────────────────────────────────────
        dcc.Tab(label="Delta by Site", value="match",
                style={"color": "#1a1a2e", "backgroundColor": CARD},
                selected_style={"color": "black", "backgroundColor": ACCENT}, children=[

            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                            "margin": "8px 0", "alignItems": "flex-end"}, children=[
                html.Div([html.Label("TEST Condition"),
                          dd("delta-cond-dd", test_conditions,
                             value=delta_cond_default, w="520px")]),
                html.Div([html.Label("POR Tool (ENTITY)"),
                          dd("delta-entity-dd", por_entities)]),
            ]),

            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
                html.Div(dcc.Graph(id="delta-graph"),    style={"flex": "1 1 560px"}),
                html.Div(dcc.Graph(id="delta-wafer-map"), style={"flex": "1 1 480px"}),
            ]),

            html.Div(id="delta-stats",
                     style={"margin": "8px 4px", "fontSize": "13px", "color": "#a0a0c0"}),
        ]),

        # ── TAB 4 : SPLINE PROFILES ────────────────────────────────────────
        dcc.Tab(label="Spline Profiles", value="spline",
                style={"color": "#1a1a2e", "backgroundColor": CARD},
                selected_style={"color": "black", "backgroundColor": ACCENT}, children=[

            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                            "margin": "8px 0"}, children=[
                html.Div([html.Label("POR – Filter by Tool"),
                          dd("spline-por-entity-dd", por_entities)]),
                html.Div([html.Label("TEST – Filter by Source"),
                          dd("spline-test-source-dd", test_sources)]),
                html.Div([html.Label("TEST – Filter by Condition"),
                          dd("spline-test-cond-dd", test_conditions, w="360px")]),
            ]),

            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
                html.Div(dcc.Graph(id="spline-por"),  style={"flex": "1 1 500px"}),
                html.Div(dcc.Graph(id="spline-test"), style={"flex": "1 1 500px"}),
            ]),
        ]),
    ]),
])


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

# ── POR ────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("por-wafer-map", "figure"),
    Output("por-mean-std",  "figure"),
    Input("por-entity-dd",  "value"),
    Input("por-wafer-dd",   "value"),
    Input("por-color-radio","value"),
)
def update_por(entity, wafer, color_col):
    df = por_df.copy()
    if entity != "All":
        df = df[df["ENTITY"] == entity]
    if wafer != "All":
        df = df[df["Wafer_ID"] == wafer]
    if df.empty:
        empty = go.Figure().update_layout(paper_bgcolor=CARD, font_color="#1a1a2e",
                                          title="No data")
        return empty, empty

    map_fig  = por_contour_figure(df, color_col,
                                   f"POR Contour Map – {color_col}")
    msg_grp  = "ENTITY" if wafer == "All" else "Wafer_ID"
    bar_fig  = std_figure(df, msg_grp,
                          f"POR Thickness σ by {msg_grp}")
    return map_fig, bar_fig


# ── TEST ───────────────────────────────────────────────────────────────────────
@app.callback(
    Output("test-wafer-map", "figure"),
    Output("test-mean-std",  "data"),
    Output("test-mean-std",  "tooltip_data"),
    Output("test-spline",    "figure"),
    Input("test-source-dd",  "value"),
    Input("test-cond-dd",    "value"),
    Input("test-wafer-dd",   "value"),
    Input("test-color-radio","value"),
)
def update_test(source, cond, wafer, color_col):
    df = test_df.copy()
    if source != "All":
        df = df[df["Source"] == source]
    if cond:  # list of selected conditions (None or [] means all)
        df = df[df["Condition"].isin(cond)]
    if wafer != "All":
        df = df[df["Wafer_ID"] == wafer]
    if df.empty:
        empty = go.Figure().update_layout(paper_bgcolor=CARD, font_color="#1a1a2e",
                                          title="No data")
        return empty, [], [], empty

    map_fig = por_contour_figure(df, color_col,
                                 f"TEST Contour Map – {color_col}")

    grp_col = "Wafer_ID" if wafer != "All" else "Condition"
    grp = df.groupby(grp_col)["Thickness_A"].agg(
        Mean_A="mean", Std_A="std"
    ).reset_index().round({"Mean_A": 1, "Std_A": 2})
    grp = grp.rename(columns={grp_col: "Condition"})

    # Radial zone std devs
    for col, rmin, rmax in [("Std_Z1", 0, 50), ("Std_Z2", 50, 100), ("Std_Z3", 100, 148)]:
        zone_std = (
            df[(df["Radius_mm"] >= rmin) & (df["Radius_mm"] < rmax)]
            .groupby(grp_col)["Thickness_A"].std()
            .round(2)
            .rename(col)
        )
        grp = grp.merge(zone_std, left_on="Condition", right_on=grp_col, how="left")

    grp = grp.sort_values("Std_A", ascending=True)
    tbl_data = grp.to_dict("records")
    tooltips = [{"Condition": {"value": str(r["Condition"]), "type": "markdown"}}
                for r in tbl_data]

    # Spline profile coloured by Condition
    spline_fig = spline_figure(df, "TEST – Radial Spline Profiles", "Condition")

    return map_fig, tbl_data, tooltips, spline_fig


# ── DELTA BY SITE ─────────────────────────────────────────────────────────────
@app.callback(
    Output("delta-graph",     "figure"),
    Output("delta-wafer-map", "figure"),
    Output("delta-stats",     "children"),
    Input("delta-cond-dd",    "value"),
    Input("delta-entity-dd",  "value"),
)
def update_delta(condition, entity):
    fig     = delta_by_site_figure(condition, entity)
    waf_fig = delta_wafer_map_figure(condition, entity)

    # stats summary
    p = por_df.copy()
    if entity != "All":
        p = p[p["ENTITY"] == entity]
    t = test_df.copy()
    if condition != "All":
        t = t[t["Condition"] == condition]

    if p.empty or t.empty:
        return fig, waf_fig, "No data after filtering."

    p["Xr"] = p["X_mm"].round(0).astype(int)
    p["Yr"] = p["Y_mm"].round(0).astype(int)
    t["Xr"] = t["X_mm"].round(0).astype(int)
    t["Yr"] = t["Y_mm"].round(0).astype(int)

    p_avg = p.groupby(["Xr", "Yr"])["Thickness_A"].mean().reset_index()
    t_avg = t.groupby(["Xr", "Yr"])["Thickness_A"].mean().reset_index()
    merged = pd.merge(t_avg, p_avg, on=["Xr", "Yr"], suffixes=("_test", "_por"))

    if merged.empty:
        return fig, waf_fig, "No overlapping XY sites."

    delta = merged["Thickness_A_test"] - merged["Thickness_A_por"]
    stats = (f"Sites: {len(merged)}  |  "
             f"Mean Δ: {delta.mean():+.1f} Å  |  "
             f"σ Δ: {delta.std():.1f} Å  |  "
             f"Max |Δ|: {delta.abs().max():.1f} Å")
    return fig, waf_fig, stats


# ── SPLINE ─────────────────────────────────────────────────────────────────────
@app.callback(
    Output("spline-por",  "figure"),
    Output("spline-test", "figure"),
    Input("spline-por-entity-dd",  "value"),
    Input("spline-test-source-dd", "value"),
    Input("spline-test-cond-dd",   "value"),
)
def update_splines(por_entity, test_source, test_cond):
    p = por_df.copy()
    if por_entity != "All":
        p = p[p["ENTITY"] == por_entity]

    t = test_df.copy()
    if test_source != "All":
        t = t[t["Source"] == test_source]
    if test_cond != "All":
        t = t[t["Condition"] == test_cond]

    por_fig  = spline_figure(p, "POR – Radial Spline Profiles (all wafers)", "Wafer_ID") \
               if not p.empty else go.Figure().update_layout(title="No POR data",
                                                             paper_bgcolor=CARD, font_color="#1a1a2e")
    test_fig = spline_figure(t, "TEST – Radial Spline Profiles (all wafers)", "Wafer_ID") \
               if not t.empty else go.Figure().update_layout(title="No TEST data",
                                                              paper_bgcolor=CARD, font_color="#1a1a2e")
    return por_fig, test_fig


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, port=8050)
