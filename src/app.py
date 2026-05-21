"""
Offshore Well Anomaly Detector — diploma defense demo.

Loads the RF Tuned model + features_test.parquet, shows a per-well timeline of
window-level anomaly probabilities with the three-tier (NORMAL/WATCH/ANOMALY)
recommendation, and SHAP explanations for any selected window.

"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# =============================================================================
# CONSTANTS
# =============================================================================

DEAD_FEATURES = ["P-PDG_mean", "P-PDG_std", "P-PDG_min",
                 "P-PDG_max", "P-PDG_range", "P-PDG_trend"]

EVENT_NAMES = {
    0: "Normal Operation",
    3: "Severe Slugging",
    4: "Flow Instability",
    7: "Scaling in PCK",
    9: "Hydrate in Service Line",
}

TIER_COLORS = {
    "NORMAL":  "#059669",
    "WATCH":   "#D97706",
    "ANOMALY": "#DC2626",
}

TIER_ICONS = {
    "NORMAL":  "🟢",
    "WATCH":   "🟡",
    "ANOMALY": "🔴",
}

ACCENT = "#D97706"
INK    = "#0F172A"

# =============================================================================
# PAGE CONFIG + CSS
# =============================================================================

st.set_page_config(
    page_title="Offshore Anomaly Detector",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  .stApp {{ background: #FFFFFF; }}
  .tier-badge {{
    padding: 24px;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 4px;
    color: white;
    text-align: center;
    border-radius: 10px;
    margin: 12px 0;
  }}
  .stMetric {{ background: #F8FAFC; padding: 12px; border-radius: 8px; }}
  div[data-testid="stMetricValue"] {{ color: {INK}; }}
  hr {{ border-color: #E2E8F0 !important; }}
  .header-bar {{
    border-left: 5px solid {ACCENT};
    padding-left: 14px;
    margin-bottom: 6px;
  }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CACHED LOADERS
# =============================================================================

@st.cache_resource(show_spinner="Loading RF Tuned model...")
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Building SHAP explainer...")
def build_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data(show_spinner="Loading test features...")
def load_test_features(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns
                    if c not in ("label", "event_type", "source")
                    and c not in DEAD_FEATURES]
    return df, feature_cols

def get_shap_values(explainer, x: np.ndarray) -> np.ndarray:
    """Robust to shap version: returns 1D array of length n_features."""
    raw = explainer.shap_values(x)
    if isinstance(raw, list):
        arr = np.array(raw[1])
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        arr = raw[:, :, 1]
    else:
        arr = np.asarray(raw)
    return arr.reshape(-1) if arr.ndim == 1 else arr[0]

def assign_tier(p: float, t_normal: float, t_anomaly: float) -> str:
    if p < t_normal:   return "NORMAL"
    if p < t_anomaly:  return "WATCH"
    return "ANOMALY"

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown('<div class="header-bar"><h2 style="margin:0;">Settings</h2></div>',
                unsafe_allow_html=True)

    st.divider()

    data_dir = st.text_input(
        "Outputs directory",
        value="./outputs",
        help="Folder containing features_test.parquet and models/random_forest_tuned.pkl",
    )

    st.divider()
    st.markdown("**Recommendation thresholds**")
    threshold_normal = st.slider(
        "NORMAL → WATCH", 0.05, 0.95, 0.30, 0.01,
        help="Below this: NORMAL. Above this: WATCH or ANOMALY.",
    )
    threshold_anomaly = st.slider(
        "WATCH → ANOMALY", 0.05, 0.95, 0.50, 0.01,
        help="Above this: ANOMALY.",
    )
    if threshold_anomaly <= threshold_normal:
        st.warning("ANOMALY threshold must be higher than NORMAL threshold.")
        st.stop()

    st.divider()
    st.markdown("**Model**")
    st.code("RF Tuned\nn_est=257, depth=14\nROC-AUC ≈ 0.9633\nFN = 7,787", language=None)

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

data_dir_path  = Path(data_dir)
model_path     = data_dir_path / "models" / "random_forest_tuned.pkl"
features_path  = data_dir_path / "features_test.parquet"

missing = [p for p in [model_path, features_path] if not p.exists()]
if missing:
    st.error("**Missing artifacts.** Update the outputs directory in the sidebar.")
    for p in missing:
        st.code(str(p), language=None)
    st.stop()

model = load_model(str(model_path))
df_test, feature_cols = load_test_features(str(features_path))
explainer = build_explainer(model)

# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    f'<div class="header-bar"><h1 style="margin:0; color:{INK};">'
    "Offshore Well Anomaly Detector</h1></div>",
    unsafe_allow_html=True,
)
st.caption(
    "Machine-learning recommendation system for offshore oil-well sensor data. "
    "Built on the Petrobras 3W dataset. Random Forest (Optuna-tuned) + SHAP "
    "TreeExplainer + three-tier alerting."
)
st.divider()

# =============================================================================
# WELL SELECTOR (MAIN AREA)
# =============================================================================

col_a, col_b = st.columns([3, 1])
sources = sorted(df_test["source"].unique().tolist())
with col_a:
    source = st.selectbox(f"Select a well from the test set ({len(sources)} wells)", sources)
with col_b:
    event_type = df_test.loc[df_test["source"] == source, "event_type"].iloc[0]
    st.metric("Event type", f"{event_type} — {EVENT_NAMES.get(int(event_type), '?')}")

df_well = df_test[df_test["source"] == source].reset_index(drop=True)
X_well  = df_well[feature_cols].values
probs   = model.predict_proba(X_well)[:, 1]
tiers   = np.array([assign_tier(p, threshold_normal, threshold_anomaly) for p in probs])

# =============================================================================
# TIER COUNTS
# =============================================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total windows",  f"{len(df_well):,}")
c2.metric("🟢 NORMAL",      int((tiers == "NORMAL").sum()))
c3.metric("🟡 WATCH",       int((tiers == "WATCH").sum()))
c4.metric("🔴 ANOMALY",     int((tiers == "ANOMALY").sum()))

# =============================================================================
# TIMELINE
# =============================================================================

st.markdown(
    f'<div class="header-bar"><h3 style="margin:0;">'
    "Probability timeline · click any point to inspect</h3></div>",
    unsafe_allow_html=True,
)

fig = go.Figure()
for tier in ["NORMAL", "WATCH", "ANOMALY"]:
    mask = tiers == tier
    if not mask.any():
        continue
    fig.add_trace(go.Scatter(
        x=np.where(mask)[0],
        y=probs[mask],
        mode="markers",
        name=f"{TIER_ICONS[tier]} {tier}",
        marker=dict(color=TIER_COLORS[tier], size=7, line=dict(width=0)),
        customdata=np.where(mask)[0],
        hovertemplate="Window %{customdata}<br>p = %{y:.3f}<extra></extra>",
    ))

# Threshold lines
fig.add_hline(y=threshold_normal,  line_dash="dash", line_color="#94A3B8",
              annotation_text=f"NORMAL → WATCH: {threshold_normal:.2f}",
              annotation_position="right",
              annotation_font_size=10)
fig.add_hline(y=threshold_anomaly, line_dash="dash", line_color="#94A3B8",
              annotation_text=f"WATCH → ANOMALY: {threshold_anomaly:.2f}",
              annotation_position="right",
              annotation_font_size=10)

# True-label band if available
if "label" in df_well.columns:
    truth = df_well["label"].values
    anomaly_runs = []
    in_run, start = False, 0
    for i, v in enumerate(truth):
        if v == 1 and not in_run:
            in_run, start = True, i
        elif v == 0 and in_run:
            anomaly_runs.append((start, i - 1))
            in_run = False
    if in_run:
        anomaly_runs.append((start, len(truth) - 1))
    for r0, r1 in anomaly_runs:
        fig.add_vrect(x0=r0 - 0.5, x1=r1 + 0.5,
                      fillcolor="#FCA5A5", opacity=0.15, layer="below",
                      line_width=0)

fig.update_layout(
    xaxis_title="Window index",
    yaxis_title="Anomaly probability",
    yaxis_range=[-0.02, 1.02],
    height=420,
    margin=dict(l=0, r=0, t=10, b=0),
    plot_bgcolor="white",
    legend=dict(orientation="h", y=1.12, x=0),
    font=dict(family="Calibri, Arial"),
)
fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")

st.plotly_chart(fig, use_container_width=True)
if "label" in df_well.columns:
    st.caption("Pink-shaded regions = ground-truth anomaly windows from the test labels.")

# =============================================================================
# WINDOW INSPECTION
# =============================================================================

st.divider()
st.markdown(
    f'<div class="header-bar"><h3 style="margin:0;">Inspect a single window</h3></div>',
    unsafe_allow_html=True,
)

idx = st.slider(
    "Window index", 0, len(df_well) - 1,
    value=int(np.argmax(probs)),   # default to highest-probability window
    help="Slide through the well's windows. The default is the most-anomalous window.",
)
x       = X_well[idx].reshape(1, -1)
proba   = float(probs[idx])
tier    = tiers[idx]
color   = TIER_COLORS[tier]
true_lbl = int(df_well.iloc[idx]["label"]) if "label" in df_well.columns else None

left, right = st.columns([1, 1])

with left:
    st.markdown(
        f'<div class="tier-badge" style="background:{color};">{TIER_ICONS[tier]} {tier}</div>',
        unsafe_allow_html=True,
    )

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba,
        number=dict(valueformat=".3f", font=dict(size=42, color=INK)),
        title=dict(text="Anomaly probability", font=dict(size=14, color="#64748B")),
        gauge=dict(
            axis=dict(range=[0, 1], tickwidth=1, tickcolor="#94A3B8"),
            bar=dict(color=color, thickness=0.25),
            bgcolor="white",
            borderwidth=0,
            steps=[
                dict(range=[0, threshold_normal],  color="#DCFCE7"),
                dict(range=[threshold_normal, threshold_anomaly], color="#FED7AA"),
                dict(range=[threshold_anomaly, 1], color="#FECACA"),
            ],
            threshold=dict(line=dict(color=INK, width=2), thickness=0.85, value=proba),
        ),
    ))
    gauge.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0),
                        font=dict(family="Calibri, Arial"))
    st.plotly_chart(gauge, use_container_width=True)

    if true_lbl is not None:
        truth_text = ("🔴 ANOMALY" if true_lbl == 1 else "🟢 NORMAL")
        correct = (true_lbl == 1 and tier != "NORMAL") or (true_lbl == 0 and tier == "NORMAL")
        st.markdown(
            f"**Ground truth:** {truth_text} &nbsp;&nbsp; "
            f"{'✅ correctly flagged' if correct else '⚠️ mismatch'}"
        )

with right:
    st.markdown("**SHAP — why this prediction?**")
    st.caption("Red bars push toward ANOMALY · Blue bars push toward NORMAL")
    shap_vals = get_shap_values(explainer, x)
    df_shap = pd.DataFrame({"feature": feature_cols, "shap": shap_vals})
    df_shap["abs"] = df_shap["shap"].abs()
    df_shap = df_shap.sort_values("abs", ascending=True).tail(10)

    fig_shap = go.Figure(go.Bar(
        x=df_shap["shap"],
        y=df_shap["feature"],
        orientation="h",
        marker=dict(color=[TIER_COLORS["ANOMALY"] if v > 0 else "#3B82F6" for v in df_shap["shap"]]),
        text=[f"{v:+.4f}" for v in df_shap["shap"]],
        textposition="outside",
        cliponaxis=False,
    ))
    fig_shap.add_vline(x=0, line_color=INK, line_width=1)
    fig_shap.update_layout(
        height=380, margin=dict(l=0, r=20, t=10, b=20),
        plot_bgcolor="white",
        xaxis_title="SHAP value",
        font=dict(family="Calibri, Arial"),
    )
    fig_shap.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig_shap.update_yaxes(showgrid=False)
    st.plotly_chart(fig_shap, use_container_width=True)

# =============================================================================
# RAW FEATURE VALUES
# =============================================================================

with st.expander("📋 Raw feature values for this window"):
    row = df_well.iloc[idx]
    feat_df = pd.DataFrame({
        "Feature": feature_cols,
        "Scaled value": [row[c] for c in feature_cols],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True, height=300)

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption(
    "Selected model: RF Tuned (Optuna, 20 trials, TPE sampler) · "
    "Features: 24 (4 sensors × 6 stats) · "
    "Split: GroupShuffleSplit by source file (798 train / 200 test) · "
    "Author: Timur Kasymbekov, AITU, June 2026"
)