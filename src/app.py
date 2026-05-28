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
# INITIAL CONFIG
# =============================================================================

@st.cache_resource(show_spinner="Downloading artifacts from Google Drive...")
def download_artifacts():
    model_p = Path("outputs/models/random_forest_tuned.pkl")
    parquet_p = Path("outputs/features_test.parquet")
    if model_p.exists() and parquet_p.exists():
        return

    import gdown
    Path("outputs/models").mkdir(parents=True, exist_ok=True)

    if not model_p.exists():
        gdown.download(
            "https://drive.google.com/file/d/1-OapY3PzX2gbcu2-l3gvdmOfKIB8qorJ/view?usp=drive_link",
            str(model_p), quiet=False,
        )
        # Sanity check: pickle файлы начинаются с байта 0x80
        with open(model_p, "rb") as f:
            head = f.read(2)
        if head[:1] != b'\x80':
            model_p.unlink()  # удалить мусор чтобы не закешировался
            raise RuntimeError(
                f"Downloaded model is not a valid pickle (got: {head}). "
                "Check Google Drive sharing permissions for the model file."
            )

    if not parquet_p.exists():
        gdown.download(
            "https://drive.google.com/file/d/1SuW5BlFJcilbZB2ODdN43ApsDxwi73lM/view?usp=drive_link",
            str(parquet_p), quiet=False,
        )
        with open(parquet_p, "rb") as f:
            head = f.read(4)
        if head != b'PAR1':
            parquet_p.unlink()
            raise RuntimeError(
                f"Downloaded parquet is not valid (got: {head}). "
                "Check Google Drive sharing permissions for the parquet file."
            )

download_artifacts()

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

SENSOR_LABELS = {
    "P-TPT":     "Tubing P",
    "T-TPT":     "Tubing T",
    "P-MON-CKP": "Manifold P",
    "T-JUS-CKP": "Downstream T",
}

STAT_LABELS = {
    "mean": "average",
    "std":  "variability",
    "min":  "min",
    "max":  "max",
    "range": "range",
    "trend": "trend"
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

def human_feature_name(col:str) -> str:
    "P-TPT_mean -> Tubing Pressure (Pa) * average"
    for sensor, label in SENSOR_LABELS.items():
        if col.startswith(sensor + "_"):
            stat = col[len(sensor) + 1:]
            return f"{label} * {STAT_LABELS.get(stat, stat)}"
    return col

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown('<div class="header-bar"><h2 style="margin:0;">Settings</h2></div>',
                unsafe_allow_html=True)

    st.divider()
    st.markdown("**Load data**")
    uploaded_parquet = st.file_uploader(
        "Upload features_test.parquet (Optional)",
        type=["parquet"],
        help="Upload a different test feature file to evaluate other wells"
    )

    st.divider()
    st.markdown("**Filter wells by event type**")
    available_types = [0, 3, 4, 7, 9]  # Hardcoded list since these types were used and expected to be used
    selected_types = st.multiselect(
        "Show only wells with:",
        options=available_types,
        default=available_types,
        format_func=lambda x: EVENT_NAMES.get(int(x), '?'),
    )

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

data_dir_path = Path(data_dir)
model_path    = data_dir_path / "models" / "random_forest_tuned.pkl"
features_path = data_dir_path / "features_test.parquet"

# Model is always loaded from disk
if not model_path.exists():
    st.error("**Missing model artifact.** Update the outputs directory in the sidebar.")
    st.code(str(model_path), language=None)
    st.stop()

model = load_model(str(model_path))
explainer = build_explainer(model)

# Features: prefer uploaded file, fall back to disk
if uploaded_parquet is not None:
    import io
    df_test = pd.read_parquet(io.BytesIO(uploaded_parquet.read()))
    feature_cols = [c for c in df_test.columns
                    if c not in ("label", "event_type", "source")
                    and c not in DEAD_FEATURES]
    st.success(f"📂 Using uploaded file: {uploaded_parquet.name} ({len(df_test):,} windows, {df_test['source'].nunique()} wells)")
else:
    if not features_path.exists():
        st.error("No parquet found. Upload one via the sidebar or fix the outputs path.")
        st.stop()
    df_test, feature_cols = load_test_features(str(features_path))
# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    f'<div class="header-bar"><h1 style="margin:0; color:{INK};">'
    "Offshore Well Anomaly Detector</h1></div>",
    unsafe_allow_html=True,
)
with st.expander("ℹ️ What does this system do?", expanded=False):
    st.markdown("""
    This system monitors **offshore oil well sensors** and flags windows of time
    where anomalous behaviour is likely — before it escalates into equipment failure.

    **How to use it:**
    1. Select a well from the dropdown below.
    2. The timeline shows the model's confidence that each 60-second window is anomalous.
    3. Click any point (or use the slider) to inspect *why* the model raised an alert — 
       which sensor drove the prediction.

    **Alert levels:**
    - 🟢 **NORMAL** — no action needed
    - 🟡 **WATCH** — monitor closely, early warning signs detected  
    - 🔴 **ANOMALY** — immediate inspection recommended
    """)
st.divider()

# =============================================================================
# WELL SELECTOR (MAIN AREA)
# =============================================================================

col_a, col_b = st.columns([3, 1])

filtered_df = df_test[df_test["event_type"].isin(selected_types)]
sources = sorted(filtered_df["source"].unique().tolist())
if not sources:
    st.warning("No wells match the selected event types.")
    st.stop()

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
    "Probability timeline</h3></div>",
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

st.plotly_chart(
    fig,
    width="stretch",
    config={
        "scrollZoom": True,
        "displayModeBar": False,
    })

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

n_windows = len(df_well)

# Сброс при смене скважины
if st.session_state.get("last_source") != source:
    st.session_state["slider_idx"] = int(np.argmax(probs))
    st.session_state["number_idx"] = int(np.argmax(probs))
    st.session_state["last_source"] = source

# Инициализация если первый запуск
if "slider_idx" not in st.session_state:
    st.session_state["slider_idx"] = int(np.argmax(probs))
    st.session_state["number_idx"] = int(np.argmax(probs))


# Зажим в допустимый диапазон (на случай смены скважины с большим индексом)
st.session_state["slider_idx"] = min(st.session_state["slider_idx"], n_windows - 1)
st.session_state["number_idx"] = min(st.session_state["number_idx"], n_windows - 1)

# Колбэки
def set_idx(new_idx):
    st.session_state["slider_idx"] = int(new_idx)
    st.session_state["number_idx"] = int(new_idx)

def sync_from_slider():
    st.session_state["number_idx"] = st.session_state["slider_idx"]

def sync_from_number():
    st.session_state["slider_idx"] = st.session_state["number_idx"]

# Кнопки быстрой навигации — ПЕРЕД слайдером
b1, b2, b3, b4 = st.columns(4)
with b1:
    st.button("🔴 Most anomalous",
              on_click=set_idx, args=(int(np.argmax(probs)),),
              width="stretch")
with b2:
    watch_idxs = np.where(tiers == "WATCH")[0]
    st.button("🟡 First WATCH",
              on_click=set_idx,
              args=(int(watch_idxs[0]) if len(watch_idxs) else 0,),
              disabled=len(watch_idxs) == 0,
              width="stretch")
with b3:
    st.button("⚠️ Near threshold",
              on_click=set_idx,
              args=(int(np.argmin(np.abs(probs - threshold_anomaly))),),
              width="stretch")
with b4:
    st.button("🟢 Most normal",
              on_click=set_idx, args=(int(np.argmin(probs)),),
              width="stretch")

# Слайдер + number input (без параметра value, всё через key)
col_slider, col_num = st.columns([5, 1])
with col_slider:
    st.slider(
        "Window index", 0, n_windows - 1,
        key="slider_idx",
        on_change=sync_from_slider,
        help="Drag, click a point on the timeline above, or type an exact index →",
    )
with col_num:
    st.number_input(
        "Exact", 0, n_windows - 1, step=1,
        key="number_idx",
        on_change=sync_from_number,
    )


idx = st.session_state["slider_idx"]

# Подготовка данных для текущего окна
x        = X_well[idx].reshape(1, -1)
proba    = float(probs[idx])
tier     = tiers[idx]
color    = TIER_COLORS[tier]
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
    st.plotly_chart(gauge, width="stretch")

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
    df_shap = pd.DataFrame({
        "feature": [human_feature_name(c) for c in feature_cols],
        "shap": shap_vals
    })
    df_shap["abs"] = df_shap["shap"].abs()
    df_shap = df_shap.sort_values("abs", ascending=True).tail(10)

    fig_shap = go.Figure(go.Bar(
        x=df_shap["shap"],
        y=df_shap["feature"],
        orientation="h",
        marker=dict(color=[TIER_COLORS["ANOMALY"] if v > 0 else "#3B82F6" for v in df_shap["shap"]]),
        text=[f"{v:+.4f}" for v in df_shap["shap"]],
        textposition="auto",
        insidetextanchor="end",
        cliponaxis=False,
    ))
    fig_shap.add_vline(x=0, line_color=INK, line_width=1)
    fig_shap.update_layout(
        height=400, margin=dict(l=140, r=40, t=10, b=20),
        plot_bgcolor="white",
        xaxis_title="SHAP value",
        font=dict(family="Calibri, Arial"),
    )
    fig_shap.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig_shap.update_yaxes(showgrid=False)
    st.plotly_chart(fig_shap, width="stretch")
# =============================================================================
# RAW FEATURE VALUES
# =============================================================================

with st.expander("📋 Raw feature values for this window"):
    row = df_well.iloc[idx]
    feat_df = pd.DataFrame({
        "Feature": feature_cols,
        "Scaled value": [row[c] for c in feature_cols],
    })
    st.dataframe(feat_df, width="stretch", hide_index=True, height=300)

# =============================================================================
# FOOTER
# =============================================================================

st.caption(
    "Selected model: RF Tuned (Optuna, 20 trials, TPE sampler)"
    "Features: 24 (4 sensors × 6 stats)"
    "Split: GroupShuffleSplit by source file (798 train / 200 test)"
    "Author: Timur Kasymbekov, AITU, June 2026"
)