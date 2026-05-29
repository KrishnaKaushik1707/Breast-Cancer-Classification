import streamlit as st
import numpy as np
import pickle
import pandas as pd
import altair as alt
from datetime import datetime

# Load trained model
model = pickle.load(open('trained_model.sav', 'rb'))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "Breast Cancer Wisconsin Dataset"
NUM_FEATURES = 30
NUM_CLASSES = 2
TOTAL_FEATURES = 30

FEATURE_NAMES = [
    'Mean Radius',
    'Mean Texture',
    'Mean Perimeter',
    'Mean Area',
    'Mean Smoothness',
    'Mean Compactness',
    'Mean Concavity',
    'Mean Concave Points',
    'Mean Symmetry',
    'Mean Fractal Dimension',
    'Radius Error',
    'Texture Error',
    'Perimeter Error',
    'Area Error',
    'Smoothness Error',
    'Compactness Error',
    'Concavity Error',
    'Concave Points Error',
    'Symmetry Error',
    'Fractal Dimension Error',
    'Worst Radius',
    'Worst Texture',
    'Worst Perimeter',
    'Worst Area',
    'Worst Smoothness',
    'Worst Compactness',
    'Worst Concavity',
    'Worst Concave Points',
    'Worst Symmetry',
    'Worst Fractal Dimension',
]

feature_names = FEATURE_NAMES

FEATURE_CATEGORIES = {
    "Mean Measurements": {
        "icon": "📊",
        "description": "Average values computed across all cells in the sample.",
        "features": FEATURE_NAMES[0:10],
    },
    "Error Measurements": {
        "icon": "📈",
        "description": "Standard error of each mean measurement.",
        "features": FEATURE_NAMES[10:20],
    },
    "Worst Measurements": {
        "icon": "🔺",
        "description": "Largest (worst) values observed among all cells.",
        "features": FEATURE_NAMES[20:30],
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_estimator(mdl):
    if hasattr(mdl, "best_estimator_"):
        return mdl.best_estimator_
    return mdl


def get_algorithm_name(mdl):
    est = get_estimator(mdl)
    name = type(est).__name__
    if hasattr(mdl, "best_estimator_"):
        return f"{name} (GridSearchCV)"
    return name


def get_training_accuracy(mdl):
    if hasattr(mdl, "best_score_"):
        return round(mdl.best_score_ * 100, 2)
    return None


def get_prediction_status():
    if st.session_state.get("prediction") is not None:
        return "Complete", "🟢"
    filled = st.session_state.get("filled_count", 0)
    if filled == TOTAL_FEATURES:
        return "Ready", "🟡"
    return "Awaiting Input", "⚪"


def count_completed_features(features):
    return sum(1 for v in features if v > 0.0)


def get_confidence(mdl, input_data):
    try:
        probability = mdl.predict_proba(input_data)
        return float(np.max(probability) * 100)
    except Exception:
        return None


def run_prediction(mdl, features):
    input_data = np.array(features).reshape(1, -1)
    prediction = mdl.predict(input_data)
    confidence = get_confidence(mdl, input_data)
    return prediction, confidence, input_data


def get_top_contributing_features(mdl, features, feature_names, top_n=5):
    """Return top influential features using SHAP or coefficient-based fallback."""
    input_data = np.array(features).reshape(1, -1)

    try:
        import shap

        est = get_estimator(mdl)
        explainer = shap.Explainer(est, input_data)
        shap_values = explainer(input_data)
        values = shap_values.values[0]
        if values.ndim > 1:
            values = values[:, 1] if values.shape[1] > 1 else values[:, 0]
        ranked = sorted(
            zip(feature_names, values, np.abs(values)),
            key=lambda x: x[2],
            reverse=True,
        )[:top_n]
        return [(name, float(val), float(abs_val), "SHAP") for name, val, abs_val in ranked]
    except Exception:
        pass

    est = get_estimator(mdl)
    if hasattr(est, "coef_"):
        coef = est.coef_.flatten()
        contributions = coef * np.array(features)
        ranked = sorted(
            zip(feature_names, contributions, np.abs(contributions)),
            key=lambda x: x[2],
            reverse=True,
        )[:top_n]
        return [(name, float(val), float(abs_val), "Coefficient") for name, val, abs_val in ranked]

    if hasattr(est, "feature_importances_"):
        importances = est.feature_importances_
        weighted = importances * np.array(features)
        ranked = sorted(
            zip(feature_names, weighted, np.abs(weighted)),
            key=lambda x: x[2],
            reverse=True,
        )[:top_n]
        return [(name, float(val), float(abs_val), "Feature Importance") for name, val, abs_val in ranked]

    return []


def prediction_label(prediction):
    if prediction == 0:
        return "Malignant", "malignant", "⚠️"
    return "Benign", "benign", "✅"


def render_result_card(prediction, confidence, timestamp):
    label, css_class, icon = prediction_label(prediction)
    conf_html = (
        f"""
        <div class="result-stat">
            <span class="result-stat-label">Confidence</span>
            <span class="result-stat-value">{confidence:.2f}%</span>
        </div>
        """
        if confidence is not None
        else ""
    )
    ts_html = (
        f"""
        <div class="result-stat">
            <span class="result-stat-label">Predicted at</span>
            <span class="result-stat-value result-stat-value-sm">{timestamp}</span>
        </div>
        """
        if timestamp
        else ""
    )
    stats_html = (
        f'<div class="result-stats">{conf_html}{ts_html}</div>'
        if conf_html or ts_html
        else ""
    )
    st.markdown(
        f"""
        <div class="result-card {css_class} animate-in">
            <div class="result-eyebrow">Diagnosis Result</div>
            <div class="result-icon-ring">{icon}</div>
            <div class="result-label">{label}</div>
            <div class="result-sub">The model classifies this sample as <strong>{label.lower()}</strong>.</div>
            {stats_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_completion_tracker(filled_count, total=TOTAL_FEATURES):
    pct = int((filled_count / total) * 100)
    st.markdown(
        f"""
        <div class="progress-wrap animate-in">
            <div class="progress-label">
                <span class="progress-title">Input Completion</span>
                <span class="progress-count">{filled_count} / {total} Features Completed</span>
            </div>
            <div class="progress-meta">
                <span class="progress-pct">{pct}%</span>
                <span class="progress-hint">{"Ready to predict" if pct == 100 else "Keep filling inputs"}</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{pct}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _light_chart(chart):
    """Ensure Altair charts render with a light background and readable text."""
    return chart.configure_view(
        fill="#ffffff",
        stroke="#e2e8f0",
        cornerRadius=10,
    ).configure_axis(
        labelColor="#334155",
        titleColor="#0f172a",
        gridColor="#e2e8f0",
    ).configure_title(
        color="#0f172a",
        fontSize=14,
        fontWeight="bold",
    )


def render_confidence_visualization(confidence):
    st.markdown(
        f"""
        <div class="conf-panel animate-in">
            <div class="conf-value-display">{confidence:.1f}<span class="conf-unit">%</span></div>
            <div class="conf-caption">Model confidence for this prediction</div>
            <div class="conf-bar-track">
                <div class="conf-bar-fill" style="width:{confidence:.1f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explanation_chart(top_features, method):
    df = pd.DataFrame(top_features, columns=["feature", "contribution", "abs_contribution", "method"])
    df["direction"] = df["contribution"].apply(lambda x: "Toward Benign" if x > 0 else "Toward Malignant")

    chart = _light_chart(
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("abs_contribution:Q", title="Influence Score"),
            y=alt.Y("feature:N", sort="-x", title=""),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Toward Benign", "Toward Malignant"],
                    range=["#059669", "#dc2626"],
                ),
                legend=alt.Legend(title="Impact Direction", labelColor="#334155", titleColor="#0f172a"),
            ),
            tooltip=["feature", "contribution", "direction"],
        )
        .properties(height=260, title=f"Top 5 Influential Features ({method})", background="#ffffff")
    )
    st.altair_chart(chart, use_container_width=True)


def render_model_metrics_grid(algorithm, accuracy, status_text, status_icon):
    accuracy_display = f"{accuracy}%" if accuracy is not None else "N/A"
    accuracy_sub = "Cross-validation best score" if accuracy is not None else "Not available"
    st.markdown(
        f"""
        <div class="metrics-grid">
            <div class="metric-card blue">
                <div class="metric-card-header">
                    <span class="metric-label">Algorithm</span>
                    <div class="metric-icon blue">🤖</div>
                </div>
                <div class="metric-value metric-value-sm">{algorithm}</div>
                <div class="metric-sub">Auto-detected from model</div>
            </div>
            <div class="metric-card teal">
                <div class="metric-card-header">
                    <span class="metric-label">Dataset</span>
                    <div class="metric-icon teal">🧬</div>
                </div>
                <div class="metric-value metric-value-sm">Wisconsin</div>
                <div class="metric-sub">{DATASET_NAME}</div>
            </div>
            <div class="metric-card navy">
                <div class="metric-card-header">
                    <span class="metric-label">Features / Classes</span>
                    <div class="metric-icon navy">📐</div>
                </div>
                <div class="metric-value">{NUM_FEATURES} / {NUM_CLASSES}</div>
                <div class="metric-sub">Input features · Output classes</div>
            </div>
            <div class="metric-card purple">
                <div class="metric-card-header">
                    <span class="metric-label">CV Accuracy</span>
                    <div class="metric-icon purple">📈</div>
                </div>
                <div class="metric-value">{accuracy_display}</div>
                <div class="metric-sub">{accuracy_sub}</div>
            </div>
        </div>
        <div class="metrics-grid metrics-grid-2">
            <div class="metric-card green">
                <div class="metric-card-header">
                    <span class="metric-label">Prediction Status</span>
                    <div class="metric-icon green">🎯</div>
                </div>
                <div class="metric-value metric-value-sm">{status_icon} {status_text}</div>
                <div class="metric-sub">Live session status</div>
            </div>
            <div class="metric-card orange">
                <div class="metric-card-header">
                    <span class="metric-label">Model Engine</span>
                    <div class="metric-icon orange">⚙️</div>
                </div>
                <div class="metric-value metric-value-sm">Scikit-learn</div>
                <div class="metric-sub">Production-ready ML pipeline</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_metric(label, value, icon):
    st.markdown(
        f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-icon">{icon}</div>
            <div>
                <div class="sidebar-metric-label">{label}</div>
                <div class="sidebar-metric-value">{value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.prediction_time = None
    st.session_state.top_features = []
    st.session_state.explanation_method = None

# Model metadata (computed once)
ALGORITHM_NAME = get_algorithm_name(model)
TRAINING_ACCURACY = get_training_accuracy(model)

# ---------------------------------------------------------------------------
# Custom CSS – premium SaaS healthcare dashboard
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #0c1e3a;
        --navy-mid: #152d54;
        --blue: #2563eb;
        --blue-light: #3b82f6;
        --teal: #0d9488;
        --teal-light: #14b8a6;
        --surface: #ffffff;
        --bg: #eef2f7;
        --bg-subtle: #e2e8f0;
        --border: #dde4ed;
        --border-light: #edf1f7;
        --text: #0f172a;
        --text-secondary: #334155;
        --text-muted: #64748b;
        --success: #059669;
        --success-bg: #ecfdf5;
        --danger: #dc2626;
        --danger-bg: #fef2f2;
        --shadow-xs: 0 1px 2px rgba(15,23,42,0.04);
        --shadow-sm: 0 2px 8px rgba(15,23,42,0.06);
        --shadow-md: 0 8px 24px rgba(15,23,42,0.08);
        --shadow-lg: 0 16px 48px rgba(15,23,42,0.10);
        --shadow-card: 0 1px 3px rgba(15,23,42,0.05), 0 4px 16px rgba(15,23,42,0.04);
        --radius: 18px;
        --radius-sm: 12px;
        --radius-xs: 8px;
        --ease: cubic-bezier(0.4, 0, 0.2, 1);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .stApp {
        background: linear-gradient(180deg, #eef2f7 0%, #f8fafc 40%, #eef2f7 100%);
    }

    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.55; transform: scale(0.92); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes ringPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(37,99,235,0.15); }
        50%       { box-shadow: 0 0 0 10px rgba(37,99,235,0); }
    }
    .animate-in {
        animation: fadeInUp 0.45s var(--ease) both;
    }
    .metrics-grid .metric-card:nth-child(1) { animation: fadeInUp 0.4s var(--ease) 0.05s both; }
    .metrics-grid .metric-card:nth-child(2) { animation: fadeInUp 0.4s var(--ease) 0.10s both; }
    .metrics-grid .metric-card:nth-child(3) { animation: fadeInUp 0.4s var(--ease) 0.15s both; }
    .metrics-grid .metric-card:nth-child(4) { animation: fadeInUp 0.4s var(--ease) 0.20s both; }

    /* ── Top nav bar ── */
    .saas-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-light);
        border-radius: var(--radius);
        padding: 1rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-card);
        transition: box-shadow 0.25s var(--ease), transform 0.25s var(--ease);
    }
    .saas-nav:hover { box-shadow: var(--shadow-md); }
    .saas-nav-brand { display: flex; align-items: center; gap: 0.75rem; }
    .saas-nav-logo {
        width: 42px; height: 42px;
        background: linear-gradient(135deg, var(--blue) 0%, var(--teal) 100%);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 4px 12px rgba(37,99,235,0.3);
    }
    .saas-nav-title { font-size: 1.05rem; font-weight: 700; color: var(--navy); line-height: 1.2; }
    .saas-nav-sub { font-size: 0.72rem; color: var(--text-muted); font-weight: 500; }
    .saas-nav-status {
        display: flex; align-items: center; gap: 0.45rem;
        background: var(--success-bg); color: var(--success);
        font-size: 0.78rem; font-weight: 600;
        padding: 0.4rem 0.9rem; border-radius: 20px;
        border: 1px solid #a7f3d0;
    }
    .status-dot {
        width: 7px; height: 7px; background: var(--success);
        border-radius: 50%; animation: pulse 2s infinite;
    }

    /* ── Hero header ── */
    .hero-banner {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 45%, #1a4a8a 100%);
        border-radius: var(--radius);
        padding: 2.75rem 3rem;
        margin-bottom: 1.75rem;
        position: relative; overflow: hidden;
        box-shadow: var(--shadow-lg);
    }
    .hero-banner::before {
        content: ''; position: absolute; top: -60%; right: -8%;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(37,99,235,0.25) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: ''; position: absolute; bottom: -40%; left: 20%;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(13,148,136,0.2) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-content { position: relative; z-index: 1; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 0.3rem 0.85rem;
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.06em; text-transform: uppercase;
        color: rgba(255,255,255,0.9);
        margin-bottom: 0.85rem;
    }
    .hero-banner h1 {
        font-size: 2.25rem; font-weight: 800; color: #ffffff;
        margin: 0 0 0.75rem 0; letter-spacing: -0.035em; line-height: 1.15;
    }
    .hero-banner p {
        font-size: 1.02rem; color: rgba(255,255,255,0.80);
        margin: 0; line-height: 1.7; max-width: 580px; font-weight: 400;
    }

    /* ── Section label ── */
    .section-eyebrow {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--text-muted);
        margin-bottom: 0.75rem;
    }

    /* ── Metric cards ── */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.125rem;
        margin-bottom: 1.125rem;
    }
    .metrics-grid-2 { grid-template-columns: repeat(2, 1fr); margin-bottom: 2rem; }
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: var(--radius);
        padding: 1.35rem 1.4rem;
        box-shadow: var(--shadow-card);
        transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease), border-color 0.25s var(--ease);
        position: relative; overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
        border-color: var(--border);
    }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        border-radius: var(--radius) var(--radius) 0 0;
    }
    .metric-card.blue::before { background: linear-gradient(90deg, var(--blue), var(--blue-light)); }
    .metric-card.teal::before { background: linear-gradient(90deg, var(--teal), var(--teal-light)); }
    .metric-card.navy::before { background: linear-gradient(90deg, var(--navy), var(--navy-mid)); }
    .metric-card.purple::before { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
    .metric-card.green::before { background: linear-gradient(90deg, #059669, #34d399); }
    .metric-card.orange::before { background: linear-gradient(90deg, #ea580c, #fb923c); }
    .metric-card-header {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 0.65rem;
    }
    .metric-icon {
        width: 42px; height: 42px; border-radius: 11px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.35rem; line-height: 1;
        flex-shrink: 0;
        border: 1px solid rgba(255,255,255,0.75);
        box-shadow: 0 2px 8px rgba(15,23,42,0.10);
        filter: none;
        opacity: 1;
    }
    .metric-icon.blue  { background: linear-gradient(135deg, #bfdbfe 0%, #60a5fa 100%); }
    .metric-icon.teal  { background: linear-gradient(135deg, #99f6e4 0%, #2dd4bf 100%); }
    .metric-icon.navy  { background: linear-gradient(135deg, #cbd5e1 0%, #64748b 100%); }
    .metric-icon.purple { background: linear-gradient(135deg, #ddd6fe 0%, #a78bfa 100%); }
    .metric-icon.green { background: linear-gradient(135deg, #a7f3d0 0%, #34d399 100%); }
    .metric-icon.orange { background: linear-gradient(135deg, #fed7aa 0%, #fb923c 100%); }
    .metric-label {
        font-size: 0.75rem; font-weight: 600; color: var(--text-muted);
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    .metric-value {
        font-size: 1.65rem; font-weight: 800; color: var(--navy);
        line-height: 1.15; letter-spacing: -0.02em;
    }
    .metric-value-sm { font-size: 1.05rem; font-weight: 700; }
    .metric-sub {
        font-size: 0.78rem; color: var(--text-muted);
        margin-top: 0.35rem; font-weight: 500;
    }

    /* ── Dashboard cards ── */
    .dash-card {
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: var(--radius);
        padding: 1.5rem 1.65rem;
        box-shadow: var(--shadow-card);
        margin-bottom: 1.125rem;
        transition: box-shadow 0.25s var(--ease), transform 0.25s var(--ease), border-color 0.25s var(--ease);
    }
    .dash-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--border);
    }
    .dash-card-header {
        display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;
    }
    .dash-card-icon {
        width: 40px; height: 40px; border-radius: var(--radius-xs);
        background: linear-gradient(135deg, #bfdbfe 0%, #60a5fa 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.15rem; line-height: 1;
        border: 1px solid rgba(255,255,255,0.75);
        box-shadow: 0 2px 6px rgba(15,23,42,0.08);
    }
    .dash-card h3 {
        font-size: 1.02rem; font-weight: 700; color: var(--navy);
        margin: 0; letter-spacing: -0.01em;
    }
    .dash-card p {
        font-size: 0.875rem; color: var(--text-muted);
        margin: 0; line-height: 1.65;
    }

    /* ── Progress bar ── */
    .progress-wrap {
        margin-top: 1.25rem;
        padding: 1.25rem 1.4rem;
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: var(--radius);
        box-shadow: var(--shadow-card);
    }
    .progress-label {
        display: flex; justify-content: space-between; align-items: baseline;
        margin-bottom: 0.5rem;
    }
    .progress-title {
        font-size: 0.875rem; font-weight: 700; color: var(--navy);
        letter-spacing: -0.01em;
    }
    .progress-count {
        font-size: 0.8125rem; font-weight: 600; color: var(--text-muted);
    }
    .progress-meta {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 0.65rem;
    }
    .progress-pct {
        font-size: 1.5rem; font-weight: 800; color: var(--navy);
        letter-spacing: -0.03em; line-height: 1;
    }
    .progress-hint {
        font-size: 0.78rem; font-weight: 500; color: var(--text-muted);
    }
    .progress-track {
        height: 8px; background: var(--bg-subtle);
        border-radius: 99px; overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--blue) 0%, var(--teal) 100%);
        background-size: 200% auto;
        border-radius: 99px;
        transition: width 0.55s var(--ease);
        animation: shimmer 3s linear infinite;
    }

    /* ── Info list ── */
    .info-list { list-style: none; padding: 0; margin: 0.75rem 0 0 0; }
    .info-list li {
        display: flex; align-items: flex-start; gap: 0.6rem;
        padding: 0.55rem 0; border-bottom: 1px solid var(--border);
        font-size: 0.84rem; color: var(--text-muted); line-height: 1.45;
    }
    .info-list li:last-child { border-bottom: none; }
    .info-list .step-num {
        min-width: 22px; height: 22px;
        background: linear-gradient(135deg, var(--blue), var(--teal));
        color: white; border-radius: 6px;
        font-size: 0.7rem; font-weight: 700;
        display: flex; align-items: center; justify-content: center;
        margin-top: 1px;
    }

    /* ── Model spec table ── */
    .spec-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.55rem 0; border-bottom: 1px solid var(--border);
        font-size: 0.84rem;
    }
    .spec-row:last-child { border-bottom: none; }
    .spec-key { color: var(--text-muted); font-weight: 500; }
    .spec-val {
        color: var(--navy); font-weight: 600;
        background: var(--bg); padding: 0.15rem 0.55rem;
        border-radius: 6px; font-size: 0.78rem;
    }

    /* ── Result cards ── */
    .result-card {
        border-radius: var(--radius);
        padding: 2.25rem 1.75rem 2rem;
        text-align: center;
        box-shadow: var(--shadow-md);
        position: relative; overflow: hidden;
        transition: box-shadow 0.3s var(--ease), transform 0.3s var(--ease);
    }
    .result-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-2px); }
    .result-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 5px;
    }
    .result-card::after {
        content: ''; position: absolute;
        top: -40%; right: -20%; width: 200px; height: 200px;
        border-radius: 50%; opacity: 0.12; pointer-events: none;
    }
    .result-card.benign {
        background: linear-gradient(165deg, #f0fdf8 0%, #d1fae5 60%, #bbf7d0 100%);
        border: 1px solid rgba(110,231,183,0.6);
    }
    .result-card.benign::before { background: linear-gradient(90deg, #059669, #34d399); }
    .result-card.benign::after { background: radial-gradient(circle, #059669, transparent); }
    .result-card.malignant {
        background: linear-gradient(165deg, #fff5f5 0%, #fecaca 60%, #fca5a5 100%);
        border: 1px solid rgba(252,165,165,0.6);
    }
    .result-card.malignant::before { background: linear-gradient(90deg, #dc2626, #f87171); }
    .result-card.malignant::after { background: radial-gradient(circle, #dc2626, transparent); }
    .result-card.empty {
        background: var(--surface); border: 2px dashed var(--border);
        padding: 2.75rem 1.75rem;
    }
    .result-card.empty::before, .result-card.empty::after { display: none; }
    .result-eyebrow {
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--text-muted);
        margin-bottom: 1rem;
    }
    .result-icon-ring {
        width: 80px; height: 80px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 2.1rem; margin: 0 auto 1rem;
        position: relative; z-index: 1;
    }
    .result-card.benign .result-icon-ring {
        background: rgba(5,150,105,0.12);
        box-shadow: 0 0 0 8px rgba(5,150,105,0.07);
        animation: ringPulse 2.5s ease infinite;
    }
    .result-card.malignant .result-icon-ring {
        background: rgba(220,38,38,0.12);
        box-shadow: 0 0 0 8px rgba(220,38,38,0.07);
        animation: ringPulse 2.5s ease infinite;
    }
    .result-card.empty .result-icon-ring {
        background: var(--bg); box-shadow: none; font-size: 1.6rem; animation: none;
    }
    .result-label {
        font-size: 1.85rem; font-weight: 800;
        letter-spacing: -0.035em; margin: 0; position: relative; z-index: 1;
    }
    .result-card.benign .result-label { color: #065f46; }
    .result-card.malignant .result-label { color: #991b1b; }
    .result-card.empty .result-label { color: var(--text-muted); font-size: 1.05rem; font-weight: 600; }
    .result-sub {
        font-size: 0.9rem; margin-top: 0.5rem; line-height: 1.6; position: relative; z-index: 1;
    }
    .result-card.benign .result-sub { color: #047857; }
    .result-card.malignant .result-sub { color: #b91c1c; }
    .result-card.empty .result-sub { color: var(--text-muted); }
    .result-stats {
        display: flex; gap: 0.75rem; justify-content: center;
        flex-wrap: wrap; margin-top: 1.5rem; position: relative; z-index: 1;
    }
    .result-stat {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.9);
        border-radius: var(--radius-sm);
        padding: 0.65rem 1.1rem; min-width: 120px;
        box-shadow: var(--shadow-xs);
    }
    .result-stat-label {
        display: block; font-size: 0.68rem; font-weight: 700;
        letter-spacing: 0.06em; text-transform: uppercase;
        color: var(--text-muted); margin-bottom: 0.2rem;
    }
    .result-stat-value {
        display: block; font-size: 1.1rem; font-weight: 800;
        color: var(--navy); letter-spacing: -0.02em;
    }
    .result-stat-value-sm { font-size: 0.78rem; font-weight: 600; line-height: 1.4; }
    .result-card.benign .result-stat-value { color: #065f46; }
    .result-card.malignant .result-stat-value { color: #991b1b; }

    /* ── Confidence panel ── */
    .conf-panel {
        background: linear-gradient(160deg, #f8fafc 0%, #ffffff 100%);
        border: 1px solid var(--border-light);
        border-radius: var(--radius);
        padding: 1.5rem 1.25rem;
        text-align: center;
        box-shadow: var(--shadow-card);
        margin-bottom: 0.75rem;
    }
    .conf-value-display {
        font-size: 2.75rem; font-weight: 800; color: var(--navy);
        letter-spacing: -0.04em; line-height: 1;
    }
    .conf-unit { font-size: 1.4rem; font-weight: 700; color: var(--text-muted); }
    .conf-caption {
        font-size: 0.8125rem; color: var(--text-muted);
        margin: 0.4rem 0 1rem; font-weight: 500;
    }
    .conf-bar-track {
        height: 6px; background: var(--bg-subtle);
        border-radius: 99px; overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--blue), var(--teal));
        border-radius: 99px;
        transition: width 0.6s var(--ease);
    }

    /* ── Explanation section ── */
    .explain-section {
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: var(--radius);
        padding: 1.65rem 1.85rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-card);
        transition: box-shadow 0.25s var(--ease);
    }
    .explain-section:hover { box-shadow: var(--shadow-md); }
    .explain-section h3 {
        color: var(--navy); font-size: 1.05rem;
        font-weight: 700; margin: 0 0 0.4rem 0; letter-spacing: -0.01em;
    }
    .explain-section p {
        color: var(--text-muted); font-size: 0.875rem;
        margin: 0 0 0.5rem 0; line-height: 1.65;
    }
    [data-testid="stArrowVegaLiteChart"] {
        background: #ffffff !important;
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-light);
        padding: 0.5rem;
    }

    /* ── Footer ── */
    .app-footer {
        margin-top: 2.5rem;
        padding: 2rem 2.25rem;
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: var(--radius);
        text-align: center;
        color: var(--text-muted);
        font-size: 0.875rem;
        line-height: 1.85;
        box-shadow: var(--shadow-card);
    }
    .app-footer .footer-name {
        font-size: 1.05rem; font-weight: 800; color: var(--navy);
        letter-spacing: -0.01em;
    }
    .app-footer .footer-degree {
        font-size: 0.875rem; color: var(--text-secondary);
        margin-top: 0.25rem; font-weight: 500;
    }
    .app-footer a {
        color: var(--blue); text-decoration: none; font-weight: 600;
        margin: 0 0.5rem;
        transition: color 0.2s var(--ease);
    }
    .app-footer a:hover { color: var(--navy); text-decoration: underline; }
    .footer-links { margin-top: 0.85rem; }
    .footer-divider { display: inline-block; margin: 0 0.5rem; opacity: 0.35; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f7fa 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    [data-testid="stSidebar"] h3 {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        color: var(--navy) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown {
        color: #1e293b !important;
        font-size: 0.92rem !important;
        line-height: 1.65 !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #475569 !important;
        font-size: 0.84rem !important;
        line-height: 1.55 !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 1.1rem 0 !important;
        border-color: var(--border) !important;
    }
    .sidebar-metric {
        display: flex; align-items: center; gap: 0.75rem;
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-sm);
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        box-shadow: var(--shadow-xs);
        transition: box-shadow 0.25s var(--ease), transform 0.25s var(--ease);
    }
    .sidebar-metric:hover { box-shadow: var(--shadow-sm); transform: translateX(2px); }
    .sidebar-metric-icon {
        width: 36px; height: 36px; border-radius: 9px;
        background: linear-gradient(135deg, #bfdbfe 0%, #60a5fa 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.15rem; line-height: 1; flex-shrink: 0;
        border: 1px solid rgba(255,255,255,0.75);
        box-shadow: 0 2px 6px rgba(15,23,42,0.08);
    }
    .sidebar-metric-label {
        font-size: 0.72rem; font-weight: 600;
        color: var(--text-muted) !important;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    .sidebar-metric-value {
        font-size: 0.88rem; font-weight: 700;
        color: var(--navy) !important;
        line-height: 1.3;
    }
    .sidebar-section-title {
        font-size: 0.82rem; font-weight: 700; color: var(--navy);
        text-transform: uppercase; letter-spacing: 0.05em;
        margin: 1rem 0 0.65rem 0;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 100%) !important;
        color: white !important; border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.85rem 2rem !important;
        font-weight: 700 !important; font-size: 1rem !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 16px rgba(37,99,235,0.32) !important;
        transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(37,99,235,0.42) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: var(--bg);
        border-radius: var(--radius-sm); padding: 4px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 0.55rem 1.1rem;
        font-weight: 600; font-size: 0.84rem; color: var(--text-muted);
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface) !important;
        color: var(--blue) !important;
        box-shadow: var(--shadow-sm);
    }

    /* ── Inputs ── */
    .stNumberInput label {
        font-weight: 600 !important; color: var(--navy) !important;
        font-size: 0.82rem !important;
    }
    .stNumberInput input {
        border-radius: var(--radius-xs) !important;
        border-color: var(--border) !important;
        transition: border-color 0.2s var(--ease), box-shadow 0.2s var(--ease) !important;
    }
    .stNumberInput input:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    }

    /* ── Column spacing ── */
    [data-testid="column"] { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }

    /* ── Responsive ── */
    @media (max-width: 1024px) {
        .metrics-grid { grid-template-columns: repeat(2, 1fr); }
        .metrics-grid-2 { grid-template-columns: repeat(2, 1fr); }
        .hero-banner { padding: 2.25rem 2rem; }
    }
    @media (max-width: 768px) {
        .metrics-grid, .metrics-grid-2 { grid-template-columns: 1fr; }
        .block-container { padding-top: 1.25rem; }
        .result-stats { flex-direction: column; align-items: center; }
        .result-stat { width: 100%; max-width: 240px; }
    }
    @media (max-width: 640px) {
        .hero-banner { padding: 1.75rem 1.5rem; }
        .hero-banner h1 { font-size: 1.6rem; }
        .saas-nav { flex-direction: column; gap: 0.85rem; align-items: flex-start; }
        .result-card { padding: 1.75rem 1.25rem; }
        .result-label { font-size: 1.45rem; }
        .conf-value-display { font-size: 2.25rem; }
        .progress-wrap { padding: 1rem 1.15rem; }
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
status_text, status_icon = get_prediction_status()

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.65rem;margin-bottom:1.25rem;">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#2563eb,#0d9488);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;">🩺</div>
            <div>
                <div style="font-weight:700;font-size:0.95rem;color:#0c1e3a;">MedAI Dashboard</div>
                <div style="font-size:0.78rem;color:#475569;">Diagnostic Suite v1.0</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">📖 About Project</div>', unsafe_allow_html=True)
    st.markdown(
        "ML-powered classification of breast tumor samples as **Benign** or "
        "**Malignant** using 30 FNA cell nucleus features from the Wisconsin dataset."
    )

    st.divider()

    st.markdown('<div class="sidebar-section-title">🔬 Model Information</div>', unsafe_allow_html=True)
    render_sidebar_metric("Algorithm", ALGORITHM_NAME, "🤖")
    render_sidebar_metric("Dataset", DATASET_NAME, "🧬")
    render_sidebar_metric("Features / Classes", f"{NUM_FEATURES} / {NUM_CLASSES}", "📐")
    acc_display = f"{TRAINING_ACCURACY}%" if TRAINING_ACCURACY else "N/A"
    render_sidebar_metric("CV Accuracy", acc_display, "📈")
    render_sidebar_metric("Status", f"{status_icon} {status_text}", "🎯")

    st.divider()

    st.markdown('<div class="sidebar-section-title">📋 Quick Guide</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. **Enter** measurement values across all 3 tabs
        2. **Track** the input completion progress bar
        3. **Click** Run Prediction to classify the sample
        4. **Review** diagnosis, confidence, and AI explanation
        """
    )

    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This tool is for educational and demonstration purposes only. "
        "It is not a substitute for professional medical diagnosis."
    )

# ---------------------------------------------------------------------------
# Top navigation bar
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="saas-nav">
        <div class="saas-nav-brand">
            <div class="saas-nav-logo">🩺</div>
            <div>
                <div class="saas-nav-title">Breast Cancer Prediction System</div>
                <div class="saas-nav-sub">Healthcare AI · Diagnostic Dashboard</div>
            </div>
        </div>
        <div class="saas-nav-status">
            <span class="status-dot"></span>
            Model Online
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-content">
            <div class="hero-badge">✦ AI-Powered Diagnostics</div>
            <h1>Tumor Classification Dashboard</h1>
            <p>
                Enter cell nucleus measurements to generate an instant
                <strong>Benign</strong> or <strong>Malignant</strong> classification
                powered by a trained machine learning model.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dynamic model information metrics
# ---------------------------------------------------------------------------
st.markdown('<div class="section-eyebrow">Model Overview</div>', unsafe_allow_html=True)
render_model_metrics_grid(ALGORITHM_NAME, TRAINING_ACCURACY, status_text, status_icon)

# ---------------------------------------------------------------------------
# Dashboard layout – inputs (left) + info panel (right)
# ---------------------------------------------------------------------------
col_inputs, col_panel = st.columns([1.65, 1], gap="large")

feature_values = {}

st.markdown('<div class="section-eyebrow">Patient Analysis</div>', unsafe_allow_html=True)

with col_inputs:
    st.markdown(
        """
        <div class="dash-card">
            <div class="dash-card-header">
                <div class="dash-card-icon">🔬</div>
                <h3>Patient Sample Inputs</h3>
            </div>
            <p>Enter all 30 tumor measurement values across the three feature groups below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_labels = [f"{info['icon']} {cat}" for cat, info in FEATURE_CATEGORIES.items()]
    tabs = st.tabs(tab_labels)

    for tab, (category, info) in zip(tabs, FEATURE_CATEGORIES.items()):
        with tab:
            st.caption(info["description"])
            in_col1, in_col2 = st.columns(2)
            for i, feature in enumerate(info["features"]):
                target_col = in_col1 if i % 2 == 0 else in_col2
                with target_col:
                    feature_values[feature] = st.number_input(
                        feature, min_value=0.0, value=0.0, key=feature
                    )

    features = [feature_values[name] for name in feature_names]
    filled_count = count_completed_features(features)
    st.session_state.filled_count = filled_count

    render_completion_tracker(filled_count)

    st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)

    if st.button("🔬 Run Prediction", use_container_width=True):
        prediction, confidence, _ = run_prediction(model, features)

        st.session_state.prediction = prediction[0]
        st.session_state.confidence = confidence
        st.session_state.prediction_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        st.session_state.top_features = get_top_contributing_features(
            model, features, feature_names
        )
        st.session_state.explanation_method = (
            st.session_state.top_features[0][3] if st.session_state.top_features else None
        )
        st.rerun()

with col_panel:
    st.markdown(
        """
        <div class="dash-card">
            <div class="dash-card-header">
                <div class="dash-card-icon">📋</div>
                <h3>How It Works</h3>
            </div>
            <ul class="info-list">
                <li><span class="step-num">1</span>Enter values in all three input tabs</li>
                <li><span class="step-num">2</span>Verify completion progress reaches 100%</li>
                <li><span class="step-num">3</span>Click <strong>Run Prediction</strong></li>
                <li><span class="step-num">4</span>Review diagnosis, confidence &amp; explanation</li>
            </ul>
        </div>
        <div class="dash-card">
            <div class="dash-card-header">
                <div class="dash-card-icon">⚙️</div>
                <h3>Model Details</h3>
            </div>
            <div class="spec-row"><span class="spec-key">Algorithm</span><span class="spec-val">{algo}</span></div>
            <div class="spec-row"><span class="spec-key">Dataset</span><span class="spec-val">Wisconsin BC</span></div>
            <div class="spec-row"><span class="spec-key">Features</span><span class="spec-val">{NUM_FEATURES} inputs</span></div>
            <div class="spec-row"><span class="spec-key">Classes</span><span class="spec-val">{NUM_CLASSES} (Benign/Malignant)</span></div>
            <div class="spec-row"><span class="spec-key">CV Accuracy</span><span class="spec-val">{acc}</span></div>
            <div class="spec-row"><span class="spec-key">Status</span><span class="spec-val">{status_icon} {status_text}</span></div>
        </div>
        """.format(
            algo=ALGORITHM_NAME,
            acc=f"{TRAINING_ACCURACY}%" if TRAINING_ACCURACY else "N/A",
            status_icon=status_icon,
            status_text=status_text,
            NUM_FEATURES=NUM_FEATURES,
            NUM_CLASSES=NUM_CLASSES,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="dash-card" style="padding-bottom:0.5rem;">
            <div class="dash-card-header">
                <div class="dash-card-icon">🎯</div>
                <h3>Diagnosis Result</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.prediction is None:
        st.markdown(
            """
            <div class="result-card empty animate-in">
                <div class="result-eyebrow">Diagnosis Result</div>
                <div class="result-icon-ring">⏳</div>
                <div class="result-label">Awaiting Prediction</div>
                <div class="result-sub">Enter sample data and click Run Prediction to see results here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_result_card(
            st.session_state.prediction,
            st.session_state.confidence,
            st.session_state.prediction_time,
        )

# ---------------------------------------------------------------------------
# Confidence visualization & AI explanation (full width, post-prediction)
# ---------------------------------------------------------------------------
if st.session_state.prediction is not None:
    st.markdown(
        '<div class="section-eyebrow" style="margin-top:1.75rem;">Analysis &amp; Explainability</div>',
        unsafe_allow_html=True,
    )

    conf_col, explain_intro_col = st.columns([1, 2], gap="large")

    with conf_col:
        st.markdown(
            """
            <div class="dash-card">
                <div class="dash-card-header">
                    <div class="dash-card-icon">📊</div>
                    <h3>Confidence Score</h3>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.confidence is not None:
            render_confidence_visualization(st.session_state.confidence)
        else:
            st.info("Confidence score is not available for this model.")

    with explain_intro_col:
        if st.session_state.top_features:
            method = st.session_state.explanation_method or "Coefficient"
            st.markdown(
                f"""
                <div class="explain-section">
                    <h3>🧠 AI Explanation</h3>
                    <p>
                        The chart below shows the <strong>top 5 most influential input features</strong>
                        for this prediction, computed using <strong>{method}</strong>-based analysis.
                        Features pushing toward <span style="color:#059669;font-weight:600;">Benign</span>
                        or <span style="color:#dc2626;font-weight:600;">Malignant</span> are color-coded.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_explanation_chart(st.session_state.top_features, method)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        <div class="footer-name">Developed by V Krishna Kaushik</div>
        <div class="footer-degree">B.Tech CSE (AIML) · JNTUH College of Engineering Hyderabad</div>
        <div class="footer-links">
            <a href="https://github.com/KrishnaKaushik1707" target="_blank">GitHub ↗</a>
            <span class="footer-divider">|</span>
            <a href="https://www.linkedin.com/in/krishna-kaushik-097884333" target="_blank">LinkedIn ↗</a>
        </div>
        <span style="font-size:0.78rem; opacity:0.7; display:block; margin-top:0.75rem;">
            © 2026 Breast Cancer Prediction System · Educational purposes only · Not for clinical use
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
