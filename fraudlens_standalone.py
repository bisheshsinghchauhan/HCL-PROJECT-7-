"""FraudLens — portable, single-file credit-card fraud analytics application.

Run in VS Code's terminal:
    python -m pip install streamlit pandas numpy scikit-learn plotly python-dotenv oracledb
    streamlit run fraudlens_standalone.py

Oracle is optional. Configure ORACLE_USER, ORACLE_PASSWORD and ORACLE_DSN to enable
the connection check; CSV analysis works without Oracle.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Application shell ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudLens · Credit Card Fraud Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_option("client.showSidebarNavigation", False)

NAVY = "#07111f"
PANEL = "#0d1b2e"
PANEL_2 = "#11243c"
BLUE = "#43b4ff"
TEAL = "#27d6b2"
AMBER = "#f7c65d"
RED = "#ff5b79"
MUTED = "#91a8c0"

st.markdown(
    f"""
    <style>
    .stApp {{background:radial-gradient(circle at 85% 0%,#102945 0%,{NAVY} 38%,#050d18 100%);}}
    [data-testid="stSidebar"] {{background:#081525;border-right:1px solid #19334e;}}
    [data-testid="stSidebarNav"] {{display:none!important;}}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{color:#a9bed2;}}
    .block-container {{max-width:1480px;padding-top:1.45rem;padding-bottom:3rem;}}
    h1,h2,h3 {{letter-spacing:-.025em;color:#eff7ff;}}
    h1 {{font-size:2rem!important;margin-bottom:.2rem!important;}}
    div[data-testid="stMetric"] {{
      background:linear-gradient(145deg,{PANEL_2},{PANEL});border:1px solid #1c3b5a;
      border-radius:14px;padding:16px 17px;box-shadow:0 14px 35px rgba(0,0,0,.16);
    }}
    div[data-testid="stMetricLabel"] {{text-transform:uppercase;letter-spacing:.08em;color:{MUTED};font-size:.68rem;}}
    div[data-testid="stMetricValue"] {{color:#f4f9ff;font-size:1.55rem;}}
    div[data-testid="stDataFrame"],div[data-testid="stPlotlyChart"] {{
      background:{PANEL};border:1px solid #193754;border-radius:14px;padding:6px;
    }}
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
      background:#0c1b2d!important;border-color:#244766!important;
    }}
    .hero {{display:flex;justify-content:space-between;align-items:flex-end;padding:5px 2px 18px;border-bottom:1px solid #17334d;margin-bottom:18px;}}
    .eyebrow {{font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:{TEAL};font-weight:700;}}
    .hero-title {{font-size:2rem;font-weight:730;color:#f3f9ff;letter-spacing:-.04em;line-height:1.12;margin:5px 0;}}
    .hero-copy {{color:{MUTED};font-size:.94rem;max-width:760px;}}
    .source-pill {{background:#102b42;border:1px solid #245173;border-radius:999px;color:#bce5ff;padding:7px 12px;font-size:.75rem;white-space:nowrap;}}
    .callout {{background:linear-gradient(90deg,#0e2940,#0b1d31);border:1px solid #1e4b6e;border-left:4px solid {BLUE};border-radius:10px;padding:13px 15px;color:#bdd2e6;margin:10px 0 18px;}}
    .step {{background:{PANEL};border:1px solid #1b3853;border-radius:12px;padding:14px;height:100%;}}
    .step-number {{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:7px;background:#143d5b;color:#75cbff;font-weight:700;margin-right:7px;}}
    .section-label {{font-size:.73rem;letter-spacing:.13em;text-transform:uppercase;color:#a4bed5;font-weight:700;margin:8px 0 10px;}}
    .risk-low {{color:{TEAL}}}.risk-medium {{color:{AMBER}}}.risk-high,.risk-critical {{color:{RED}}}
    .stButton>button,.stDownloadButton>button {{border-radius:9px;border:1px solid #2b638b;background:#103858;color:#dff3ff;font-weight:650;}}
    .stButton>button:hover,.stDownloadButton>button:hover {{border-color:{BLUE};background:#164a70;color:white;}}
    [data-testid="stFileUploaderDropzone"] {{background:#0c1d30;border:1px dashed #32678e;border-radius:12px;}}
    hr {{border-color:#17334d!important;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data and detection engine ───────────────────────────────────────────────
FEATURES = ["Time", "Amount", *[f"V{i}" for i in range(1, 29)]]


class DataValidationError(ValueError):
    """Friendly error for invalid uploaded transaction files."""


@dataclass(frozen=True)
class RuleConfig:
    high_amount_quantile: float = 0.99
    amount_zscore: float = 3.5
    rapid_window_seconds: int = 300
    rapid_count: int = 4
    frequency_quantile: float = 0.99


def validate_and_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if raw.empty:
        raise DataValidationError("This CSV has column headings but no transaction rows.")
    missing = [column for column in FEATURES if column not in raw.columns]
    if missing:
        raise DataValidationError("Missing required columns: " + ", ".join(missing))
    result = raw.copy()
    quality = {
        "Source rows": int(len(result)),
        "Missing values": int(result.isna().sum().sum()),
        "Duplicate rows": int(result.duplicated().sum()),
    }
    numeric = [*FEATURES, *(["Class"] if "Class" in result else [])]
    invalid = 0
    for column in numeric:
        converted = pd.to_numeric(result[column], errors="coerce")
        invalid += int(converted.isna().sum() - result[column].isna().sum())
        result[column] = converted
    quality["Invalid numeric values"] = invalid
    quality["Negative amounts"] = int((result["Amount"] < 0).sum())
    result = result.dropna(subset=FEATURES).drop_duplicates()
    result = result[(result["Amount"] >= 0) & (result["Time"] >= 0)].copy()
    if "Class" in result:
        invalid_labels = result["Class"].notna() & ~result["Class"].isin([0, 1])
        result = result[~invalid_labels]
        result["Class"] = result["Class"].astype("Int64")
    if result.empty:
        raise DataValidationError("No valid rows remain after cleaning.")
    result = result.reset_index(drop=True)
    if "TransactionID" not in result:
        result.insert(0, "TransactionID", [f"TXN-{n:08d}" for n in range(1, len(result) + 1)])
    else:
        result["TransactionID"] = result["TransactionID"].astype(str)
    result["Hour"] = ((result["Time"] // 3600) % 24).astype(int)
    result["Day"] = (result["Time"] // 86400).astype(int) + 1
    quality["Rows retained"] = int(len(result))
    quality["Rows removed"] = int(quality["Source rows"] - len(result))
    return result, quality


def apply_rules(frame: pd.DataFrame, cfg: RuleConfig) -> pd.DataFrame:
    result = frame.copy()
    amount = result["Amount"].astype(float)
    median = float(amount.median())
    mad = float(np.median(np.abs(amount - median)))
    denominator = mad if mad > 0 else max(float(amount.std()), 1.0)
    robust_z = 0.6745 * (amount - median) / denominator
    result["Rule_HighAmount"] = amount >= amount.quantile(cfg.high_amount_quantile)

    ordered = result.assign(_position=np.arange(len(result))).sort_values("Time")
    ordered["_event_time"] = pd.to_datetime(ordered["Time"], unit="s")
    count = ordered.rolling(f"{cfg.rapid_window_seconds}s", on="_event_time")["Amount"].count()
    ordered["_rapid"] = count.to_numpy() >= cfg.rapid_count
    result["Rule_RapidTransactions"] = ordered.sort_values("_position")["_rapid"].to_numpy()

    hourly_counts = result["Hour"].map(result["Hour"].value_counts())
    cutoff = float(result["Hour"].value_counts().quantile(cfg.frequency_quantile))
    result["Rule_HighFrequency"] = hourly_counts >= cutoff
    result["Rule_StatisticalAnomaly"] = robust_z.abs() >= cfg.amount_zscore
    result["Rule_SuspiciousPattern"] = result["Hour"].between(0, 4) & (amount >= amount.quantile(0.95))
    rule_columns = [column for column in result if column.startswith("Rule_")]
    result["RuleFlagCount"] = result[rule_columns].sum(axis=1).astype(int)
    result["RulePrediction"] = (result["RuleFlagCount"] > 0).astype(int)
    labels = {
        "Rule_HighAmount": "unusually high amount",
        "Rule_RapidTransactions": "rapid transaction sequence",
        "Rule_HighFrequency": "unusual hourly frequency",
        "Rule_StatisticalAnomaly": "statistical amount anomaly",
        "Rule_SuspiciousPattern": "high-value overnight pattern",
    }
    result["RuleReasons"] = result.apply(
        lambda row: "; ".join(labels[c] for c in rule_columns if row[c]) or "No rule triggered", axis=1
    )
    return result


def calculate_metrics(y_true, y_pred, y_score=None) -> dict[str, float | int | None]:
    true, predicted = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()
    auc = None
    if y_score is not None and np.unique(true).size == 2:
        auc = float(roc_auc_score(true, y_score))
    return {
        "Accuracy": float(accuracy_score(true, predicted)),
        "Precision": float(precision_score(true, predicted, zero_division=0)),
        "Recall": float(recall_score(true, predicted, zero_division=0)),
        "F1": float(f1_score(true, predicted, zero_division=0)),
        "ROC-AUC": auc,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def run_models(frame: pd.DataFrame, contamination: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    x = result[FEATURES].astype(float)  # Class is intentionally excluded: no target leakage.
    isolation = IsolationForest(contamination=contamination, n_estimators=150, random_state=42, n_jobs=-1)
    raw_prediction = isolation.fit_predict(x)
    result["IsolationPrediction"] = (raw_prediction == -1).astype(int)
    result["IsolationScore"] = -isolation.decision_function(x)
    evaluations: list[dict] = []

    valid_labels = "Class" in result and result["Class"].notna().all()
    enough_labels = valid_labels and result["Class"].nunique() == 2 and result["Class"].value_counts().min() >= 2
    if enough_labels:
        y = result["Class"].astype(int)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.25, stratify=y, random_state=42)
        logistic = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ])
        logistic.fit(train_x, train_y)
        full_probability = logistic.predict_proba(x)[:, 1]
        test_probability = logistic.predict_proba(test_x)[:, 1]
        result["LogisticProbability"] = full_probability
        result["LogisticPrediction"] = (full_probability >= .5).astype(int)
        evaluations.append({
            "Model": "Logistic Regression",
            **calculate_metrics(test_y, (test_probability >= .5).astype(int), test_probability),
        })
        evaluations.append({
            "Model": "Isolation Forest",
            **calculate_metrics(y, result["IsolationPrediction"], result["IsolationScore"]),
        })
    else:
        result["LogisticProbability"] = np.nan
        result["LogisticPrediction"] = 0
    return result, pd.DataFrame(evaluations)


def score_risk(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    probability = result["LogisticProbability"].fillna(0).clip(0, 1)
    result["RiskScore"] = (
        result["RuleFlagCount"].clip(0, 5) * 10
        + result["IsolationPrediction"] * 25
        + probability * 25
    ).round().clip(0, 100).astype(int)
    result["RiskLevel"] = pd.cut(
        result["RiskScore"], [-1, 24, 49, 74, 100], labels=["Low", "Medium", "High", "Critical"]
    ).astype(str)
    result["MLPrediction"] = ((result["IsolationPrediction"] == 1) | (result["LogisticPrediction"] == 1)).astype(int)
    result["FinalPrediction"] = ((result["RulePrediction"] == 1) | (result["MLPrediction"] == 1)).astype(int)

    def explain(row: pd.Series) -> str:
        reasons = []
        if row["RulePrediction"]:
            reasons.append("Rules: " + row["RuleReasons"])
        if row["IsolationPrediction"]:
            reasons.append("ML: Isolation Forest anomaly")
        if row["LogisticPrediction"]:
            reasons.append(f"ML: logistic fraud probability {row['LogisticProbability']:.1%}")
        return " | ".join(reasons) or "No suspicious signal"

    result["DetectionReasons"] = result.apply(explain, axis=1)
    return result


@st.cache_data(show_spinner=False)
def synthetic_sample(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fraud_count = max(20, int(rows * .012))
    labels = np.zeros(rows, dtype=int)
    labels[rng.choice(rows, fraud_count, replace=False)] = 1
    data: dict[str, np.ndarray] = {
        "Time": np.sort(rng.integers(0, 172800, rows)),
        "Amount": rng.lognormal(3.4, 1.05, rows),
    }
    data["Amount"][labels == 1] *= rng.uniform(3, 8, fraud_count)
    for index in range(1, 29):
        values = rng.normal(0, 1, rows)
        if index in {1, 3, 4, 10, 12, 14, 17}:
            values[labels == 1] += rng.choice([-1, 1]) * rng.uniform(2, 4)
        data[f"V{index}"] = values
    data["Class"] = labels
    return pd.DataFrame(data).round(6)


@st.cache_data(show_spinner=False)
def process(raw: pd.DataFrame, cfg: RuleConfig, contamination: float):
    cleaned, quality = validate_and_clean(raw)
    ruled = apply_rules(cleaned, cfg)
    modeled, metrics = run_models(ruled, contamination)
    return score_risk(modeled), quality, metrics


# ── Reusable UI helpers ─────────────────────────────────────────────────────
def hero(title: str, description: str, section: str) -> None:
    source = st.session_state.get("source_name", "Synthetic demo · 2,500 rows")
    st.markdown(
        f"""<div class="hero"><div><div class="eyebrow">{section}</div>
        <div class="hero-title">{title}</div><div class="hero-copy">{description}</div></div>
        <div class="source-pill">● &nbsp; {source}</div></div>""",
        unsafe_allow_html=True,
    )


def section(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def label_notice() -> None:
    if st.session_state.get("is_synthetic", True):
        st.markdown(
            '<div class="callout"><b>Demonstration data:</b> the active dataset is synthetic. '
            "Upload the public ULB/Kaggle <code>creditcard.csv</code> before presenting real findings. "
            "V1–V28 are anonymized and have no disclosed business meaning.</div>",
            unsafe_allow_html=True,
        )


def styled_figure(figure, height: int = 360):
    figure.update_layout(
        height=height,
        margin=dict(l=25, r=20, t=55, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b8ccdf", family="Inter, Arial"),
        title_font=dict(color="#edf7ff", size=15),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#18314a", zerolinecolor="#18314a"),
        yaxis=dict(gridcolor="#18314a", zerolinecolor="#18314a"),
    )
    return figure


def risk_color(level: str) -> str:
    return {"Low": TEAL, "Medium": AMBER, "High": "#ff8c55", "Critical": RED}.get(level, BLUE)


def display_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in [
            "TransactionID", "Time", "Amount", "Class", "RiskScore", "RiskLevel",
            "RulePrediction", "IsolationPrediction", "LogisticPrediction",
            "FinalPrediction", "DetectionReasons",
        ] if column in frame
    ]


# ── Sidebar and application state ───────────────────────────────────────────
if "raw_data" not in st.session_state:
    st.session_state.raw_data = synthetic_sample()
    st.session_state.source_name = "Synthetic demo · 2,500 rows"
    st.session_state.is_synthetic = True

st.sidebar.markdown("## ◈ FraudLens")
st.sidebar.caption("TRANSACTION INTELLIGENCE")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Transactions", "Detection Lab", "Fraud Analytics", "Model Performance", "Database & Reports"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption("DETECTION SETTINGS")
high_quantile = st.sidebar.slider("High amount percentile", 90.0, 99.9, 99.0, .1, help="Flags amounts at or above this dataset percentile.") / 100
amount_z = st.sidebar.slider("Amount anomaly sensitivity", 1.5, 6.0, 3.5, .1, help="Lower values flag more robust statistical outliers.")
rapid_count = st.sidebar.slider("Rapid transaction count", 2, 15, 4)
rapid_window = st.sidebar.slider("Rapid window (seconds)", 60, 1800, 300, 60)
contamination = st.sidebar.slider("Expected anomaly rate", .001, .10, .01, .001, format="%.3f")
cfg = RuleConfig(high_quantile, amount_z, rapid_window, rapid_count)

with st.sidebar.expander("How scoring works"):
    st.write("Rules contribute up to 50 points, Isolation Forest adds 25, and supervised fraud probability adds up to 25.")
with st.sidebar.expander("Dataset requirements"):
    st.write("Required: Time, Amount, V1–V28. Optional: Class and TransactionID.")

try:
    with st.spinner("Validating data and refreshing fraud signals…"):
        data, quality, metrics = process(st.session_state.raw_data, cfg, contamination)
except Exception as exc:
    st.error(f"Analysis could not run: {exc}")
    st.stop()


# ── Pages ───────────────────────────────────────────────────────────────────
def overview_page() -> None:
    hero("Fraud command center", "A focused view of transaction exposure, labeled fraud, detection signals, and review priority.", "Executive overview")
    label_notice()
    truth = "Class" if "Class" in data and data["Class"].notna().all() else "FinalPrediction"
    fraud = data[truth] == 1
    top = st.columns(6)
    cards = [
        ("Transactions", f"{len(data):,}"),
        ("Fraud transactions", f"{int(fraud.sum()):,}"),
        ("Fraud rate", f"{fraud.mean():.2%}"),
        ("Fraud amount", f"{data.loc[fraud, 'Amount'].sum():,.2f}"),
        ("Average amount", f"{data['Amount'].mean():,.2f}"),
        ("High priority", f"{data['RiskLevel'].isin(['High', 'Critical']).sum():,}"),
    ]
    for column, (label, value) in zip(top, cards):
        column.metric(label, value)

    section("Transaction posture")
    left, right = st.columns([.82, 1.55], gap="large")
    with left:
        counts = data[truth].map({0: "Legitimate", 1: "Fraud"}).value_counts().reset_index()
        counts.columns = ["Status", "Transactions"]
        fig = px.pie(counts, names="Status", values="Transactions", hole=.68,
                     color="Status", color_discrete_map={"Legitimate": BLUE, "Fraud": RED}, title="Fraud vs legitimate")
        fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color=PANEL, width=2)))
        st.plotly_chart(styled_figure(fig), width="stretch")
    with right:
        trend = data.groupby("Day")[truth].agg(Transactions="count", Fraud="sum").reset_index()
        fig = px.area(trend, x="Day", y="Fraud", markers=True, title="Fraud trend over elapsed days", color_discrete_sequence=[TEAL])
        fig.update_traces(line=dict(width=3), fillcolor="rgba(39,214,178,.12)")
        st.plotly_chart(styled_figure(fig), width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        risk = data["RiskLevel"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0).reset_index()
        risk.columns = ["Risk level", "Transactions"]
        fig = px.bar(risk, x="Risk level", y="Transactions", color="Risk level", title="Review priority",
                     color_discrete_map={level: risk_color(level) for level in risk["Risk level"]})
        st.plotly_chart(styled_figure(fig), width="stretch")
    with right:
        compare = data[["RulePrediction", "IsolationPrediction", "LogisticPrediction", "FinalPrediction"]].sum().reset_index()
        compare.columns = ["Method", "Flagged"]
        compare["Method"] = compare["Method"].str.replace("Prediction", "", regex=False).replace({"Final": "Combined"})
        fig = px.bar(compare, x="Method", y="Flagged", color="Method", title="Detection method comparison")
        st.plotly_chart(styled_figure(fig), width="stretch")


def transactions_page() -> None:
    hero("Transaction explorer", "Search and narrow the ledger, then inspect exactly why an individual transaction was flagged.", "Investigation workspace")
    section("Find transactions")
    c1, c2, c3, c4 = st.columns([1.25, 1.5, 1, 1])
    search = c1.text_input("Transaction ID", placeholder="e.g. TXN-00000042")
    amount_range = c2.slider("Amount range", 0.0, float(data["Amount"].max()), (0.0, float(data["Amount"].max())))
    selected_risk = c3.multiselect("Risk", ["Low", "Medium", "High", "Critical"], default=["Low", "Medium", "High", "Critical"])
    status = c4.selectbox("Status", ["All", "Flagged", "Not flagged", "Labeled fraud", "Labeled legitimate"])
    sort_col, descending = st.columns([1, 1])
    sort_by = sort_col.selectbox("Sort by", ["RiskScore", "Amount", "Time", "TransactionID"])
    sort_desc = descending.toggle("Highest first", value=sort_by != "TransactionID")

    filtered = data[data["Amount"].between(*amount_range) & data["RiskLevel"].isin(selected_risk)].copy()
    if search:
        filtered = filtered[filtered["TransactionID"].str.contains(search, case=False, na=False)]
    if status == "Flagged": filtered = filtered[filtered["FinalPrediction"] == 1]
    if status == "Not flagged": filtered = filtered[filtered["FinalPrediction"] == 0]
    if status.startswith("Labeled") and "Class" in filtered:
        filtered = filtered[filtered["Class"] == (1 if status.endswith("fraud") else 0)]
    filtered = filtered.sort_values(sort_by, ascending=not sort_desc)
    st.caption(f"{len(filtered):,} of {len(data):,} transactions match these filters")
    st.dataframe(
        filtered[display_columns(filtered)], width="stretch", height=510, hide_index=True,
        column_config={
            "RiskScore": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%d"),
            "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
        },
    )
    if not filtered.empty:
        section("Transaction detail")
        selected_id = st.selectbox("Choose a transaction to inspect", filtered["TransactionID"].head(1000))
        row = filtered.loc[filtered["TransactionID"] == selected_id].iloc[0]
        detail = st.columns(4)
        detail[0].metric("Amount", f"{row['Amount']:,.2f}")
        detail[1].metric("Risk score", f"{row['RiskScore']}/100")
        detail[2].metric("Risk level", row["RiskLevel"])
        detail[3].metric("Final flag", "Review" if row["FinalPrediction"] else "No alert")
        st.info(row["DetectionReasons"], icon="🔎")


def detection_page() -> None:
    hero("Detection lab", "Upload compatible data, adjust thresholds in the sidebar, compare detection layers, and export scored results.", "Run fraud detection")
    steps = st.columns(3, gap="medium")
    for column, number, title, copy in zip(
        steps, ["1", "2", "3"], ["Upload", "Review", "Export"],
        ["Choose a Kaggle-compatible CSV.", "Check quality and flagged reasons.", "Download the complete scored ledger."],
    ):
        column.markdown(f'<div class="step"><span class="step-number">{number}</span><b>{title}</b><br><span style="color:{MUTED};font-size:.83rem">{copy}</span></div>', unsafe_allow_html=True)
    st.write("")
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"], help="Required columns: Time, Amount, V1–V28. Class is optional.")
    action1, action2 = st.columns([1, 4])
    if uploaded is not None and action1.button("Use uploaded data", type="primary"):
        try:
            candidate = pd.read_csv(uploaded)
            validate_and_clean(candidate)
            st.session_state.raw_data = candidate
            st.session_state.source_name = f"Uploaded · {uploaded.name}"
            st.session_state.is_synthetic = False
            st.rerun()
        except (DataValidationError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(str(exc))
    if action2.button("Restore safe synthetic demo"):
        st.session_state.raw_data = synthetic_sample()
        st.session_state.source_name = "Synthetic demo · 2,500 rows"
        st.session_state.is_synthetic = True
        st.rerun()
    label_notice()

    section("Data quality checkpoint")
    qcols = st.columns(len(quality))
    for column, (label, value) in zip(qcols, quality.items()):
        column.metric(label, f"{value:,}")

    flagged = data[data["FinalPrediction"] == 1]
    section("Transactions requiring review")
    c1, c2, c3 = st.columns(3)
    c1.metric("Combined flags", f"{len(flagged):,}")
    c2.metric("Rule flags", f"{data['RulePrediction'].sum():,}")
    c3.metric("ML flags", f"{data['MLPrediction'].sum():,}")
    st.dataframe(flagged[display_columns(flagged)], width="stretch", height=430, hide_index=True)
    csv_bytes = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download complete scored results", csv_bytes, "fraud_detection_results.csv", "text/csv", type="primary")


def analytics_page() -> None:
    hero("Fraud pattern analytics", "Understand where fraud or alerts concentrate across amount bands, time, anomaly scores, and false positives.", "Deep-dive analysis")
    truth = "Class" if "Class" in data and data["Class"].notna().all() else "FinalPrediction"
    bands = pd.cut(data["Amount"], [-.01, 10, 50, 100, 250, 500, np.inf], labels=["0–10", "10–50", "50–100", "100–250", "250–500", "500+"])
    amount_report = data.assign(AmountBand=bands).groupby("AmountBand", observed=False)[truth].agg(Transactions="count", Fraud="sum", FraudRate="mean").reset_index()
    amount_report["FraudRate"] *= 100
    fig = px.bar(amount_report, x="AmountBand", y="FraudRate", color="FraudRate", text_auto=".2f",
                 color_continuous_scale=[[0, BLUE], [1, RED]], title="Fraud percentage by transaction amount range")
    st.plotly_chart(styled_figure(fig), width="stretch")
    left, right = st.columns(2, gap="large")
    with left:
        hourly = data.groupby("Hour")[truth].agg(Transactions="count", Fraud="sum", FraudRate="mean").reset_index()
        fig = px.line(hourly, x="Hour", y="FraudRate", markers=True, title="Fraud rate by hour", color_discrete_sequence=[TEAL])
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(styled_figure(fig), width="stretch")
    with right:
        labels = data[truth].map({0: "Legitimate", 1: "Fraud"})
        fig = px.histogram(data, x="IsolationScore", color=labels, nbins=50, barmode="overlay",
                           color_discrete_map={"Legitimate": BLUE, "Fraud": RED}, title="Isolation anomaly distribution")
        st.plotly_chart(styled_figure(fig), width="stretch")
    if truth == "Class":
        false_positive = data[(data["Class"] == 0) & (data["FinalPrediction"] == 1)]
        section(f"False-positive review · {len(false_positive):,} rows")
        st.dataframe(false_positive[["TransactionID", "Amount", "RiskScore", "RiskLevel", "DetectionReasons"]], width="stretch", hide_index=True)
    else:
        st.info("False-positive analysis requires trustworthy Class labels.")


def performance_page() -> None:
    hero("Model performance", "Evaluate rare-class performance with precision, recall, F1, ROC-AUC, and confusion matrices—not accuracy alone.", "Validation center")
    if metrics.empty:
        st.warning("A labeled dataset containing both Class values 0 and 1 is required for model evaluation.")
        return
    section("Balanced evaluation metrics")
    st.dataframe(metrics.style.format({name: "{:.3f}" for name in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]}), width="stretch", hide_index=True)
    long = metrics.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", range_y=[0, 1], title="Model comparison")
    st.plotly_chart(styled_figure(fig, 410), width="stretch")
    section("Confusion matrices")
    matrix_columns = st.columns(len(metrics))
    for column, (_, row) in zip(matrix_columns, metrics.iterrows()):
        matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        fig = go.Figure(go.Heatmap(z=matrix, x=["Predicted legitimate", "Predicted fraud"], y=["Actual legitimate", "Actual fraud"],
                                   text=matrix, texttemplate="%{text}", colorscale=[[0, PANEL_2], [1, BLUE]], showscale=False))
        fig.update_layout(title=row["Model"])
        column.plotly_chart(styled_figure(fig, 330), width="stretch")
    st.markdown('<div class="callout"><b>Interpretation:</b> recall measures how much actual fraud is found; precision measures how trustworthy alerts are. Accuracy can look high even when rare fraud is missed.</div>', unsafe_allow_html=True)


def database_page() -> None:
    hero("Database and reports", "Check optional Oracle readiness, reconcile record counts, and create analyst-ready CSV reports.", "Data operations")
    missing = [name for name in ["ORACLE_USER", "ORACLE_PASSWORD", "ORACLE_DSN"] if not os.getenv(name)]
    connected = False
    if missing:
        st.warning("Oracle unavailable: missing " + ", ".join(missing) + ". CSV mode is fully operational.")
    else:
        try:
            import oracledb
            connection = oracledb.connect(user=os.environ["ORACLE_USER"], password=os.environ["ORACLE_PASSWORD"], dsn=os.environ["ORACLE_DSN"])
            connection.close()
            connected = True
            st.success("Oracle connection verified.")
        except Exception as exc:
            st.warning(f"Oracle unavailable: {type(exc).__name__}: {exc}. CSV mode is fully operational.")
    counts = st.columns(4)
    counts[0].metric("Oracle status", "Connected" if connected else "CSV mode")
    counts[1].metric("Transaction rows", f"{len(data):,}")
    counts[2].metric("Prediction rows", f"{len(data):,}")
    counts[3].metric("Flagged rows", f"{data['FinalPrediction'].sum():,}")
    section("Data quality report")
    quality_frame = pd.DataFrame({"Check": quality.keys(), "Value": quality.values()})
    st.dataframe(quality_frame, width="stretch", hide_index=True)

    reports = {
        "Risk summary": data.groupby("RiskLevel", as_index=False).agg(Transactions=("TransactionID", "count"), TotalAmount=("Amount", "sum"), Flagged=("FinalPrediction", "sum")),
        "Hourly report": data.groupby("Hour", as_index=False).agg(Transactions=("TransactionID", "count"), TotalAmount=("Amount", "sum"), Flagged=("FinalPrediction", "sum")),
        "Flagged transactions": data[data["FinalPrediction"] == 1],
        "Model results": metrics,
    }
    section("Exportable analytical reports")
    report_name = st.selectbox("Choose report", list(reports))
    report = reports[report_name]
    st.dataframe(report, width="stretch", hide_index=True)
    st.download_button("Download selected report", report.to_csv(index=False).encode(), report_name.lower().replace(" ", "_") + ".csv", "text/csv")


PAGES = {
    "Overview": overview_page,
    "Transactions": transactions_page,
    "Detection Lab": detection_page,
    "Fraud Analytics": analytics_page,
    "Model Performance": performance_page,
    "Database & Reports": database_page,
}
PAGES[page]()

st.sidebar.divider()
st.sidebar.caption("FraudLens · Educational analytics project")
