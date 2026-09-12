
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="AquaClass | Explainable Water Quality", page_icon="💧", layout="wide")

APP_DIR = Path(__file__).parent
HISTORY_FILE = APP_DIR / "historical_water_issues.csv"

PARAMETERS = [
    "pH", "Turbidity", "TDS", "Dissolved Oxygen", "Nitrate", "BOD",
    "COD", "Conductivity", "Coliform Indicator", "Temperature"
]

ALIASES = {
    "pH": ["ph", "p_h"],
    "Turbidity": ["turbidity", "turbidity_ntu", "ntu"],
    "TDS": ["tds", "total_dissolved_solids", "total dissolved solids"],
    "Dissolved Oxygen": ["dissolved_oxygen", "dissolved oxygen", "do", "oxygen"],
    "Nitrate": ["nitrate", "nitrate_mg_l", "no3", "no3_n"],
    "BOD": ["bod", "biochemical_oxygen_demand"],
    "COD": ["cod", "chemical_oxygen_demand"],
    "Conductivity": ["conductivity", "electrical_conductivity", "ec"],
    "Coliform Indicator": ["coliform", "coliform_indicator", "total_coliform", "fecal_coliform"],
    "Temperature": ["temperature", "temp", "water_temperature"],
}

# Screening bands. These are deliberately exposed in the UI and are not a substitute
# for a jurisdiction-specific drinking-water, recreational-water, or discharge standard.
# Values are generic screening thresholds and can be edited in the app.
DEFAULT_BANDS = {
    "pH": {"Good": (6.5, 8.5), "Moderate": (6.0, 9.0), "Poor": (5.5, 9.5)},
    "Turbidity": {"Good": (0, 1), "Moderate": (1, 5), "Poor": (5, 10)},
    "TDS": {"Good": (0, 300), "Moderate": (300, 600), "Poor": (600, 1000)},
    "Dissolved Oxygen": {"Good": (6, np.inf), "Moderate": (4, 6), "Poor": (2, 4)},
    "Nitrate": {"Good": (0, 10), "Moderate": (10, 25), "Poor": (25, 50)},
    "BOD": {"Good": (0, 3), "Moderate": (3, 6), "Poor": (6, 10)},
    "COD": {"Good": (0, 20), "Moderate": (20, 50), "Poor": (50, 100)},
    "Conductivity": {"Good": (0, 500), "Moderate": (500, 1000), "Poor": (1000, 2000)},
    "Coliform Indicator": {"Good": (0, 1), "Moderate": (1, 10), "Poor": (10, 100)},
    "Temperature": {"Good": (5, 25), "Moderate": (25, 30), "Poor": (30, 35)},
}

HIGHER_IS_BETTER = {"Dissolved Oxygen"}
LOWER_IS_BETTER = {
    "Turbidity", "TDS", "Nitrate", "BOD", "COD", "Conductivity",
    "Coliform Indicator"
}

def normalize(s):
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())

def find_column(df, parameter):
    normalized = {normalize(c): c for c in df.columns}
    candidates = [parameter] + ALIASES[parameter]
    for c in candidates:
        if normalize(c) in normalized:
            return normalized[normalize(c)]
    return None

def standardize_columns(df):
    out = df.copy()
    rename = {}
    for p in PARAMETERS:
        c = find_column(out, p)
        if c:
            rename[c] = p
    return out.rename(columns=rename)

def screen_one(parameter, value, bands):
    if pd.isna(value):
        return "Not available"
    v = float(value)
    b = bands[parameter]
    if parameter in HIGHER_IS_BETTER:
        if v >= b["Good"][0]: return "Good"
        if v >= b["Moderate"][0]: return "Moderate"
        if v >= b["Poor"][0]: return "Poor"
        return "Very Poor"
    if parameter == "pH":
        if b["Good"][0] <= v <= b["Good"][1]: return "Good"
        if b["Moderate"][0] <= v <= b["Moderate"][1]: return "Moderate"
        if b["Poor"][0] <= v <= b["Poor"][1]: return "Poor"
        return "Very Poor"
    if parameter == "Temperature":
        if b["Good"][0] <= v <= b["Good"][1]: return "Good"
        if b["Moderate"][0] <= v <= b["Moderate"][1]: return "Moderate"
        if b["Poor"][0] <= v <= b["Poor"][1]: return "Poor"
        return "Very Poor"
    # lower is better
    if v <= b["Good"][1]: return "Good"
    if v <= b["Moderate"][1]: return "Moderate"
    if v <= b["Poor"][1]: return "Poor"
    return "Very Poor"

def score_one(parameter, value, bands):
    label = screen_one(parameter, value, bands)
    return {"Good": 4, "Moderate": 3, "Poor": 2, "Very Poor": 1}.get(label, np.nan)

def classify_row(row, bands):
    scores = [score_one(p, row[p], bands) for p in PARAMETERS if p in row and not pd.isna(row[p])]
    scores = [x for x in scores if not pd.isna(x)]
    if not scores:
        return "Not available"
    avg = np.mean(scores)
    if avg >= 3.5: return "Good"
    if avg >= 2.5: return "Moderate"
    if avg >= 1.5: return "Poor"
    return "Very Poor"

def parameter_result_row(row, bands):
    return {p: screen_one(p, row[p], bands) if p in row else "Not available" for p in PARAMETERS}

def clean_numeric(df):
    for p in PARAMETERS:
        if p in df:
            df[p] = pd.to_numeric(df[p], errors="coerce")
    return df

def infer_target(df):
    for c in ["Water-quality classifier", "Water Quality Class", "water_quality_class", "class", "label", "target"]:
        if c in df.columns:
            return c
    return None

def train_model(df):
    work = clean_numeric(df.copy())
    target = infer_target(work)
    if target is None:
        return None, None, "No target column found."
    X = work[PARAMETERS].copy()
    y = work[target].astype(str)
    valid = X.notna().sum(axis=1) >= 3
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < 30 or y.nunique() < 2:
        return None, None, "At least 30 usable rows and 2 classes are recommended for ML training."
    X = X.fillna(X.median(numeric_only=True))
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    model = RandomForestClassifier(
        n_estimators=350, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, zero_division=0, output_dict=True)
    return model, report, None

def make_contribution(model, row):
    if model is None:
        return pd.DataFrame()
    x = pd.DataFrame([{p: row.get(p, np.nan) for p in PARAMETERS}]).fillna(0)
    # Tree impurity importance is global; permutation importance on the single row is unstable.
    # We therefore present normalized model feature importance as "model contribution".
    imp = pd.Series(model.feature_importances_, index=PARAMETERS)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    return imp.rename("Contribution %").reset_index(names="Parameter")

st.title("💧 AquaClass")
st.caption("Explainable Water Quality Classification & Global Water-Issue Explorer")

with st.sidebar:
    st.header("Analysis settings")
    uploaded = st.file_uploader("Upload water-quality CSV", type=["csv"])
    show_history = st.checkbox("Show historical global water issues", True)
    st.divider()
    st.subheader("Screening bands")
    st.caption("Edit these generic screening bands for your project. For real decisions, use the applicable local/regulatory standard.")

# Load data
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        source_name = uploaded.name
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    df = pd.DataFrame([{
        "Location": "Demo sample",
        "Country": "Pakistan",
        "Latitude": 30.3753,
        "Longitude": 69.3451,
        "pH": 7.2, "Turbidity": 2.2, "TDS": 420,
        "Dissolved Oxygen": 5.2, "Nitrate": 14, "BOD": 4.5,
        "COD": 35, "Conductivity": 720, "Coliform Indicator": 4,
        "Temperature": 27
    }])
    source_name = "Built-in demonstration record"

df = standardize_columns(df)
df = clean_numeric(df)

# Session-persistent bands
if "bands" not in st.session_state:
    st.session_state.bands = DEFAULT_BANDS.copy()
bands = st.session_state.bands

with st.expander("Customize screening bands", expanded=False):
    cols = st.columns(2)
    for i, p in enumerate(PARAMETERS):
        with cols[i % 2]:
            b = bands[p]
            if p == "pH":
                lo = st.number_input(f"{p} Good min", value=float(b["Good"][0]), key=f"{p}_glo")
                hi = st.number_input(f"{p} Good max", value=float(b["Good"][1]), key=f"{p}_ghi")
                bands[p]["Good"] = (lo, hi)
            else:
                hi = st.number_input(f"{p} Good upper", value=float(b["Good"][1]), key=f"{p}_ghi")
                bands[p]["Good"] = (b["Good"][0], hi)
                if p in HIGHER_IS_BETTER:
                    mid = st.number_input(f"{p} Moderate lower", value=float(b["Moderate"][0]), key=f"{p}_mlo")
                    poor = st.number_input(f"{p} Poor lower", value=float(b["Poor"][0]), key=f"{p}_plo")
                    bands[p]["Moderate"] = (mid, b["Moderate"][1])
                    bands[p]["Poor"] = (poor, b["Poor"][1])
                else:
                    mid = st.number_input(f"{p} Moderate upper", value=float(b["Moderate"][1]), key=f"{p}_mhi")
                    poor = st.number_input(f"{p} Poor upper", value=float(b["Poor"][1]), key=f"{p}_phi")
                    bands[p]["Moderate"] = (b["Moderate"][0], mid)
                    bands[p]["Poor"] = (b["Poor"][0], poor)

# Determine usable parameter columns
available = [p for p in PARAMETERS if p in df.columns]
missing = [p for p in PARAMETERS if p not in df.columns]

if missing:
    st.warning("Missing parameters: " + ", ".join(missing) + ". Uploading aliases such as `ph`, `do`, `ec`, `ntu` is supported.")

if available:
    df["Water-quality classifier"] = df.apply(lambda r: classify_row(r, bands), axis=1)
    result_cols = [c for c in ["Location", "Country", "Date", "Latitude", "Longitude"] if c in df.columns] + PARAMETERS + ["Water-quality classifier"]
else:
    st.error("No recognized water-quality parameter columns were found.")
    st.info("Expected parameters: " + ", ".join(PARAMETERS))
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Classification", "🔎 Explainability", "🌍 Global Map", "📚 Historical Issues", "🤖 ML Model"
])

with tab1:
    st.subheader("Water-quality classification")
    st.write(f"Data source: **{source_name}** · {len(df):,} records")
    counts = df["Water-quality classifier"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    for col, label in zip([c1,c2,c3,c4], ["Good","Moderate","Poor","Very Poor"]):
        col.metric(label, int(counts.get(label, 0)))
    st.dataframe(df[result_cols], use_container_width=True, hide_index=True)
    export = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download analyzed CSV", export, "aquaclass_analyzed.csv", "text/csv")

with tab2:
    st.subheader("Parameter contribution")
    idx = st.number_input("Select record number", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
    row = df.iloc[int(idx)]
    st.markdown(f"### Final result: **{row['Water-quality classifier']}**")
    result = pd.DataFrame({
        "Parameter": PARAMETERS,
        "Value": [row.get(p, np.nan) for p in PARAMETERS],
        "Class": [screen_one(p, row.get(p, np.nan), bands) if p in row else "Not available" for p in PARAMETERS],
        "Score": [score_one(p, row.get(p, np.nan), bands) if p in row else np.nan for p in PARAMETERS]
    })
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.bar_chart(result.dropna(subset=["Score"]).set_index("Parameter")["Score"])
    st.caption("Score is a transparent screening score (1–4). It is not a regulatory water-quality index.")

with tab3:
    st.subheader("World map of uploaded observations")
    if {"Latitude", "Longitude"}.issubset(df.columns):
        map_df = df.dropna(subset=["Latitude", "Longitude"]).copy()
        if len(map_df):
            fig = px.scatter_geo(
                map_df, lat="Latitude", lon="Longitude",
                color="Water-quality classifier",
                hover_name="Location" if "Location" in map_df else None,
                hover_data=[c for c in ["Country"] + PARAMETERS if c in map_df.columns],
                projection="natural earth",
                title="Observed water-quality classifications"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid latitude/longitude records found.")
    else:
        st.info("Add `Latitude` and `Longitude` columns to your CSV for global observation mapping.")

    if show_history and HISTORY_FILE.exists():
        hist = pd.read_csv(HISTORY_FILE)
        fig2 = px.scatter_geo(
            hist, lat="Latitude", lon="Longitude", color="Severity",
            hover_name="Event", hover_data=["Country", "Year", "Issue Type"],
            projection="natural earth", title="Selected historical water-related issues"
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.subheader("Historical global water issues")
    if HISTORY_FILE.exists():
        hist = pd.read_csv(HISTORY_FILE)
        st.dataframe(hist, use_container_width=True, hide_index=True)
        st.caption("The bundled file is a starter dataset for demonstration. Expand it with documented events from authoritative public datasets before making research claims.")
        st.download_button("⬇️ Download historical issue dataset", hist.to_csv(index=False).encode(), "historical_water_issues.csv", "text/csv")
    else:
        st.warning("Historical dataset not found.")

with tab5:
    st.subheader("Optional supervised ML classifier")
    st.write("If your uploaded CSV contains a labeled target column such as `Water-quality classifier`, AquaClass can train a Random Forest and report validation metrics.")
    if uploaded is None:
        st.info("Upload a labeled CSV to train a model.")
    else:
        model, report, err = train_model(df)
        if err:
            st.warning(err)
        else:
            st.success("Model trained successfully.")
            accuracy = report.get("accuracy", np.nan)
            st.metric("Validation accuracy", f"{accuracy:.1%}" if not pd.isna(accuracy) else "n/a")
            st.dataframe(pd.DataFrame(report).T, use_container_width=True)
            # global feature importance
            imp = pd.Series(model.feature_importances_, index=PARAMETERS).sort_values(ascending=True)
            st.subheader("Global model parameter contribution")
            st.bar_chart(imp)

st.divider()
st.caption("AquaClass is a screening and educational decision-support application. Classifications depend on the thresholds and/or training data supplied. Do not use it as a substitute for accredited laboratory testing, regulatory compliance, or public-health decisions.")
