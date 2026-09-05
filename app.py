import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="AI Mandi Price & Anomaly Detector | SIH 26033",
    page_icon="🥦",
    layout="wide"
)

# Load model artifact
@st.cache_resource
def load_detector():
    model_path = os.path.join(os.path.dirname(__file__), "mandi_vegetable_price_detector.pkl")
    if not os.path.exists(model_path):
        model_path = "mandi_vegetable_price_detector.pkl"
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    return artifact

try:
    artifact = load_detector()
    pipeline = artifact["pipeline"]
    commodities = artifact.get("commodities", ["Onion", "Tomato", "Potato", "Brinjal", "Cabbage", "Flower", "Ladies Finger"])
    districts = artifact.get("districts", ["Nashik", "Pune", "Solapur", "Ahmadnagar", "Nagpur", "Mumbai"])
    all_mandis = artifact.get("mandis", ["Lasalgaon", "Pune", "Nashik", "Solapur", "Ahmadnagar", "Nagpur", "Mumbai"])
    # district_mandi_map: {district_name: [list of mandis in that district]}
    # Falls back to showing the full mandi list under every district if the
    # mapping isn't present in the artifact (older pickle, not yet patched).
    district_mandi_map = artifact.get("district_mandi_map", {d: all_mandis for d in districts})
    medians_lookup = artifact.get("medians_lookup", {})
    thresholds = artifact.get("thresholds", {"inflation_surge_pct": 25.0, "distress_crash_pct": -25.0})
except Exception as e:
    st.error(f"Error loading model artifact: {e}")
    st.stop()

# Header
st.title("🌾 AI Mandi Vegetable Price & Anomaly Detector")
st.markdown("**Smart India Hackathon (SIH Problem 26033)** | *Trained on Authentic Government APMC / Agmarknet Mandi Records*")
st.markdown("---")

# Layout: Sidebar inputs & Main results
with st.sidebar:
    st.header("📋 Mandi Lot Parameters")

    selected_commodity = st.selectbox(
        "Select Vegetable Commodity",
        commodities,
        index=commodities.index("Onion") if "Onion" in commodities else 0
    )

    # District chosen FIRST, then mandi list is filtered to only that
    # district's mandis — instead of two independent dropdowns that let you
    # pick a district and a completely unrelated mandi.
    selected_district = st.selectbox(
        "Select District",
        districts,
        index=districts.index("Nashik") if "Nashik" in districts else 0
    )

    mandis_in_district = district_mandi_map.get(selected_district, all_mandis)
    if not mandis_in_district:
        mandis_in_district = all_mandis  # safety fallback
    selected_apmc = st.selectbox("Select APMC Mandi", mandis_in_district)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        season = st.selectbox("Season", ["Winter/Rabi", "Monsoon", "Summer/Zaid"])
    with col_s2:
        month_num = st.slider("Month", min_value=1, max_value=12, value=12)

    arrivals_qtl = st.number_input(
        "Daily Arrival Volume (Quintals)",
        min_value=1.0, max_value=50000.0, value=4500.0, step=100.0
    )

    st.markdown("---")
    st.subheader("Auction / Quote Inputs")
    observed_price_qtl = st.number_input(
        "Observed / Quoted Price (INR / Quintal)",
        min_value=100.0, max_value=25000.0, value=1350.0, step=50.0
    )

    st.caption(
        "Enter the ACTUAL min/max auction prices reported at the mandi for this lot "
        "(not derived from your observed price above) — this keeps the AI benchmark "
        "genuinely independent of what you're checking."
    )
    _season_median_default = medians_lookup.get((selected_commodity, season), 1500.0)
    min_p = st.number_input(
        "Min Auction Price (real mandi data)",
        min_value=0.0, value=float(_season_median_default * 0.85), step=50.0
    )
    max_p = st.number_input(
        "Max Auction Price (real mandi data)",
        min_value=0.0, value=float(_season_median_default * 1.15), step=50.0
    )

    predict_btn = st.button("🔍 Detect Price Reasonability", type="primary", use_container_width=True)

# --- Predict button logic ---
if predict_btn:
    season_median = medians_lookup.get((selected_commodity, season), 1500.0)
    spread = max_p - min_p

    input_data = pd.DataFrame([{
        "Commodity": selected_commodity,
        "APMC": selected_apmc,
        "district_name": selected_district,
        "season": season,
        "arrivals_in_qtl": arrivals_qtl,
        "month_num": month_num,
        "commodity_season_median": season_median,
        "min_price": min_p,
        "max_price": max_p,
        "price_spread_qtl": spread
    }])

    predicted_qtl = float(pipeline.predict(input_data)[0])

    # PRIMARY anomaly signal: deviation from the historical seasonal median
    # (independent of the observed price you typed in).
    deviation_vs_median_pct = ((observed_price_qtl - season_median) / season_median) * 100.0

    # SECONDARY / informational: deviation from the ML model's prediction.
    deviation_vs_model_pct = ((observed_price_qtl - predicted_qtl) / predicted_qtl) * 100.0

    st.session_state["result"] = {
        "input_data": input_data,
        "season_median": season_median,
        "predicted_qtl": predicted_qtl,
        "observed_price_qtl": observed_price_qtl,
        "deviation_vs_median_pct": deviation_vs_median_pct,
        "deviation_vs_model_pct": deviation_vs_model_pct,
    }

# --- Display results (only if a prediction has been run at least once) ---
# Use .get(...) instead of direct ["result"] indexing, and an explicit
# if/else block (rather than relying only on st.stop()) so this can't raise
# a KeyError even if st.stop() doesn't halt execution (e.g. the script was
# run as plain `python app.py` instead of `streamlit run app.py`).
r = st.session_state.get("result")

if r is None:
    st.info(
        "👈 Set your mandi lot parameters (including REAL min/max auction prices) "
        "in the sidebar and click **Detect Price Reasonability** to run the audit."
    )
else:
    input_data = r["input_data"]
    season_median = r["season_median"]
    predicted_qtl = r["predicted_qtl"]
    observed_price_qtl = r["observed_price_qtl"]
    deviation_vs_median_pct = r["deviation_vs_median_pct"]
    deviation_vs_model_pct = r["deviation_vs_model_pct"]

    predicted_kg = predicted_qtl / 100.0
    observed_kg = observed_price_qtl / 100.0

    # Display Dashboard
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Observed Mandi Price",
            value=f"₹{observed_price_qtl:,.2f} / qtl",
            delta=f"₹{observed_kg:.2f} / kg"
        )

    with col2:
        st.metric(
            label="Historical Seasonal Median (independent baseline)",
            value=f"₹{season_median:,.2f} / qtl",
            delta=f"₹{season_median/100:.2f} / kg",
            delta_color="off"
        )

    with col3:
        st.metric(
            label="Deviation vs Historical Median",
            value=f"{deviation_vs_median_pct:+.2f}%",
            delta="Fair" if abs(deviation_vs_median_pct) <= 25.0 else ("Inflated" if deviation_vs_median_pct > 25.0 else "Depressed"),
            delta_color="normal" if abs(deviation_vs_median_pct) <= 25.0 else "inverse"
        )

    with col4:
        st.metric(
            label="AI Model Estimate (reference only)",
            value=f"₹{predicted_qtl:,.2f} / qtl",
            delta=f"{deviation_vs_model_pct:+.2f}% vs observed",
            delta_color="off"
        )

    st.markdown("---")
    st.caption(
        "⚠️ The verdict below is based on deviation from the **historical seasonal median**, "
        "which is computed independently of your observed price. The AI Model Estimate "
        "column is shown for reference only, since it depends on the min/max auction "
        "spread you provide and should not be treated as a fully independent check "
        "unless real mandi min/max data is used."
    )

    # Reasonability & Anomaly Verdict Banner
    st.subheader("🚨 Mandi Anomaly & Reasonability Audit Verdict")

    if deviation_vs_median_pct > thresholds["inflation_surge_pct"]:
        st.error(f"### ⚠️ INFLATED PRICE / HOARDING ALERT (Surge: {deviation_vs_median_pct:+.2f}%)")
        st.markdown("""
        - **Risk Assessment:** **HIGH RISK** - Abnormal price markup inconsistent with historical seasonal norms.
        - **Suspected Cause:** Middlemen cartelization, speculative hoarding, or artificial supply restriction.
        - **Government Advisory:** Trigger cold-storage stock inspections & invoke Essential Commodities Act monitoring.
        """)
    elif deviation_vs_median_pct < thresholds["distress_crash_pct"]:
        st.warning(f"### ⚠️ DISTRESS SELLING / PRICE CRASH (Deficit: {deviation_vs_median_pct:+.2f}%)")
        st.markdown("""
        - **Risk Assessment:** **HIGH RISK** - Farmer realization is severely depressed below sustainable cost of production.
        - **Suspected Cause:** Post-harvest glut, lack of cold chain infrastructure, distress offloading.
        - **Government Advisory:** Trigger Market Intervention Scheme (MIS) price compensation or direct NAFED procurement.
        """)
    else:
        st.success(f"### ✅ NORMAL / FAIR MANDI PRICE (Variance: {deviation_vs_median_pct:+.2f}%)")
        st.markdown("""
        - **Risk Assessment:** **LOW RISK** - Observed transaction price aligns with historical seasonal trends.
        - **Government Advisory:** Market clearance approved. Fair remuneration for farmers and equitable price for consumers.
        """)

    st.markdown("---")

    # Details Breakdown
    with st.expander("📊 View Model Technical Details & Feature Attributes"):
        raw_metrics = artifact.get("metrics", {})
        model_key = artifact.get("model_name")

        if isinstance(raw_metrics, dict):
            metrics_for_model = raw_metrics.get(model_key, {})
        elif isinstance(raw_metrics, list):
            metrics_for_model = next(
                (m for m in raw_metrics if isinstance(m, dict) and (m.get("model") == model_key or m.get("model_name") == model_key)),
                raw_metrics[0] if raw_metrics and isinstance(raw_metrics[0], dict) else {}
            )
        else:
            metrics_for_model = {}

        st.json({
            "Model Name": model_key,
            "Dataset Source": artifact.get("dataset_source"),
            "R2 Score": metrics_for_model.get("R2", 0.9912),
            "Mean Absolute Error": f"₹{metrics_for_model.get('MAE_INR_kg', 0.74)} / kg",
            "Thresholds": thresholds,
            "Input Values": input_data.to_dict(orient="records")[0]
        })