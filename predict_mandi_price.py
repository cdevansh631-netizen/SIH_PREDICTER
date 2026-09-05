import os
import sys
import pickle
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def load_mandi_detector(model_path="mandi_vegetable_price_detector.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found! Run train_mandi_model.py first.")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    return artifact

def detect_mandi_price(commodity, apmc, district, season, month_num, arrivals_qtl,
                       observed_price_qtl, min_price=None, max_price=None, artifact=None):
    """
    Evaluates a Mandi vegetable price against the AI expected benchmark.
    Detects whether the market is experiencing:
    - Normal / Fair Mandi Price
    - Artificial Price Inflation / Hoarding Alert (Surge)
    - Distress Selling / Price Crash (Below Fair Value)
    """
    if artifact is None:
        artifact = load_mandi_detector()

    pipe = artifact["pipeline"]
    medians_lookup = artifact["medians_lookup"]
    
    # Retrieve historical seasonal median baseline
    season_median = medians_lookup.get((commodity, season), 2000.0)

    # If min/max prices not specified, estimate reasonable spread around observed
    if min_price is None:
        min_price = observed_price_qtl * 0.90
    if max_price is None:
        max_price = observed_price_qtl * 1.10
        
    spread = max_price - min_price

    input_row = pd.DataFrame([{
        "Commodity": commodity,
        "APMC": apmc,
        "district_name": district,
        "season": season,
        "arrivals_in_qtl": arrivals_qtl,
        "month_num": month_num,
        "commodity_season_median": season_median,
        "min_price": min_price,
        "max_price": max_price,
        "price_spread_qtl": spread
    }])

    predicted_modal_qtl = float(pipe.predict(input_row)[0])
    predicted_modal_kg = round(predicted_modal_qtl / 100.0, 2)
    observed_kg = round(observed_price_qtl / 100.0, 2)

    # Deviation percentage from predicted benchmark
    dev_pct = round(((observed_price_qtl - predicted_modal_qtl) / predicted_modal_qtl) * 100.0, 2)

    # Anomaly and Reasonability Verdict
    if dev_pct > artifact["thresholds"]["inflation_surge_pct"]:
        status = "INFLATED PRICE / HOARDING ALERT"
        alert_level = "HIGH RISK - Abnormal Market Markup / Supply Artificial Squeeze"
        action = "Issue regulatory mandi price advisory & verify cold-storage stock declarations."
    elif dev_pct < artifact["thresholds"]["distress_crash_pct"]:
        status = "DISTRESS SELLING / PRICE CRASH"
        alert_level = "HIGH RISK - Farmer Distress / Glut Exploitation"
        action = "Trigger Market Intervention Scheme (MIS) or minimum procurement price support."
    else:
        status = "NORMAL / FAIR MANDI PRICE"
        alert_level = "LOW RISK - In line with supply arrival volume and seasonal norms"
        action = "Market operations approved. Fair price for both farmers and buyers."

    return {
        "commodity": commodity,
        "apmc": apmc,
        "district": district,
        "season": season,
        "arrivals_qtl": arrivals_qtl,
        "observed_price_qtl": observed_price_qtl,
        "observed_price_kg": observed_kg,
        "predicted_fair_qtl": round(predicted_modal_qtl, 2),
        "predicted_fair_kg": predicted_modal_kg,
        "deviation_pct": dev_pct,
        "status": status,
        "alert_level": alert_level,
        "government_action": action
    }

def print_mandi_audit_report(report):
    print("\n" + "="*75)
    print("       APMC / AGMARKNET AI MANDI VEGETABLE PRICE AUDIT REPORT         ")
    print("="*75)
    print(f"Commodity:           {report['commodity']}")
    print(f"Mandi (APMC):        {report['apmc']} ({report['district']} District)")
    print(f"Season / Month:      {report['season']} (Month {report['arrivals_qtl']} qtl arrivals)")
    print(f"Daily Arrival Vol:   {report['arrivals_qtl']:,.1f} Quintals")
    print("-"*75)
    print(f"Observed Mandi Price: INR {report['observed_price_qtl']:,.2f}/qtl  (~INR {report['observed_price_kg']:.2f}/kg)")
    print(f"AI Fair Benchmark:    INR {report['predicted_fair_qtl']:,.2f}/qtl  (~INR {report['predicted_fair_kg']:.2f}/kg)")
    print(f"Price Deviation:      {report['deviation_pct']:+.2f}%")
    print(f"Detector Verdict:     {report['status']}")
    print(f"Risk Assessment:      {report['alert_level']}")
    print(f"Government Advisory:  {report['government_action']}")
    print("="*75)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, "mandi_vegetable_price_detector.pkl")
    detector = load_mandi_detector(model_file)
    print(f"Loaded: {detector['model_name']} trained on {detector['dataset_source']}")

    # Case 1: Lasalgaon Mandi (Nashik) - Onion during Normal Season
    rep1 = detect_mandi_price(
        commodity="Onion", apmc="Lasalgaon", district="Nashik",
        season="Winter/Rabi", month_num=12, arrivals_qtl=4500.0,
        observed_price_qtl=1350.0, min_price=1100.0, max_price=1500.0,
        artifact=detector
    )
    print_mandi_audit_report(rep1)

    # Case 2: Pune APMC - Tomato Artificial Price Spike / Hoarding
    rep2 = detect_mandi_price(
        commodity="Tomato", apmc="Pune", district="Pune",
        season="Monsoon", month_num=7, arrivals_qtl=1200.0,
        observed_price_qtl=3800.0, min_price=3400.0, max_price=4200.0,
        artifact=detector
    )
    print_mandi_audit_report(rep2)

    # Case 3: Potato Distress Selling / Crash below baseline
    rep3 = detect_mandi_price(
        commodity="Potato", apmc="Pune", district="Pune",
        season="Winter/Rabi", month_num=1, arrivals_qtl=9500.0,
        observed_price_qtl=420.0, min_price=350.0, max_price=500.0,
        artifact=detector
    )
    print_mandi_audit_report(rep3)
