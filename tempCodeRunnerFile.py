import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def train_and_pickle_mandi_model():
    script_dir = r"C:\Users\cdeva\.gemini\antigravity\scratch\sih_price_detector"
    dataset_path = os.path.join(script_dir, "mandi_vegetable_pricing_dataset.csv")
    model_output_path = os.path.join(script_dir, "mandi_vegetable_price_detector.pkl")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    print("="*70)
    print("TRAINING MANDI VEGETABLE PRICE DETECTOR ON AUTHENTIC GOVT DATASET")
    print("="*70)
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} authentic vegetable mandi records across {df['Commodity'].nunique()} vegetables.")
    print(f"Mandis: {df['APMC'].nunique()} | Districts: {df['district_name'].nunique()}")

    # Feature definitions
    categorical_features = ["Commodity", "APMC", "district_name", "season"]
    numerical_features = [
        "arrivals_in_qtl",
        "month_num",
        "commodity_season_median",
        "min_price",
        "max_price",
        "price_spread_qtl"
    ]
    target = "modal_price"

    X = df[categorical_features + numerical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\nTrain set: {X_train.shape[0]} records | Test set: {X_test.shape[0]} records")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=16, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.1, max_depth=5, random_state=42
        )
    }

    best_pipeline = None
    best_r2 = -float("inf")
    best_model_name = ""
    metrics = {}

    for name, regressor in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        metrics[name] = {
            "R2": round(float(r2), 4),
            "MAE_INR_qtl": round(float(mae), 2),
            "MAE_INR_kg": round(float(mae / 100.0), 2),
            "RMSE_INR_qtl": round(float(rmse), 2),
            "MAPE_pct": round(float(mape), 2)
        }

        print(f"\nModel: {name}")
        print(f"  R2 Score:      {r2:.4f}")
        print(f"  MAE (Quintal): INR {mae:,.2f}")
        print(f"  MAE (per Kg):  INR {mae / 100.0:,.2f}/kg")
        print(f"  RMSE:          INR {rmse:,.2f}")
        print(f"  MAPE:          {mape:.2f}%")

        if r2 > best_r2:
            best_r2 = r2
            best_pipeline = pipe
            best_model_name = name

    print("\n" + "="*70)
    print(f"CHAMPION MODEL: {best_model_name} (R2 = {best_r2:.4f})")
    print("="*70)

    # Compute commodity baseline medians for inference lookup
    medians_lookup = df.groupby(["Commodity", "season"])["modal_price"].median().to_dict()

    artifact = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "feature_columns": categorical_features + numerical_features,
        "metrics": metrics,
        "medians_lookup": medians_lookup,
        "commodities": sorted(df["Commodity"].unique().tolist()),
        "mandis": sorted(df["APMC"].unique().tolist()),
        "districts": sorted(df["district_name"].unique().tolist()),
        "thresholds": {
            "inflation_surge_pct": 25.0,  # > 25% deviation from seasonal median -> Hoarding/Surge
            "distress_crash_pct": -25.0   # < -25% deviation -> Distress Glut/Crash
        },
        "dataset_source": "Government of Maharashtra APMC / Agmarknet Mandi Records (62k rows)",
        "version": "2.0.0",
        "sih_problem": "SIH Problem 26033 - AI Mandi Vegetable Price & Anomaly Detector"
    }

    with open(model_output_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\n[SUCCESS] Mandi model successfully pickled to:")
    print(f"--> {model_output_path}")
    print(f"File size: {os.path.getsize(model_output_path) / 1024:.2f} KB")

if __name__ == "__main__":
    train_and_pickle_mandi_model()
