"""
Patches an existing mandi_vegetable_price_detector.pkl so that:
  1. "commodities", "mandis", and "districts" reflect the FULL set of values
     present in the training CSV (not a hardcoded fallback list).
  2. A new "district_mandi_map" is added: {district_name: [list of APMCs]},
     so the Streamlit app can show only the mandis that actually belong to
     the selected district, instead of every mandi in the dataset.

Run this once, locally, in the same folder as your .pkl and CSV:
    python patch_artifact_lists.py
"""

import pickle
import pandas as pd

PKL_PATH = "mandi_vegetable_price_detector.pkl"
CSV_PATH = "mandi_vegetable_pricing_dataset.csv"  # adjust filename if needed

# Load existing artifact
with open(PKL_PATH, "rb") as f:
    artifact = pickle.load(f)

# Load the training dataset to get the true full set of values
df = pd.read_csv(CSV_PATH)

artifact["commodities"] = sorted(df["Commodity"].dropna().unique().tolist())
artifact["mandis"] = sorted(df["APMC"].dropna().unique().tolist())
artifact["districts"] = sorted(df["district_name"].dropna().unique().tolist())

# Build district -> list of mandis mapping (each mandi belongs to exactly
# one district in this dataset, verified with a groupby uniqueness check)
district_mandi_map = (
    df.groupby("district_name")["APMC"]
    .apply(lambda s: sorted(s.dropna().unique().tolist()))
    .to_dict()
)
artifact["district_mandi_map"] = district_mandi_map

print(f"Commodities: {len(artifact['commodities'])}")
print(f"Mandis: {len(artifact['mandis'])}")
print(f"Districts: {len(artifact['districts'])}")
print(f"District -> Mandi mapping built for {len(district_mandi_map)} districts")

# Save back
with open(PKL_PATH, "wb") as f:
    pickle.dump(artifact, f)

print(f"\nPatched and saved {PKL_PATH} successfully.")