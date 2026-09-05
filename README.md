# SIH Problem 26033 - AI Mandi Vegetable Price & Anomaly Detector

An AI Price Detector & Anomaly Monitoring Solution for **Smart India Hackathon (SIH Problem 26033)** trained on **100% Authentic Indian Government APMC Mandi / Agmarknet Data**.

The system detects:
1. **Accurate Fair Mandi Benchmark Price** (INR per quintal and INR per kg).
2. **Artificial Price Inflation & Hoarding Alerts** (Price surges exceeding seasonal economic supply benchmarks).
3. **Distress Selling & Price Crashes** (When farmer market quotes drop below sustainable production baselines).

---

## 📂 Project Structure

```
sih_price_detector/
├── Monthly_data_cmo.csv                  # Official Govt APMC raw dataset (62,429 rows)
├── mandi_vegetable_pricing_dataset.csv   # Cleaned & enriched vegetable dataset (12,644 rows)
├── mandi_vegetable_price_detector.ipynb  # Interactive Jupyter Notebook (EDA, Models, Anomaly Engine)
├── mandi_vegetable_price_detector.pkl    # Serialized Champion Model Pipeline (Gradient Boosting)
├── train_mandi_model.py                  # Standalone training & pickling script
├── predict_mandi_price.py                # Standalone inference & audit report script
└── README.md                             # Documentation & user guide
```

---

## 🏛️ Authentic Government Mandi Dataset

- **Origin:** Maharashtra State Agricultural Marketing Board / Directorate of Marketing & Inspection (Agmarknet), Government of India.
- **Total Records:** 62,429 records (Raw), 12,644 records (Vegetable subset).
- **Geographic Coverage:** 111 APMC Mandis across 22 Agricultural Districts (including Lasalgaon, Pune, Solapur, Nashik, Ahmadnagar, Nagpur, Vashi, etc.).
- **Vegetable Staples Covered:**
  - Onion, Tomato, Potato, Cauliflower (`Flower`), Brinjal (`Baingan`), Cabbage, Ladies Finger (`Bhindi`), Capsicum, Carrot, Cucumber, Bitter Gourd (`Karela`), Cluster Bean (`Gawar`), Ginger, Garlic, Green Peas, Pumpkin, Bottle Gourd, Spinach, etc.
- **Key Features:**
  - `APMC`: Mandi market name
  - `Commodity`: Specific vegetable crop
  - `district_name` & `state_name`: Geographic location
  - `arrivals_in_qtl`: Daily market arrival volume in quintals (1 quintal = 100 kg)
  - `min_price` & `max_price`: Daily mandi auction spread
  - `modal_price`: Representative wholesale transaction price (Target)
  - `modal_price_per_kg`: Retail/unit benchmark
  - `season`: `Monsoon`, `Winter/Rabi`, `Summer/Zaid`
  - `price_detector_status`: Ground truth classification (`Normal / Fair Mandi Price`, `Inflated Price / Hoarding Alert`, `Distress Glut / Price Crash`)

---

## 📊 Model Evaluation Results

Trained on an 80/20 train/test split:

| Model | R² Score | MAE (INR / Quintal) | MAE (INR / Kg) | RMSE (INR / Quintal) | MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | **0.9912** | **₹73.84** | **₹0.74 / kg** | **₹140.42** | **6.62%** | **Champion (Pickled)** |
| **Random Forest Regressor** | 0.9874 | ₹79.03 | ₹0.79 / kg | ₹168.27 | 4.64% | Runner-up |

> [!NOTE]
> The model achieves an average prediction error of **only ₹0.74 per kilogram** on real mandi vegetable transactions across Maharashtra!

---

## 🚀 Quick Usage Guide

### 1. Run the Jupyter Notebook
Open `mandi_vegetable_price_detector.ipynb` in VS Code, Jupyter Notebook, or Colab:
```bash
jupyter notebook mandi_vegetable_price_detector.ipynb
```

### 2. Retrain and Pickle the Model
```bash
python train_mandi_model.py
```

### 3. Run Inference on Any Mandi Vegetable
```bash
python predict_mandi_price.py
```

### 4. Load the Pickled Model in Your Python Code
```python
import pickle
import pandas as pd

# Load pickled artifact
with open("mandi_vegetable_price_detector.pkl", "rb") as f:
    artifact = pickle.load(f)

pipeline = artifact["pipeline"]

# Input mandi details
lot_data = pd.DataFrame([{
    "Commodity": "Onion",
    "APMC": "Lasalgaon",
    "district_name": "Nashik",
    "season": "Winter/Rabi",
    "arrivals_in_qtl": 4500.0,
    "month_num": 12,
    "commodity_season_median": 1300.0,
    "min_price": 1100.0,
    "max_price": 1500.0,
    "price_spread_qtl": 400.0
}])

# Predict fair benchmark modal price
pred_qtl = pipeline.predict(lot_data)[0]
print(f"Fair Benchmark: INR {pred_qtl:,.2f}/quintal (~INR {pred_qtl/100.0:.2f}/kg)")
```
