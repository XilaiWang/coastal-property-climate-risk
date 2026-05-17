# Coastal Property Climate Risk Pricing

> **How much does coastal erosion risk discount a property's market price?**  
> A machine learning approach to quantifying environmental risk in mortgage portfolios.

---

## Why This Project Matters

British banks hold billions in mortgage assets along England's eroding coastline. As climate change accelerates coastal retreat, properties near erosion zones face declining values, rising insurance costs, and reduced mortgage availability. If left unmeasured, this environmental risk premium becomes a systemic threat to lenders.

This project simulates the work of a **risk management analyst at a UK commercial bank**. The brief: build a predictive model that estimates the *true* market price of a coastal property after accounting for erosion risk, so the bank can identify overvalued assets in its mortgage book before they become bad debts.

**The core question:** For every metre closer to a predicted erosion zone, what discount does the market already price in?

---

## What This Project Demonstrates

### End-to-End Data Science Workflow

| Stage | What I Did |
|---|---|
| **Data Integration** | Merged 3 UK government open datasets (HM Land Registry, Ordnance Survey NSPL, Defra NCERM) totaling ~1.7M transaction records |
| **Spatial Engineering** | Broke a cross-agency data silo by converting 800K+ string postcodes into geographic coordinates, then computing metre-level distances to erosion boundaries |
| **Data Cleaning** | Diagnosed and handled real-world messiness: headerless CSVs, terminated postcodes (~53% join loss with bias analysis), extreme outliers (£1 symbolic transfers), right-skewed price distributions |
| **Feature Engineering** | Selected predictive features while explicitly removing high-cardinality ID-leakage columns (door numbers, individual postcodes) that would cause memorisation |
| **Exploratory Analysis** | Produced 5 publication-quality visualisations confirming the non-linear, threshold-based relationship between erosion proximity and price |
| **Model Iteration** | Progressed from naive baselines through ensemble methods with rigorous justification at each step |
| **Pipeline Engineering** | Built a production-ready `sklearn.pipeline.Pipeline` with `GridSearchCV` — the workflow a data scientist hands to a credit risk team |
| **Validation Rigour** | Fixed random seeds globally, documented a temporal pooling vs. out-of-time validation trade-off, and applied 3-fold cross-validation throughout |

### Technical Stack

`Python` · `pandas` · `GeoPandas` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn` · `shapely` · `NumPy` · `SciPy`

---

## Results

### Model Performance (log-transformed price prediction)

| Model | RMSE | R² | Strategy |
|---|---|---|---|
| LinearRegression (baseline) | 0.3734 | 0.5624 | — |
| DecisionTree (depth=3, baseline) | 0.4655 | 0.3199 | — |
| **RandomForest + GridSearchCV** ★ | **0.3338** | **0.6503** | Bagging |
| XGBoost + GridSearchCV | 0.3401 | 0.6371 | Boosting |

### Key Business Insight

The relationship between erosion proximity and property price is **non-linear with threshold effects** — prices don't drop gradually as you approach the coast. Instead, there's a measurable discount zone where the market suddenly prices in environmental risk. A linear model cannot capture this; ensemble tree methods can.

Bagging (Random Forest) outperformed Boosting (XGBoost) because coastal property data has inherently high variance from macro-level confounders (interest rates, local demand shocks). Bagging's variance-reduction mechanism is better suited to this signal-to-noise environment.

---

## Reproducibility

### Quick Start

```bash
# 1. Clone and install dependencies
git clone https://github.com/XilaiWang/coastal-property-climate-risk.git
cd coastal-property-climate-risk
pip install -r requirements.txt

# 2. Download raw data (not included due to GitHub file size limits)
#    - HM Land Registry Price Paid Data (2020, 2025): https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
#    - Place both CSV files in the project root

# 3. Open the notebook
jupyter notebook Coastal_Property_Climate_Risk_Pricing_IB9JV.ipynb
```

The notebook auto-installs dependencies in its first cell and reads from pre-computed intermediate files (`processed/*.gpkg`) to ensure full reproducibility without raw data dependency.

### Data Sources

| Dataset | Provider | Purpose |
|---|---|---|
| Price Paid Data (PPD) | HM Land Registry | ~1.7M property transactions with prices |
| National Statistics Postcode Lookup (NSPL) | ONS / Ordnance Survey | Postcode → coordinate mapping |
| National Coastal Erosion Risk Mapping (NCERM) | Defra / Environment Agency | Predicted erosion zones to 2100 |

---

## Project Context

- **Course:** IB9JV Programming for Data Analytics, Warwick Business School (2025–2026)
- **Scenario:** Risk management analyst at a British commercial bank
- **Assessment:** Full decision-record trail documenting every analytical choice with statistical justification

---

*Built by Xilai Wang — interested in data science and analytics roles where rigorous methodology meets real business risk.*
