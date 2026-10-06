# E-commerce Customer Intelligence

**Customer Segmentation • Repurchase Prediction • Product Recommendation**

## Business Problem

E-commerce teams need a practical way to decide who to engage, when to engage them, and what to recommend. This compact case study converts transaction data into customer intelligence: behavioral segments for targeting, a 45-day repurchase score for prioritization, and explainable product recommendations. The outputs map directly to retention, reactivation, onboarding, and cross-sell actions.

## Solution

```text
E-commerce Data
      ↓
SQL Feature Engineering
      ↓
Customer Segmentation → Repurchase Prediction → Product Recommendation
      ↓
Business Action and Measurement
```

## Dataset

This repository uses **synthetic e-commerce data generated specifically for demonstration and interview purposes**. It contains 4,000 fictional customers, 180 products, 22,373 orders, and 52,993 order items covering January 2024–December 2025. No private, client, or real-company data is included, and the reported metrics are not real business results.

## Tech Stack

Python • SQL • Pandas • NumPy • scikit-learn • Matplotlib • SQLite • Jupyter

## Customer Segmentation

Readable SQLite CTEs transform normalized tables into customer-level RFM and behavioral features. After skew correction and standardization, K-Means diagnostics compare `k=2…6`. Five clusters are retained because they preserve actionable lifecycle groups while maintaining similar quantitative separation to nearby choices.

![Customer segments](assets/customer_segments.png)

The resulting segments are **High Value Loyal**, **Loyal Customers**, **New / Promising**, **At Risk**, and **Low Frequency**.

## Repurchase Prediction

Each training row is a temporal snapshot: features end at a cutoff, and the target asks whether the customer purchases in the following 45 days. All training label windows close before the final test cutoff, preventing future purchases from leaking into features.

Actual held-out results at the default 0.50 threshold:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.587 | 0.743 | 0.656 | 0.793 | 0.669 |
| Gradient Boosting | 0.656 | 0.617 | 0.636 | 0.811 | 0.698 |

For a low-cost retention campaign, a `0.44` gradient-boosting threshold raises recall to `0.702` with precision of `0.617`. This makes the business trade-off explicit instead of treating `0.50` as automatically optimal.

## Recommendation

The recommender combines each customer's category affinity with product popularity inside preferred categories. It is deliberately explainable and permits repeat products for replenishment use cases. A leave-last-order-out check produces **HitRate@3 = 19.8%** across 600 synthetic customers.

Example: customer `C2911` → **Home Item 02**, **Home Item 01**, **Home Item 03**.

## Business Use

**WHO** is the customer? → segment  
**WHEN** might they buy? → repurchase probability  
**WHAT** should we show? → recommended products  
**ACTION** → retention, reactivation, onboarding, or cross-sell  
**MEASURE** → campaign uplift, incremental revenue, and recommendation CTR

## Production Considerations

A production version could expose batch scores or a small API, package the scorer with Docker, and monitor PR-AUC, calibration, latency, errors, feature/category drift, prediction drift, repurchase rate, and revenue per customer. Online A/B testing is required to establish causal campaign and recommendation impact. Infrastructure is intentionally not implemented here.

## Repository Structure

```text
ecommerce-customer-intelligence/
├── README.md
├── requirements.txt
├── data/
│   ├── generate_synthetic_data.py
│   └── sample/
├── sql/customer_features.sql
├── notebooks/
│   ├── 01_customer_analytics_and_segmentation.ipynb
│   └── 02_repurchase_and_recommendation.ipynb
└── assets/
```

## Running the Project

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python data/generate_synthetic_data.py
jupyter notebook
```

Run the notebooks in numerical order. Both are saved with executed outputs so the complete case study can also be reviewed directly on GitHub.
