<div align="center">

# 🧠 StockMind — Demand Forecasting & Inventory Decision Support System

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-Regressor-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Supply%20Chain-Decision%20Engine-purple?style=for-the-badge">
</p>

<p>
  <i>"Predict Demand. Optimize Inventory. Make Smarter Decisions."</i>
</p>

---

**Author**: [Nguyễn Nhật Long](https://github.com/nlong05013456-oss) & Nguyễn Trần Quang Huy  
**Advisor**: Th.S Nguyễn Văn Chức | **Team**: Nhóm 2  

</div>

---

## 📌 Project Overview

**StockMind AI** is a production-style Data Science & Supply Chain Decision Support System engineered to predict multi-series demand across 100 Store-Product pairs (`SKU_ID = Store_ID + "_" + Product_ID`) and compute mathematically optimal replenishment orders.

```text
               RAW RETAIL DATA
                      │
                      ▼
         1. LEAKAGE-FREE PREPROCESSING
          Lag & Rolling Features (Train-only P99 clip)
                      │
                      ▼
         2. TRAIN-FITTED DEMAND SEGMENTATION
          KMeans Clustering (Slow / Medium / Fast Moving)
                      │
                      ▼
         3. TEMPORAL MODEL BENCHMARKING
          Naive (120.20) | XGBoost (88.24) | Random Forest (87.89 🏆)
                      │
                      ▼
         4. INVENTORY DECISION ENGINE
          Horizon Demand + Safety Stock (Z · σ · √LeadTime) - Current Inventory
                      │
                      ▼
         5. INTERACTIVE STREAMLIT DASHBOARD
          Decision Matrix | What-If Simulation | Risk Alerts (DOI)
```

---

## 🏗️ 3-Layer System Architecture

| Layer | Component | Description & Mathematical Foundation |
| :--- | :--- | :--- |
| **Layer 1** | **Supervised Demand Forecasting** | Predicts daily demand using temporal lag/rolling features (`Lag_1`, `Lag_7`, `Lag_14`, `Rolling_Mean_7/30`, `Rolling_STD_7/30`, promo & calendar features). Benchmarks Naive, Random Forest, and XGBoost on a temporal test split (`Date > 2023-06-30`). |
| **Layer 2** | **Inventory Planning Engine** | Computes **Expected Demand** over the planning horizon and **Safety Stock** based on forecast uncertainty:<br>$$\text{Expected Demand} = \text{Daily Forecast} \times \text{Planning Horizon}$$<br>$$\text{Safety Stock} = Z \times \sigma_{\text{error}} \times \sqrt{\text{Lead Time}}$$ |
| **Layer 3** | **Decision & Risk Engine** | Calculates optimal replenishment quantity and classifies stock risk using Days of Inventory ($\text{DOI}$):<br>$$\text{Recommended Order} = \max\Big(0, \text{Expected Demand} + \text{Safety Stock} - \text{Current Inventory}\Big)$$ |

---

## 📊 Model Benchmark Results (Temporal Test Set)

Models are evaluated on an independent temporal test split (`Date > 2023-06-30`):

| Model | Test MAE | Test RMSE | Status |
| :--- | :---: | :---: | :---: |
| **Naive Baseline** ($\hat{y}_t = y_{t-1}$) | 120.20 | 153.44 | Baseline |
| **XGBoost Regressor** | 88.24 | 118.94 | Candidate |
| **Random Forest Regressor** | **87.89** | **118.83** | **🏆 Selected Production Model** (-26.9% MAE vs Naive) |

---

## 💡 Fallback Error Hierarchy & Risk Engine

To compute accurate Safety Stock across sparse multi-series SKUs, StockMind AI applies a **3-tier Error Fallback Hierarchy**:
$$\text{SKU Error } \sigma_{\text{error}} \xrightarrow{\text{fallback}} \text{Cluster Error } \sigma_{\text{error}} \xrightarrow{\text{fallback}} \text{Global Error } \sigma_{\text{error}}$$

### Days of Inventory (DOI) Risk Alerts:
- 🔴 **Stock-Out Risk**: $\text{DOI} < \text{Lead Time}$
- 🟡 **Low Stock**: $\text{DOI} < \text{Lead Time} + \text{Safety Days}$
- 🟢 **Optimal**: $\text{DOI} \le \text{Planning Horizon} \times 1.5$
- 🔵 **Overstocked**: $\text{DOI} > \text{Planning Horizon} \times 1.5$

---

## 📂 Project Directory Structure

```text
Cap1/
├── .gitignore                 # Git ignore configuration
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── unprocessed/           # Raw dataset (retail_store_inventory.csv)
│   └── processed/             # Cleaned datasets & segment mappings
│       ├── clean_retail_data.csv
│       └── product_segments.csv
│
├── models/                    # Production Model Artifacts
│   ├── final_model.pkl        # Best trained model (Random Forest)
│   ├── business_logic.pkl     # Test residual metrics & Z-scores
│   ├── clustering_model.pkl   # KMeans model
│   └── cluster_scaler.pkl     # StandardScaler model
│
├── notebooks/                 # Exploratory Data Analysis & Prototyping
│   ├── 1.0-EDA.ipynb
│   ├── 2.0-clustering-products.ipynb
│   └── 3.0-model-prototyping.ipynb
│
└── src/                       # Core Source Code Modules
    ├── data_processing.py     # Module 1: Preprocessing & Leakage-Free Features
    ├── clustering.py          # Module 2: KMeans Product Segmentation (Train-only)
    ├── train.py               # Module 3: Temporal Split & Benchmark Training
    ├── inventory_engine.py    # Module 4: 3-Layer Supply Chain Decision Engine
    └── predict.py             # Module 5: Interactive Streamlit Dashboard
```

---

## ⚡ Quick Start Guide

### 1. Environment Setup
```bash
git clone https://github.com/nlong05013456-oss/Cap1.git
cd Cap1
pip install -r requirements.txt
```

### 2. Run Data Preprocessing Pipeline
```bash
python src/data_processing.py
```

### 3. Run Product Demand Segmentation
```bash
python src/clustering.py
```

### 4. Train Models & Execute Benchmark
```bash
python src/train.py
```

### 5. Launch StockMind AI Dashboard
```bash
python -m streamlit run src/predict.py
```

---

<div align="center">

Developed with ❤️ by [Nguyễn Nhật Long](https://github.com/nlong05013456-oss)

</div>
