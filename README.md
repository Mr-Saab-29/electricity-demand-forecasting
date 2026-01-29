# Electricity Demand Forecasting — Global, Regional & Temporal Models

This project implements a **robust, leakage-safe electricity demand forecasting pipeline** for France, covering **national and regional (perimeter-level) demand** across multiple forecasting horizons.

The goal is to **compare strong statistical baselines, feature-based machine learning models, and a foundational temporal Transformer**, under a consistent backtesting framework.

---

## 🔍 Problem Statement

Given historical electricity consumption data (national + 12 regions) and weather signals, we aim to forecast demand for multiple horizons (1h → 168h) while answering:

- How far can **feature-engineered tree models** go?
- Do **global models transfer across regions**?
- Can a **shared Temporal Transformer** learn regional dynamics without manual lag engineering?

---

## 🧠 Models Implemented

### Baselines
- **Seasonal Naive**
  - Same-hour, same-day historical repetition

### Machine Learning
- **LightGBM (Global)**
  - Single model trained across all regions
- **LightGBM + Weather**
  - Adds meteorological exogenous variables
- **LightGBM Perimeter-Specific**
  - One model per region (bias–variance tradeoff study)

### Deep Learning
- **Temporal Transformer Encoder**
  - Shared encoder across all regions
  - Multi-horizon output head
  - No handcrafted lag features
  - Perimeter-wise target normalization (train-only statistics)

---

## 🏗 Project Structure

.
├── Data/
│ ├── processed/
├── src/
│ ├── backtesting.py
│ ├── config.py
│ ├── data_loader.py
│ ├── evaluate_model.py
│ ├── features.py
│ ├── init.py
│ ├── metrics.py
│ ├── splits.py
│ ├── models/
│ │ ├── init.py
│ │ ├── lgbm.py
│ │ ├── lgbm_perimeter.py
│ │ ├── seasonal_naive.py
│ │ └── temporal_transformer.py
│ ├── train_transformer.py
│ ├── windowing.py
├── outputs/
│ ├── metrics/
│ ├── predictions/
│ └── models/
│ └── plots/
├── run_backtest.py
├── requirements.txt
└── README.md


---

---

## 🔁 Backtesting Methodology

- **Walk-forward anchor evaluation**
- Weekly anchors (`stride = 7 days`)
- Forecast horizons:  
  `1, 2, ..., 24, 48, 72, 96, 120, 144, 168`
- Metrics:
  - MAE
  - RMSE
  - MAPE
- No data leakage:
  - Train-only statistics
  - Strict split boundaries

---

## 🚀 How to Run

### 1️⃣ Create & activate environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate     # Windows
```

To install the dependecies

```bash
pip install -r requirements.txt # Install dependencies
```
To run baselines and LightGBM

```bash
python -m run_backtest # Run Baselines and LightGBM
```

To run the temporal transformer 

```bash
python -m src.train_transformer
```
Compute Transformer metrics

```bash
python -m src.evaluate_model
```

Run again the backtest to compile all the metrics of the transformer with the baselines and LightGBM

## 📊Results
| Model                | MAPE (%) |
| -------------------- | -------- |
| LightGBM + Weather   | **1.76** |
| LightGBM Global      | 1.82     |
| LightGBM Perimeter   | 1.90     |
| Temporal Transformer | 6.25     |
| Seasonal Naive       | 7.33     |

##  📌Key Takeaways
  - Weather is a critical signal
  - Global models generalise well across regions
  - Perimeter specific models reduce RMSE but increase variance
  - Temporal Transformers require more data, stronger inductive bias or hybrid architectures

##  🔮Future Works
  - PatchTST/Informer style architectures
  - Region Embeddings
  - Probablisitic Forecasting 
  - Multi task heads
  - Error decomposition by region

##  Reproducibility Checklist
1. Install requirements using pip as given by instructions above
2. Strongly recommend not to freeze PyTorch wheels directly, Instead install it separetely 

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

