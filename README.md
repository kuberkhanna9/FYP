# AI-Driven Stock Trend Forecasting and Signal Generation Web App

This repository contains the implementation of my MSc AI final project:  
**“AI-Driven Stock Trend Forecasting and Signal Generation Web App”**.  

The project combines **Random Forest** and **xLSTM-TS** to forecast stock movements and generate **BUY/SELL signals**. It also integrates **explainability** (SHAP, feature importance, attention mechanisms) and **financial backtesting** to evaluate strategy performance.  

---

## Project Overview

- **Objective**: Build a scalable, explainable, and sector-aware system for stock signal generation.  
- **Scope**:
  - Forecast BUY/SELL for NASDAQ-100 stocks.  
  - Integrate sector ETFs for market context.  
  - Use technical indicators + Shannon Mutual Information (SMI) for feature selection.  
  - Benchmark models: Random Forest, Enhanced xLSTM, ARIMA, KNN.  
  - Provide backtesting metrics (Sharpe ratio, cumulative return, drawdown).  
- **Outcome**: A full pipeline deployed in a Streamlit app, supported by a robust backend.  

---

## Repository Structure

```bash
AI & CS/
├── __pycache__/                 # Python cache files
├── cache_live/                  # Cached parquet stock/ETF data
├── outputs/                     # Generated plots, charts, predictions CSVs
│
├── app.py                       # Streamlit frontend app (UI for signals & plots)
├── data.py                      # Helper script for structured dataset handling
├── final_test.py                 # Manual test script for end-to-end validation
├── quick_test.py                 # Manual script for quick debugging
│
├── signals_backend.py           # Core backend pipeline (data, indicators, models, backtesting)
├── model_comparison_study.ipynb # Inspector-driven comparative analysis (RF, xLSTM, ARIMA, KNN)
│
├── Gen_AI_Journal.md            # Transparent record of AI tool usage (minimal)
├── Project_Journal.md           # Weekly project development log
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
│
├── test_enhanced_performance.py # AI-generated test script to verify performance
├── test_enhanced_xlstm.py       # AI-generated test script for xLSTM model
├── test_metrics_fix.py          # AI-generated test script for metrics consistency
├── test_scenario.py             # AI-generated test script for scenario testing
├── test_signals_backend.py      # AI-generated unit test for backend logic
├── test_simplified_pipeline.py  # AI-generated simplified pipeline tests

## Installation & Setup

1. **Clone the repository**
```bash
git clone https://git.cs.bham.ac.uk/projects-2024-25/kxk157
cd kxk157
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501.

## File Descriptions

### Core Project Files

**app.py** – Streamlit frontend:
- Ticker selection and sector analysis.
- Displays technical indicators, signals, backtesting results.
- Visualises comparison charts: Strategy vs Buy & Hold.

**signals_backend.py** – Core backend:
- Downloads and caches NASDAQ-100 + ETF data (cache_live/).
- Implements 20+ technical indicators (RSI, MACD, Bollinger, ATR, OBV, ADX, etc.).
- Adds ETF volatility/returns as context features.
- Implements Random Forest baseline and Enhanced xLSTM model.
- Provides backtesting metrics and GARCH volatility analysis.

**model_comparison_study.ipynb** – Comparative analysis:
- Created after inspector feedback (August 20, 2025).
- Benchmarks RF + xLSTM against ARIMA and KNN.
- Computes metrics (Accuracy, RMSE, MAE, MAPE).
- Plots predicted vs actual for model comparisons.

**data.py** – Helper script for dataset formatting and validation.

**final_test.py / quick_test.py** – Manual scripts for end-to-end and quick debugging.

### Documentation

**Gen_AI_Journal.md** – Weekly record of AI usage (minimal, mostly syntax/test files).

**Project_Journal.md** – Weekly project development log (with supervisor & inspector meetings).

**README.md** – This file, containing full documentation.

### Test Files (AI-Generated)

These files were created using Generative AI to test my manually written logic. They do not contain original project code. They were used only for validation/debugging:

- **test_enhanced_performance.py** – Tests performance outputs.
- **test_enhanced_xlstm.py** – Tests xLSTM training pipeline.
- **test_metrics_fix.py** – Tests metric calculations (RMSE, MAE, etc.).
- **test_scenario.py** – Runs scenario tests on pipelines.
- **test_signals_backend.py** – Unit tests for backend logic.
- **test_simplified_pipeline.py** – Simplified version to validate pipeline steps.

## Workflow

1. **Data Collection** → Stocks + ETF data (via signals_backend.py).
2. **Feature Engineering** → Indicators + ETF context features.
3. **Model Training** → Random Forest baseline + Enhanced xLSTM.
4. **Evaluation** → Metrics (Accuracy, RMSE, MAE, MAPE), backtesting results.
5. **Comparative Study** → model_comparison_study.ipynb (ARIMA, KNN).
6. **Visualization & UI** → app.py Streamlit dashboard.

## Results

- **RF Baseline**: ~45% accuracy, solid baseline.
- **Enhanced xLSTM**: ~80% accuracy, well-calibrated confidence.
- **ARIMA & KNN**: Lower performance, included for benchmarking.
- **Backtesting**: AI strategy produced higher Sharpe ratios than Buy & Hold on select tickers.

## Acknowledgements

**Supervisor**: Dr. Usman Ilyas – directed project scope (NASDAQ-100, Neural Networks, explainability).

**Inspector**: Dr. Syed Fawad Hussain – requested comparative model study, leading to model_comparison_study.ipynb.

**Tools**: Python, Streamlit, PyTorch, scikit-learn, yfinance, arch, SHAP, matplotlib.

## Notes on Generative AI Usage

- AI tools were used only for syntax clarifications, scaffolding journals/markdown, and creating test files.
- All core project code (app.py, signals_backend.py, model_comparison_study.ipynb) was manually implemented.
- Test files (test_*.py) were AI-generated to validate my logic and did not contribute to the main pipeline.

This ensures originality, transparency, and academic integrity as stated in the GenAI Usage Policy by the University of Birmingham.