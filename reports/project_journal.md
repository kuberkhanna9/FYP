
# Project Development Journal

## Day 1–3: Idea Brainstorming & Literature Review (June 20–22, 2025)

### Tasks Completed
- Researched common approaches to stock trend prediction using technical indicators
- Explored academic papers and Kaggle notebooks on AI + finance integration
- Compared rule-based vs machine learning vs deep learning models
- Identified gaps in current prediction tools – lack of explainability and adaptability

### Technical Direction
- Decided to start with 3 representative NASDAQ stocks (AAPL, TSLA, MSFT)
- Aim: Build an explainable AI system that predicts BUY / HOLD / SELL
- Planned to use Streamlit for UI and yfinance for data

## Day 4–5: Initial Technical Setup (June 23–24, 2025)

### Tasks Completed
- Created basic folder structure, data folders, and `notebooks/`
- Implemented `01_data_collection.ipynb` for initial stock downloading
- Set up basic preprocessing functions
- Initialized Git repo and logging system

### Technical Decisions
- Tools: Python, yfinance, pandas, matplotlib, Streamlit
- Scope: Predict short-term directional trends using classical indicators

## Day 6–7: Basic Feature Engineering (June 25–26, 2025)

### Tasks Completed
- Created custom functions for RSI, EMA, MACD
- Built `02_feature_engineering.ipynb` for indicator generation
- Generated enriched dataset with ~6 core features per stock

### Observations
- TSLA showed higher volatility patterns
- Feature computation verified manually vs known values

## Day 8–10: Model Planning + Initial UI Prototype (June 27–29, 2025)

### Tasks Completed
- Implemented basic Streamlit UI with stock selection and signal display
- Developed `04_label_generation_and_model_plan.ipynb` with simple labeling strategy
- Defined early plan for using Logistic Regression and XGBoost

## Day 11: Supervisor Meeting – Project Expansion (July 21, 2025)

### Supervisor: Dr. Usman

### Key Instructions
- Expand dataset to full NASDAQ-100 stocks
- Integrate sector ETFs for volatility tracking
- Adopt Shannon Mutual Information (SMI) for feature selection
- Move away from 3-stock toy model

### Actions Taken
- Created `01b_download_nasdaq_top100.ipynb` and `01c_download_sector_indices.ipynb`
- Started overhauling data architecture to support 100+ tickers
- Designed sector mapping CSV for stock-to-sector alignment

## Day 12–16: Full Pipeline Modernization (July 22–26, 2025)

### Tasks Completed
- Added sector ETF data, enriched data with ~20 indicators
- Created `01d_merge_and_validate_data.ipynb` for robust merging and inspection
- Updated `03b_feature_selection_smi.ipynb` for sector-wise SMI scoring
- Refactored all legacy code to support >95 stocks across 8–10 sectors

### Observations
- Some stocks (e.g., ATVI) had missing data (delisting or API issues)
- Feature enrichment successful, with ~200K rows total

## Day 17: Supervisor Check-in – Neural Network Direction (July 28, 2025)

### Supervisor: Dr. Usman (Virtual Meeting)

### Guidance
- Consider Neural Networks for modeling phase
- Evaluate if RNNs, LSTMs, or 1D CNNs add predictive power
- Maintain explainability (via SHAP, attention maps, etc.)
- Focus on clean, modular code and reproducibility

### Outcome
- Updated roadmap to include deep learning path alongside XGBoost
- Began planning neural architecture pipeline

## Day 18–20: Validation & Cleanup Phase (July 29–31, 2025)

### Tasks Completed
- Created checklist to validate each file in `/notebooks`
- Deprecated old notebooks (e.g., `01_data_collection.ipynb`)
- Verified output structure: `data/stocks`, `data/sectors`, `data/processed`
- Updated all sector, label, and feature files to support NASDAQ-100

## Day 21: Final Data Pipeline Cleanup (August 1, 2025)

### Tasks Completed
- Validated all outputs from 01–04 series
- Confirmed ~97 tickers processed correctly
- Ensured SMI scores, labels, and features were available for model training

## Day 22: Ready to Start Modeling Phase (August 2, 2025)

### Status
- Data Collection: Complete
- Feature Engineering: Complete
- Labeling: Complete
- Sector ETFs: Integrated
- SMI: Implemented
- Technical Cleanup: Done

### Next Steps
- Begin `05_model_baseline_training.ipynb` updates for sector-level models
- Start neural network modeling pipeline
- Add evaluation dashboard with time-series metrics
- Extend app frontend for sector selection and multi-model comparison