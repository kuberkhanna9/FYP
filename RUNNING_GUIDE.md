# AI-Driven Stock Trend Forecasting and Signal Generation Web App
## Running Guide

This guide provides comprehensive instructions for running and understanding the MSc AI project that predicts stock market trends using machine learning and technical analysis.

## 1. Project Overview

This project combines technical analysis with machine learning to predict stock market trends (Buy/Sell/Hold signals) for selected stocks. It features:

- Technical indicator calculation (RSI, EMA, MACD)
- XGBoost model for signal prediction
- SHAP values for model explainability
- Interactive Streamlit web interface
- Comprehensive backtesting framework

The system analyzes historical stock data, computes technical indicators, and generates trading signals with explanations.

## 2. Environment Setup

Follow these steps to set up your development environment:

```bash
# Clone the repository
git clone <repo-url>
cd ai-stock-signals

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: If deploying to Streamlit Cloud
chmod +x setup.sh  # On Unix/MacOS
./setup.sh
```

## 3. File & Folder Structure

```
project_root/
├── app/
│   └── app.py                 # Streamlit web interface
├── data/
│   ├── merged_stocks.csv      # Raw stock data
│   └── enriched_merged_stocks.csv  # Processed data with indicators
├── models/
│   ├── xgb_model.pkl         # Trained XGBoost model
│   └── shap_values.pkl       # Pre-computed SHAP values
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_indicator_visualization.ipynb
│   ├── 04_label_generation_and_model_plan.ipynb
│   ├── 05_model_baseline_training.ipynb
│   └── 06_xgboost_training.ipynb
├── reports/
│   ├── project_journal.md    # Development log
│   └── genai_usage_journal.md  # AI usage transparency
├── utils/
│   └── indicators.py         # Technical indicator functions
├── requirements.txt          # Python dependencies
└── setup.sh                 # Streamlit Cloud configuration
```

## 4. Running the Notebooks

Execute the notebooks in numerical order to reproduce the full analysis pipeline:

1. **01_data_collection.ipynb**
   - Downloads historical data for AAPL, TSLA, MSFT
   - Creates merged dataset in `/data/`

2. **02_feature_engineering.ipynb**
   - Implements technical indicators
   - Generates enriched dataset with features

3. **03_indicator_visualization.ipynb**
   - Visualizes technical indicators
   - Analyzes indicator relationships

4. **04_label_generation_and_model_plan.ipynb**
   - Creates Buy/Sell/Hold labels
   - Defines prediction targets

5. **05_model_baseline_training.ipynb**
   - Trains initial models
   - Evaluates baseline performance

6. **06_xgboost_training.ipynb**
   - Trains XGBoost model
   - Generates SHAP explanations

7. **07_backtesting.ipynb**
   - Implements trading simulation
   - Calculates performance metrics
   - Visualizes strategy results

## 5. Running the Web App

Launch the Streamlit interface locally:

```bash
cd app
streamlit run app.py
```

### App Features

1. **Sidebar Controls**
   - Stock selector (AAPL, TSLA, MSFT)
   - Date range picker
   - Predict button

2. **Main Panel**
   - Current stock information
   - Prediction signal (Buy/Sell/Hold)
   - Confidence metrics
   - Signal explanation

3. **Visualizations**
   - Price chart with EMAs
   - RSI indicator
   - SHAP feature importance
   - Prediction probability bars

## 6. Deployment

To deploy on Streamlit Cloud:

1. Push your code to a public GitHub repository
2. Visit https://share.streamlit.io/
3. Connect your GitHub repository
4. Configure deployment:
   - Set main file as `app/app.py`
   - Verify requirements.txt
   - Check all data files are included

### Deployment Checklist
- [ ] All paths in app.py are relative
- [ ] Required data files are in repository
- [ ] setup.sh is configured correctly
- [ ] requirements.txt includes all dependencies

## 7. Final Notes

### Customization
- Modify stock tickers in `01_data_collection.ipynb`
- Adjust technical indicators in `utils/indicators.py`
- Retrain model using training notebooks

### Documentation
- Development progress in `reports/project_journal.md`
- AI usage documented in `reports/genai_usage_journal.md`
- Model performance metrics in training notebooks

### Support
For issues or questions:
1. Check the project documentation
2. Review relevant notebooks
3. Examine the project journals

### Performance Notes
- App loads faster with pre-computed SHAP values
- Model retraining may take significant time
- Consider data size when modifying stock selection

---

*Note: This project was developed as part of an MSc AI program, focusing on combining traditional technical analysis with modern machine learning approaches.* 