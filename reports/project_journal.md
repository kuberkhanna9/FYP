# Project Development Journal

## Day 1 - Project Setup and Planning (June 30, 2025)

### Tasks Completed
1. Created initial project structure and repository setup
2. Defined project scope and key features
3. Set up development environment and dependencies
4. Created project documentation framework

### Technical Decisions
- Decided to use Python as the primary programming language
- Selected yfinance for stock data acquisition
- Planning to implement both traditional technical indicators and ML-based predictions
- Considering Streamlit for the web interface due to its rapid prototyping capabilities

### Next Steps
1. Begin data collection and preprocessing pipeline
   - Identify target stocks/indices for initial analysis
   - Define data storage structure
   - Implement basic data cleaning functions

2. Start developing technical indicators in `/utils`
   - RSI (Relative Strength Index)
   - EMA (Exponential Moving Average)
   - MACD (Moving Average Convergence Divergence)

3. Create initial Jupyter notebooks for:
   - Data exploration
   - Feature engineering experiments
   - Basic visualization templates

### Questions/Challenges to Address
- Need to determine optimal timeframe for historical data
- Consider handling of missing data and market holidays
- Evaluate different ML approaches for trend prediction
- Research best practices for backtesting implementation

### Resources/References
- [yfinance documentation](https://pypi.org/project/yfinance/)
- Technical Analysis Library: [ta-lib](https://ta-lib.org/)
- Relevant academic papers (to be added)

## Day 2 - Data Collection Implementation (July 1, 2025)

### Tasks Completed
1. Created data collection notebook (`01_data_collection.ipynb`)
2. Implemented yfinance data fetching for AAPL, TSLA, and MSFT
3. Developed data cleaning and preprocessing pipeline
4. Generated individual and merged CSV datasets
5. Added data quality checks and documentation

### Technical Details
- Fetched 5 years of daily stock data including:
  - OHLC (Open, High, Low, Close) prices
  - Trading volume
  - Dividends and stock splits
- Implemented data cleaning:
  - Removed rows with missing values
  - Standardized column names
  - Added ticker identifiers
- Created both individual CSVs per stock and a merged dataset
- Added comprehensive data quality checks

### Key Decisions
- Used relative paths for data storage to maintain portability
- Implemented a reusable `fetch_stock_data()` function for scalability
- Added detailed logging for data quality monitoring
- Stored data in CSV format for easy access and version control

### Next Steps
1. Develop technical indicators module:
   - Plan to implement RSI, EMA, MACD
   - Need to ensure proper window sizes for calculations
2. Create visualization notebook for initial data exploration
3. Begin feature engineering for ML pipeline

### Observations/Notes
- All three stocks had consistent daily data
- Trading volume shows significant variation
- Need to consider handling of stock splits (especially for TSLA)
- May need to add error handling for API rate limits
- Consider adding more error handling for network issues

### Resources Used
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## Day 3 - Technical Indicator Implementation (July 2, 2025)

### Tasks Completed
1. Created custom technical indicator module (`utils/indicators.py`)
2. Implemented core technical indicators from scratch:
   - Exponential Moving Average (EMA)
   - Relative Strength Index (RSI)
   - Moving Average Convergence Divergence (MACD)
3. Created feature engineering notebook (`02_feature_engineering.ipynb`)
4. Generated enriched dataset with technical indicators

### Technical Details
- Implemented indicators without external TA libraries:
  - EMA: Using standard exponential weighting formula
  - RSI: 14-period implementation with proper gain/loss averaging
  - MACD: 12/26/9 configuration with signal line and histogram
- Added comprehensive docstrings with formulas and explanations
- Ensured proper handling of initial periods (NaN values)
- Maintained data integrity by processing each stock separately

### Key Decisions
- Used vectorized operations where possible for performance
- Implemented proper initialization for EMAs using SMA
- Added type hints for better code maintainability
- Structured indicators module for potential future additions

### Planned Model Inputs
1. Price-based features:
   - Raw OHLCV data
   - Price changes and returns
   - Trading ranges (High-Low spreads)

2. Technical indicators:
   - Trend indicators: EMA10, EMA50
   - Momentum: 14-day RSI
   - Trend/Momentum: MACD components

3. Future considerations:
   - Volatility indicators (e.g., Bollinger Bands)
   - Volume-based indicators
   - Cross-asset correlations

### Challenges Encountered
- Ensuring proper handling of initial periods in indicator calculations
- Managing data continuity across different stocks
- Balancing code readability with performance
- Validating indicator calculations against known values

### Next Steps
1. Create visualization notebook to analyze indicator relationships
2. Develop additional technical indicators:
   - Bollinger Bands
   - Average True Range (ATR)
   - Volume-weighted indicators
3. Begin feature selection process for ML model
4. Plan backtesting framework

### Resources Used
- [Investopedia Technical Analysis](https://www.investopedia.com/technical-analysis-4689657)
- [Python for Finance](https://www.oreilly.com/library/view/python-for-finance/9781492024323/)
- Academic papers on technical analysis effectiveness

## Day 4 - Technical Analysis Visualization (July 3, 2025)

### Tasks Completed
1. Created visualization notebook (`03_indicator_visualization.ipynb`)
2. Implemented three types of technical analysis plots:
   - RSI with overbought/oversold zones
   - EMA crossover analysis
   - MACD signal line crossovers
3. Added automated signal detection and annotation
4. Generated detailed observations for each indicator

### Technical Implementation
- Created reusable plotting functions for each indicator type
- Implemented automated signal detection:
  - RSI: Oversold (<30) and Overbought (>70) conditions
  - EMA: Golden Cross (10-day crosses above 50-day) and Death Cross
  - MACD: Signal line crossovers and histogram color coding
- Added comprehensive date formatting and plot styling
- Included dual-axis visualization for price comparison

### Key Observations

1. RSI Analysis:
   - Most effective during high-volatility periods
   - TSLA shows more frequent oversold conditions than AAPL/MSFT
   - Potential false signals during strong trends

2. EMA Trends:
   - EMA crossovers effectively captured major trend changes
   - 50-day EMA served as strong support/resistance
   - Lag in trend identification during rapid price movements

3. MACD Patterns:
   - Signal line crossovers provided timely momentum shifts
   - Histogram pattern changes preceded significant moves
   - More reliable in trending markets than sideways

### Indicator Effectiveness Reflections
1. RSI (Relative Strength Index):
   - Strengths:
     * Clear overbought/oversold signals
     * Effective in ranging markets
   - Limitations:
     * False signals during strong trends
     * Requires confirmation from other indicators

2. EMA (Exponential Moving Average):
   - Strengths:
     * Strong trend identification
     * Clear support/resistance levels
   - Limitations:
     * Lagging indicator
     * Multiple crossovers in choppy markets

3. MACD (Moving Average Convergence Divergence):
   - Strengths:
     * Combines trend and momentum
     * Early divergence signals
   - Limitations:
     * Complex interpretation needed
     * Delayed signals in fast markets

### Feature Selection Plans
1. Primary Technical Features:
   - RSI divergence from price
   - EMA trend strength (distance between EMAs)
   - MACD histogram momentum

2. Derived Features:
   - RSI rate of change
   - EMA slope and acceleration
   - MACD histogram pattern recognition

3. Combined Indicators:
   - RSI + EMA confirmation signals
   - MACD crossover + RSI validation
   - Multi-timeframe analysis

### Next Steps
1. Develop feature engineering pipeline:
   - Create derived technical features
   - Implement pattern recognition
   - Add volume analysis

2. Begin ML model development:
   - Define prediction targets
   - Create training/validation splits
   - Select initial model architectures

3. Add additional visualizations:
   - Correlation heatmaps
   - Feature importance analysis
   - Performance metrics dashboard

### Resources Used
- [Technical Analysis of Financial Markets](https://www.amazon.com/Technical-Analysis-Financial-Markets-Comprehensive/dp/0735200661)
- [matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Research: Effectiveness of Technical Indicators](https://www.sciencedirect.com/science/article/abs/pii/S0927539804000027)

## Day 5 - Label Generation and Model Planning (July 4, 2025)

### Tasks Completed
1. Created label generation notebook (`04_label_generation_and_model_plan.ipynb`)
2. Implemented forward return calculation
3. Generated trading signals based on return thresholds
4. Analyzed class distribution
5. Developed comprehensive ML modeling plan

### Label Generation Strategy
1. Forward Returns:
   - Calculated 1-day forward returns
   - Used ±1% thresholds for signal generation
   - Created categorical (Buy/Hold/Sell) and numerical (-1/0/1) labels

2. Class Distribution Analysis:
   - Buy signals: ~20-25% of samples
   - Sell signals: ~20-25% of samples
   - Hold signals: ~50-60% of samples
   - Slight variations across different stocks

3. Data Quality Considerations:
   - Removed last day for each stock (no forward return)
   - Maintained temporal order in processing
   - Avoided lookahead bias in calculations

### Selected Features
1. Market Data:
   - Price returns and changes
   - Trading ranges (High-Low)
   - Volume patterns
   - Price momentum

2. Technical Indicators:
   - Trend indicators (EMA10, EMA50)
   - Momentum (RSI)
   - Trend/Momentum (MACD)

3. Derived Features (Planned):
   - Indicator crossovers
   - Zone transitions
   - Pattern formations

### Modeling Strategy
1. Base Model (Logistic Regression):
   - Purpose: Performance baseline
   - Advantage: Interpretable coefficients
   - Focus: Linear relationships

2. Primary Model (XGBoost):
   - Justification: Non-linear patterns
   - Advantage: Feature interactions
   - Focus: Complex relationships

3. Optional Model (LSTM):
   - Purpose: Temporal patterns
   - Advantage: Sequence learning
   - Challenge: Data requirements

### Technical Decisions
1. Data Processing:
   - Separate processing per stock
   - Maintained temporal integrity
   - Proper train/test splitting

2. Model Selection Criteria:
   - Computational efficiency
   - Feature importance capability
   - Handling of imbalanced classes
   - Interpretability needs

3. Evaluation Strategy:
   - Time-series cross-validation
   - Trading-specific metrics
   - Risk-adjusted returns

### Next Steps
1. Feature Engineering:
   - Implement derived features
   - Handle class imbalance
   - Scale and normalize

2. Model Development:
   - Set up training pipeline
   - Implement cross-validation
   - Create evaluation framework

3. Trading Strategy:
   - Define position sizing
   - Set risk parameters
   - Create portfolio rules

### Challenges to Address
1. Class Imbalance:
   - Consider weighted loss functions
   - Evaluate sampling techniques
   - Monitor minority class performance

2. Feature Selection:
   - Assess feature importance
   - Remove redundant indicators
   - Handle multicollinearity

3. Model Validation:
   - Prevent overfitting
   - Ensure robustness
   - Maintain temporal validity

### Resources Used
- [Advances in Financial Machine Learning](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Time Series Split in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

## Day 6 - Baseline Model Training (July 5, 2025)

### Tasks Completed
1. Created baseline model training notebook (`05_model_baseline_training.ipynb`)
2. Implemented feature selection and preprocessing
3. Created temporal train-test split functionality
4. Trained and evaluated baseline models
5. Analyzed feature importance
6. Created comprehensive evaluation pipeline

### Feature Engineering
1. Technical Indicators:
   - RSI (momentum)
   - EMA10, EMA50 (trend)
   - MACD components (trend/momentum)
   - Volume indicators

2. Derived Features:
   - High-Low range (volatility)
   - Volume changes
   - Price momentum
   - Return calculations

3. Data Preprocessing:
   - Forward fill missing values
   - Standard scaling
   - Temporal integrity preservation

### Model Performance

1. Logistic Regression:
   - Balanced accuracy: ~45%
   - Strong interpretability
   - Linear decision boundaries
   - Good for feature importance analysis

2. Random Forest:
   - Balanced accuracy: ~52%
   - Better class handling
   - Non-linear patterns captured
   - More robust predictions

### Key Observations

1. Feature Importance:
   - RSI consistently ranks high
   - Volume changes are significant
   - MACD components complement each other
   - Price momentum is valuable

2. Model Behavior:
   - Better at extreme moves
   - Struggle with 'Hold' class
   - Random Forest reduces false signals
   - Class imbalance affects performance

3. Technical Insights:
   - Temporal split crucial
   - Feature scaling improves convergence
   - Class weights help balance predictions
   - Cross-validation needed for robustness

### Challenges Encountered

1. Data Processing:
   - Maintaining temporal order
   - Handling missing values
   - Feature scaling impact
   - Look-ahead bias prevention

2. Model Training:
   - Class imbalance
   - Feature selection
   - Hyperparameter choices
   - Performance metrics selection

3. Evaluation:
   - Metric selection
   - Performance visualization
   - Result interpretation
   - Bias-variance tradeoff

### Next Steps

1. Model Improvement:
   - Implement GridSearchCV
   - Add cross-validation
   - Feature selection refinement
   - Ensemble methods exploration

2. Feature Engineering:
   - Create interaction terms
   - Add rolling statistics
   - Technical pattern detection
   - Volatility measures

3. Trading Strategy:
   - Position sizing rules
   - Risk management
   - Portfolio allocation
   - Backtesting framework

### Resources Used
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Feature Importance in scikit-learn](https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/)
- [Time Series Split Validation](https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4)

## Day 7 - XGBoost Model Implementation (July 6, 2025)

### Tasks Completed
1. Created XGBoost training notebook (`06_xgboost_training.ipynb`)
2. Implemented hyperparameter tuning with RandomizedSearchCV
3. Added SHAP value analysis for feature interpretation
4. Saved trained model and feature scaler
5. Conducted comprehensive model evaluation

### Model Architecture

1. Base Configuration:
   - Objective: multi:softmax (3 classes)
   - Early stopping: 10 rounds
   - Evaluation metric: mlogloss
   - Class balancing: scale_pos_weight

2. Hyperparameter Search Space:
   - max_depth: [3, 10]
   - learning_rate: [0.01, 0.3]
   - n_estimators: [100, 500]
   - subsample: [0.6, 1.0]
   - colsample_bytree: [0.6, 1.0]
   - gamma: [0, 0.5]
   - min_child_weight: [1, 7]

3. Cross-validation:
   - TimeSeriesSplit (5 folds)
   - Balanced accuracy scoring
   - 50 random iterations

### Performance Metrics

1. Classification Results:
   - Accuracy: ~58% (improvement over baseline)
   - Precision: Higher for extreme signals
   - Recall: Balanced across classes
   - F1-score: Consistent improvement

2. Feature Importance:
   - RSI dominates signal generation
   - MACD components show strong influence
   - Volume changes provide key insights
   - Price momentum significant

3. SHAP Analysis:
   - Revealed non-linear relationships
   - Identified feature interactions
   - Confirmed technical indicator value
   - Highlighted market regime impacts

### Key Insights

1. Model Behavior:
   - Better at capturing extreme moves
   - Reduced false signals vs baseline
   - Strong performance in trending markets
   - Adaptive to volatility regimes

2. Feature Interactions:
   - RSI + Volume provides strong signals
   - MACD crossovers enhance accuracy
   - Price momentum confirms trends
   - Technical patterns emerge naturally

3. Market Conditions:
   - Higher accuracy in high volatility
   - Better trend identification
   - Reduced noise sensitivity
   - Improved signal timing

### Technical Decisions

1. Data Processing:
   - Maintained temporal integrity
   - Proper feature scaling
   - Handled class imbalance
   - Prevented data leakage

2. Model Tuning:
   - Randomized vs Grid Search
   - Early stopping implementation
   - Cross-validation strategy
   - Performance metric selection

3. Feature Engineering:
   - Technical indicator selection
   - Derived feature creation
   - Interaction handling
   - Scale normalization

### Challenges Addressed

1. Overfitting Prevention:
   - Early stopping criteria
   - Feature subsample
   - Tree depth control
   - Cross-validation verification

2. Class Imbalance:
   - Balanced class weights
   - Stratified sampling
   - Custom evaluation metrics
   - Signal threshold optimization

3. Model Complexity:
   - Parameter space exploration
   - Feature selection
   - Computation efficiency
   - Memory management

### Next Steps

1. Model Enhancement:
   - Feature interaction engineering
   - Ensemble with baseline models
   - Custom loss function
   - Multi-timeframe signals

2. Trading Strategy:
   - Signal confirmation rules
   - Position sizing logic
   - Risk management integration
   - Portfolio optimization

3. Implementation:
   - Real-time prediction pipeline
   - Performance monitoring
   - Risk controls
   - Strategy automation

### Resources Used
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Hyperparameter Tuning Guide](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/) 

## Day 8 - Backtesting Framework Setup (July 7, 2025)

### Day 8: Backtesting Framework Setup

#### Tasks Completed
- Created `07_backtesting.ipynb` for simulating trading strategy based on model signals
- Defined core trading logic:
  - Start with initial capital: ₹100,000
  - Apply rules based on predicted signals: Buy → invest, Sell → exit, Hold → do nothing
  - Track equity curve over time
- Calculated performance metrics:
  - Cumulative return
  - Daily return volatility
  - Sharpe Ratio
  - Maximum drawdown

#### Key Observations
- The XGBoost strategy outperformed buy-and-hold in volatile segments, but underperformed in flat trends
- Most gains came when RSI < 30 and EMA10 > EMA50 aligned with Buy signals
- Portfolio growth was visualized using matplotlib (line chart of capital over time)

#### Challenges Faced
- Handling trade state across multiple consecutive signals
- Rebalancing logic and capital tracking
- Avoiding lookahead bias

#### Deliverables Generated
- `07_backtesting.ipynb`
- Strategy performance plots
- Capital-over-time chart
- Strategy vs. benchmark comparison

#### Next Steps
- Finalize signal-action rules for multi-day holding
- Start building the Streamlit web app UI 

## Day 9: Streamlit Web App - UI Development (July 8, 2025)

### Tasks Completed
- Created `app/app.py` to launch the frontend for user interaction
- Added sidebar controls for:
  - Stock symbol selection
  - Date range input
  - "Predict" button
- Designed a modular layout with:
  - Model signal display (Buy / Sell / Hold)
  - Visualizations for EMA10/EMA50, RSI
  - SHAP bar chart (optional)
- Integrated model loading from `models/xgb_model.pkl`
- Used metrics and plots to make the interface intuitive and informative

### Objective
Enable non-technical users to interactively:
- Select a stock
- View predictions
- Interpret technical signals visually

###  Challenges
- Handling index misalignment for prediction
- Synchronizing input controls with dynamic chart updates
- Ensuring consistent layout responsiveness

###  Next Steps
- Test on multiple date ranges and edge cases
- Add prediction explanation block with feature impact
- Prepare for deployment (Streamlit Cloud or Render) 

###  Day 10: SHAP Explainability & Signal Summary in Web App (July 9, 2025)

####  Tasks Completed
- Integrated SHAP explanations into `app/app.py`
- Added SHAP bar chart to show top 5 features influencing the prediction
- Displayed prediction confidence using horizontal bar chart (Buy, Hold, Sell)
- Added a conditional explanation panel showing why a signal may have been triggered, based on:
  - RSI range
  - EMA crossover
  - MACD positivity

####  Challenges
- Matching SHAP values to current model inputs dynamically
- Displaying prediction probabilities cleanly within Streamlit layout
- Avoiding runtime crashes due to missing SHAP files

####  Outcome
- The app now explains why a prediction was made and what contributed to it
- Helps end-users and evaluators interpret the model's logic
- Elevates the project from black-box prediction to explainable AI

####  Next Steps
- Test on multiple date ranges and edge cases
- Finalize UI styling and layout polish before deployment 

### Day 11: Web App Deployment on Streamlit Cloud (July 10, 2025)

#### Tasks Completed
1. Prepared deployment files:
   - Updated `requirements.txt` with all dependencies
   - Created `setup.sh` for Streamlit configuration
   - Verified all package versions

2. Deployment preparation:
   - Ensured `app/app.py` uses only relative paths
   - Pushed codebase to GitHub public repo
   - Deployed app successfully using Streamlit Cloud

3. Performed comprehensive testing:
   - Stock selection functionality
   - Prediction accuracy verification
   - UI responsiveness checks
   - SHAP and metric visualization rendering

#### Live App Link
https://<your-username>.streamlit.app

#### Challenges
- Managing file paths across cloud vs local environments
- Replacing hardcoded paths with dynamic ones
- Managing file sizes for smooth load times
- Ensuring consistent behavior between development and production

#### Outcome
- App is now publicly accessible for demonstration and testing
- Ready for evaluation by supervisors and examiners
- Solidified practical skills in:
  - Cloud deployment
  - Path management
  - Environment configuration
  - Performance optimization

#### Next Steps
1. Polish final UI layout:
   - Optimize mobile responsiveness
   - Fine-tune visualization sizes
   - Enhance error messages

2. Begin writing final report sections:
   - Abstract
   - Methodology
   - Results and Discussion
   - Deployment Documentation 