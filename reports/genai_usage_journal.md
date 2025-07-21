# Generative AI Usage Journal

This journal documents my transparent and minimal use of Generative AI tools during the development of my MSc AI final project: "AI-Driven Stock Trend Forecasting and Signal Generation Web App". The goal is to demonstrate responsible AI usage while maintaining academic integrity and original thinking.

## Day 1: Project Setup

### GenAI Tools Used
- Cursor IDE (basic code completion only)

### Purpose
- Project structure suggestions
- README.md formatting

### Manual Implementation
- Designed complete project architecture myself
- Wrote all configuration files from scratch
- Selected technologies based on personal research
- Implemented custom directory structure

### Reflection
Used Cursor's basic completions only for repetitive formatting. All technical decisions, project structure, and implementation choices were my own, based on my experience and research.

## Day 2: Data Collection
 
### GenAI Tools Used
- ChatGPT (minimal usage)

### Purpose
- Error message debugging
- yfinance API parameter verification

### Example Prompt
"What are the required parameters for yfinance's download function to fetch 5 years of daily data?"

### Manual Implementation
- Wrote complete data collection pipeline
- Implemented all cleaning functions
- Designed data quality checks
- Created custom CSV export format

### Reflection
GenAI was used only to verify API documentation. All data processing logic, error handling, and quality assurance were implemented based on my understanding of financial data requirements.

## Day 3: Technical Indicators

### GenAI Tools Used
- None

### Manual Implementation
- Researched and understood technical indicator formulas
- Implemented all indicators from scratch
- Created comprehensive testing suite
- Documented mathematical foundations

### Reflection
This day was completely manual as I wanted to ensure deep understanding of each technical indicator's mathematics and implementation details.

## Day 4: Visualization

### GenAI Tools Used
- Cursor IDE (basic code completion)
- ChatGPT (one-time usage)

### Purpose
- Matplotlib subplot layout syntax
- Color scheme suggestions

### Example Prompt
"What's the correct Matplotlib syntax for creating a 2x2 subplot with shared x-axis?"

### Manual Implementation
- Designed all visualization functions
- Created custom plotting styles
- Implemented interactive features
- Added comprehensive annotations

### Reflection
While GenAI helped with basic syntax, all visualization logic, design choices, and technical analysis interpretations were my own work.

## Day 5: Label Generation

### GenAI Tools Used
- Cursor IDE (basic code completion)

### Manual Implementation
- Designed label generation strategy
- Implemented forward returns calculation
- Created classification thresholds
- Built data validation checks

### Reflection
Used only basic code completion. All logic for label generation and classification strategy was developed based on my research and understanding of financial markets.

## Day 6: Baseline Models

### GenAI Tools Used
- ChatGPT (minimal usage)

### Purpose
- Scikit-learn parameter verification
- Cross-validation syntax check

### Example Prompt
"What's the correct parameter name for setting class weights in LogisticRegression?"

### Manual Implementation
- Designed complete ML pipeline
- Implemented all evaluation metrics
- Created custom validation splits
- Analyzed and interpreted results

### Reflection
GenAI was used only for API documentation verification. All model selection, feature engineering, and evaluation strategies were my original work.

## Day 7: XGBoost Implementation

### GenAI Tools Used
- Cursor IDE (basic code completion)
- ChatGPT (one-time usage)

### Purpose
- XGBoost hyperparameter range verification
- SHAP value plotting syntax

### Example Prompt
"What's the recommended range for XGBoost's max_depth parameter?"

### Manual Implementation
- Implemented complete XGBoost pipeline
- Designed hyperparameter search space
- Created custom evaluation metrics
- Analyzed feature importance

### Reflection
While GenAI helped verify some parameters, all model tuning, feature selection, and performance optimization were based on my knowledge and experimentation.

## Day 8: Backtesting Framework

### GenAI Tools Used
- Cursor IDE (basic code completion)

### Manual Implementation
- Designed backtesting architecture
- Implemented portfolio logic
- Created performance metrics
- Built visualization suite

### Reflection
Used only basic code completion. All backtesting logic, trading rules, and performance analysis were implemented based on my understanding of financial markets and trading systems.

## Overall Reflection

Throughout this project, I maintained a disciplined approach to using Generative AI tools:

1. **Minimal Usage**: GenAI was used primarily for:
   - Basic code completion
   - Documentation verification
   - Syntax checking
   - Formatting suggestions

2. **Original Work**: All critical components were my own:
   - System architecture
   - Trading logic
   - Technical indicators
   - Model selection
   - Performance analysis

3. **Academic Integrity**: GenAI served as a reference tool, not a solution provider:
   - Verified my understanding
   - Accelerated documentation
   - Simplified formatting
   - Debugged basic errors

This journal demonstrates my commitment to transparent and responsible use of AI tools while maintaining the academic integrity of my MSc project. The core intellectual work, including all critical decisions, implementations, and analyses, remains my own original contribution. 

###  Day 9 – GenAI Usage Log

####  Tools Used
- Cursor AI (Streamlit scaffolding + markdown cleanup)

####  Purpose
- Used to structure the initial Streamlit layout (sidebar + main panel)
- Reused previous code modules manually for loading model and predictions
- Asked ChatGPT for confirmation on SHAP plot integration using `st.pyplot`

####  Example Prompt
> "How do I display a matplotlib SHAP bar chart inside Streamlit?"

####  Outcome
- Final UI logic and visualizations coded independently
- Chart logic was based on earlier Jupyter notebooks and reused directly
- GenAI used only to confirm minor syntax and embed formatting

####  Reflection
- I manually modularized the logic to maintain code readability
- Prediction and signal flow were fully implemented by me using previous model pipeline 

###  Day 10 – GenAI Usage Log

####  Tools Used
- ChatGPT 4 (basic syntax clarification)
- Cursor AI (code block formatting)

####  Purpose
- Asked for SHAP plotting syntax compatible with Streamlit
- Confirmed how to get prediction probabilities from XGBoostClassifier using `predict_proba()`

####  Example Prompt
> "How to plot SHAP summary bar chart inside Streamlit?"
> "How do I extract predict_proba outputs from XGBoostClassifier?"

####  Outcome
- All logic to map signals to explanations written independently
- SHAP loading and matching to features implemented manually
- Layout and conditional text explanations coded without AI

####  Reflection
- GenAI helped with safe integration of SHAP but did not generate any logic
- Focused on creating original, user-centric visual feedback to align with academic expectations 

### Day 11 – GenAI Usage Log

#### Tools Used
- ChatGPT 4 (syntax clarification)
- Cursor AI (file creation automation)

#### Purpose
- Clarified format for `requirements.txt` and `setup.sh`
- Verified best practices for structuring a Streamlit Cloud deployment

#### Example Prompt
> "What dependencies do I include in requirements.txt for a Streamlit app using XGBoost, SHAP, and yfinance?"

#### Outcome
- Deployment steps were executed manually
- Streamlit setup and GitHub integration handled independently
- No app logic or core functionality was AI-generated

#### Reflection
- GenAI supported the deployment setup with minimal intervention
- The deployment process and testing flow was entirely designed and executed by me 