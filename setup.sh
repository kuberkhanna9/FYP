mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
\n\
[theme]\n\
base = 'light'\n\
primaryColor = '#6c63ff'\n\
backgroundColor = '#ffffff'\n\
secondaryBackgroundColor = '#f0f2f6'\n\
textColor = '#262730'\n\
" > ~/.streamlit/config.toml 