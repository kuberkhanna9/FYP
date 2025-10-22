import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Download VIX index data from Yahoo Finance
vix = yf.download("^VIX", start="2007-01-01", end="2023-12-31")

# Create figure
plt.figure(figsize=(10,5))
plt.plot(vix.index, vix["Close"], color="navy", label="VIX Index (Volatility)")

# Highlight major events
events = {
    "2008 GFC": "2008-09-15",
    "US-China Trade War": "2018-06-01",
    "COVID-19 Crash": "2020-03-15"
}
for event, date in events.items():
    plt.axvline(pd.to_datetime(date), color="red", linestyle="--", alpha=0.7)
    plt.text(pd.to_datetime(date), 60, event, rotation=90, color="red", fontsize=9, va="bottom")

# Styling
plt.title("Global Stock Market Volatility (VIX Index, 2007–2023)", fontsize=13)
plt.xlabel("Year")
plt.ylabel("VIX Level")
plt.ylim(0, 90)
plt.legend(loc="upper right")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
