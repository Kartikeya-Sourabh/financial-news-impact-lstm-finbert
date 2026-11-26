# Financial News Sentiment and Stock Price Prediction (LSTM + FinBERT)

A learning project that pairs a **PyTorch LSTM price forecaster** with **sentiment signals from financial tweets** (VADER-labelled classical ML models and FinBERT) to explore how news sentiment relates to stock prices.

> Not financial advice. This is an experiment, not a trading system.

## What it does

| Part | Method | Output |
|---|---|---|
| Price forecast | 2-layer LSTM (hidden size 50, 50-day window) on 5 years of AAPL daily closes | 30-day forecast with rough volatility, 95% interval and trend label |
| Sentiment models | VADER-labelled tweets, TF-IDF (5,000 features), Naive Bayes / Logistic Regression / Random Forest | Accuracy and per-class metrics on a 20% holdout |
| Sentiment signal | FinBERT (`ProsusAI/finbert`) over each stock's tweets, majority vote | Buy / Hold / Sell recommendation per ticker |

Data: about 80,800 tweets across 25 tickers (TSLA, MSFT, AAPL, AMZN, META, and others). AAPL prices (Feb 2020 to Feb 2025) are pulled with `yfinance`.

## Results (saved outputs of the original run)

**Sentiment classifiers** (80/20 split, 16,159 held-out tweets, labels from VADER):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes | 0.70 | 0.64 |
| Random Forest | 0.80 | 0.75 |
| **Logistic Regression** | **0.87** | **0.85** |

**AAPL LSTM** (10 epochs, Adam, lr 0.001, MSE loss): training loss fell from 0.026 to about 0.0006. The 30-day forecast was about 250.06 USD (+1.20%), classified as a mild bullish trend. FinBERT's majority vote for AAPL tweets was **Sell**, so the two signals disagree on this run.

## Limitations (please read)

- **Labels come from VADER, not humans**, so the 0.87 accuracy measures agreement with VADER, not true sentiment accuracy.
- **The LSTM and FinBERT signals are shown side by side, not fused** into one model. Testing whether sentiment features improve out-of-sample forecasts is the main next step.
- The LSTM uses closing price only and has **no time-based validation/test split**.
- The included backtest is **in-sample** with no costs. It returned 113.57% versus 242.19% for buy-and-hold, so it does not show a tradable edge.
- The Sharpe ratio and confidence interval are rough approximations.

## Project structure

```
.
├── financial_news_lstm_finbert.ipynb   # full notebook with saved outputs
├── requirements.txt
├── data/
│   └── README.md                       # where to put stock_tweets.csv
└── .gitignore
```

## How to run

```bash
git clone https://github.com/<your-username>/financial-news-impact-lstm-finbert.git
cd financial-news-impact-lstm-finbert
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put stock_tweets.csv in data/ (see data/README.md)
jupyter notebook financial_news_lstm_finbert.ipynb
```

The first run downloads NLTK resources and the FinBERT model (about 440 MB). A GPU is optional.

## Next steps

- Walk-forward, time-based train/validation/test evaluation.
- Feed sentiment features into the LSTM and compare out-of-sample error with and without them.
- Evaluate sentiment models on a human-labelled finance dataset.

## Tech

Python, PyTorch, scikit-learn, NLTK (VADER), Hugging Face Transformers (FinBERT), yfinance, pandas, Matplotlib.
