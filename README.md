# Chargebee Subscription Analytics 📊

A powerful Python CLI tool that fetches all account data from the Chargebee API and runs rule-based analytics to surface business insights. Detect churn risk, forecast revenue, predict customer lifetime value (CLV), analyze payment failures, segment customers, and more.

## Features ✨

### 🚀 Data Fetching
- **Comprehensive:** Fetches all 36 Chargebee entity types (customers, subscriptions, invoices, transactions, etc.)
- **Efficient:** Uses async HTTP with configurable concurrency and rate limiting.
- **Incremental:** Supports incremental sync via `updated_at` timestamp tracking.
- **Streaming:** Streams data directly to a local SQLite database to handle large datasets with minimal memory usage.
- **Dual Output:** Outputs to both JSON files and a SQLite database for flexibility.

### 📈 Analytics (12 Analyzers)

All analyzers are rule-based and require no external ML libraries, making them lightweight and easy to deploy.

| Analyzer | Description |
|---|---|
| **Churn** 🚨 | Identifies customers at risk of churn based on payment failures, dunning status, recent downgrades, and low engagement signals. |
| **Revenue** 💰 | Forecasts MRR/ARR trends using time series analysis, detecting growth rates and seasonality. |
| **CLV** 💎 | Predicts Customer Lifetime Value using historical revenue, purchase frequency, and expected tenure. |
| **Payment** 💳 | Scores payment failure risk based on transaction history, method reliability, and card expiration. |
| **Segmentation** 🧩 | Groups customers into RFM (Recency, Frequency, Monetary) segments to target high-value or at-risk users. |
| **Expansion** 🚀 | Identifies upsell/cross-sell opportunities based on usage growth, plan limits, and feature adoption. |
| **Refund** 💸 | Analyzes refund patterns and credit notes to identify product issues or high-maintenance customers. |
| **Anomaly** 📉 | Detects billing anomalies like unusual spikes, drops in revenue, or suspicious transaction patterns. |
| **Coupon** 🎟️ | Evaluates coupon campaign effectiveness, ROI, and redemption rates. |
| **Cohort** 👥 | Tracks retention and churn rates by signup cohort (monthly/quarterly). |
| **Events** 📜 | Mines event logs for sequence patterns that precede key customer actions (churn, upgrade). |
| **Deduplication** 👯 | Finds duplicate customer records using fuzzy matching on names, emails, and company details. |

## Requirements 📋

- Python 3.11+
- A Chargebee account with API access

## Installation 🛠️

```bash
# Clone the repository
git clone https://github.com/<your-org>/chargebee-subscription-analytics.git
cd chargebee-subscription-analytics

# Set up a virtual environment
python -m venv .venv
# Activate:
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration ⚙️

Create a `.env` file in the project root:

```env
CHARGEBEE_API_KEY=your_api_key_here
CHARGEBEE_SITE=your-site-name

# Optional Tuning
CHARGEBEE_MAX_CONCURRENCY=20      # Max concurrent API requests
CHARGEBEE_PAGE_SIZE=100           # Records per page (max 100)
CHARGEBEE_MAX_RETRIES=5           # Retries on rate limits/errors
CHARGEBEE_RATE_LIMIT_RPM=150      # API Rate limit (requests per minute)
CHARGEBEE_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
CHARGEBEE_OUTPUT_DIR=output       # Directory for data and results
```

## Usage 🚀

### 1. Fetch Data
First, download your Chargebee data to the local database.

```bash
python -m chargebee_mapper fetch
```

This will:
- Connect to the Chargebee API.
- Fetch all entity types (customers, subscriptions, invoices, etc.).
- Stream records to `output/chargebee_data.db` (SQLite).
- Export JSON files to `output/json/` for easy inspection.
- Generate a summary report in `output/summary.json`.

### 2. Run Analytics
Once data is fetched, run the analyzers to generate insights.

```bash
# Run all 12 analyses (recommended)
python -m chargebee_mapper analyze all

# Run specific analyses
python -m chargebee_mapper analyze churn
python -m chargebee_mapper analyze revenue
python -m chargebee_mapper analyze segmentation
```

Results are saved as JSON files in `output/analysis/`.

### Available Analysis Commands

| Command | Description |
|---|---|
| `analyze all` | Run all available analyzers |
| `analyze churn` | Predict churn risk scores |
| `analyze revenue` | Forecast MRR and growth |
| `analyze clv` | Calculate Customer Lifetime Value |
| `analyze payment` | Analyze payment failure risks |
| `analyze segmentation` | Perform RFM segmentation |
| `analyze expansion` | Find upsell opportunities |
| `analyze refund` | Analyze refund trends |
| `analyze anomaly` | Detect billing anomalies |
| `analyze coupon` | Analyze coupon performance |
| `analyze cohort` | Run cohort retention analysis |
| `analyze events` | Mine event sequence patterns |
| `analyze deduplication` | Find duplicate records |

## Troubleshooting 🔧

- **Rate Limits:** If you hit API rate limits (HTTP 429), reduce `CHARGEBEE_MAX_CONCURRENCY` or `CHARGEBEE_RATE_LIMIT_RPM` in `.env`.
- **Memory Usage:** The tool uses streaming to handle large datasets. If you still encounter memory issues, ensure you are running the `fetch` command which uses the optimized SQLite storage backend.
- **Missing Data:** Ensure your API key has "Read-Only" access to all resources. Some entities (like "Entitlements" or "Omnichannel") may require specific Chargebee plan features enabled.

## Project Structure 📂

```
chargebee_mapper/
  __main__.py              # CLI entry point
  client.py                # Async HTTP client with rate limiting
  fetcher.py               # Orchestrates parallel data fetching
  storage.py               # SQLite and JSON storage backend
  utils.py                 # Shared utility functions

  # Analyzers
  churn_analyzer.py        # Churn risk scoring
  revenue_forecaster.py    # MRR/ARR forecasting
  clv_predictor.py         # CLV prediction
  payment_failure_predictor.py # Payment risk analysis
  customer_segmentation.py # RFM segmentation
  ... (other analyzers)
```

## License 📄

MIT
