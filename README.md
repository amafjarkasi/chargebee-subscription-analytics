# Chargebee Subscription Analytics

A Python CLI tool that fetches all account data from the Chargebee API and runs rule-based analytics to surface business insights -- churn risk, revenue forecasting, customer lifetime value, payment failures, segmentation, and more.

## Features

### Data Fetching
- Fetches all 36 Chargebee entity types (customers, subscriptions, invoices, transactions, etc.)
- Async HTTP with configurable concurrency and rate limiting
- Paginated iteration with exponential backoff and jitter on retries
- Outputs to both JSON files and a SQLite database
- Incremental sync support via `updated_at` timestamp tracking

### Analytics (12 Analyzers)

| Analyzer | Description |
|---|---|
| **Churn** | Rule-based churn risk scoring using payment failures, dunning status, billing recency, tenure, and value signals |
| **Revenue** | MRR/ARR time series forecasting with trend detection and compound monthly growth rate |
| **CLV** | Customer lifetime value prediction using discount-factor model with tenure, recency, and payment history |
| **Payment** | Payment failure risk scoring from transaction history, card expiration, dunning, and gateway patterns |
| **Segmentation** | RFM (Recency, Frequency, Monetary) analysis creating 9 customer segments with geographic distribution |
| **Expansion** | Upsell/cross-sell opportunity detection based on revenue growth, tenure, MRR, and plan hierarchy |
| **Refund** | Credit note pattern analysis by reason code with high-refund customer flagging |
| **Anomaly** | Z-score anomaly detection for billing amounts, overdue invoices, fraud flags, and billing spikes |
| **Coupon** | Coupon effectiveness analysis with ROI calculation, redemption tracking, and retention correlation |
| **Cohort** | Monthly cohort retention analysis tracking MRR and churn rates by signup month |
| **Events** | Event sequence pattern mining to identify common customer journeys and pre-churn patterns |
| **Deduplication** | Duplicate record detection using email matching and fuzzy company name similarity |

All analyzers are rule-based and require no external ML libraries.

## Requirements

- Python 3.11+
- A Chargebee account with API access

## Installation

```bash
git clone https://github.com/<your-org>/chargebee-subscription-analytics.git
cd chargebee-subscription-analytics
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
CHARGEBEE_API_KEY=your_api_key_here
CHARGEBEE_SITE=your-site-name

# Optional
CHARGEBEE_MAX_CONCURRENCY=20
CHARGEBEE_PAGE_SIZE=100
CHARGEBEE_MAX_RETRIES=5
CHARGEBEE_RATE_LIMIT_RPM=150
CHARGEBEE_LOG_LEVEL=INFO
CHARGEBEE_OUTPUT_DIR=output
```

## Usage

### Fetch Data

```bash
python -m chargebee_mapper
```

This connects to the Chargebee API, fetches all entity types, and writes the results to:
- `output/json/` -- One JSON file per entity type
- `output/chargebee_data.db` -- SQLite database with all records
- `output/summary.json` -- Fetch run metadata

### Run Analytics

```bash
# Run all 12 analyses
python -m chargebee_mapper analyze all

# Run a specific analysis
python -m chargebee_mapper analyze churn
python -m chargebee_mapper analyze revenue
python -m chargebee_mapper analyze segmentation

# Use a custom data directory
python -m chargebee_mapper analyze all --data-dir /path/to/data
```

Analysis results are saved to `output/analysis/` as JSON files.

### Available Analysis Types

```
all             Run all analyses
churn           Churn risk prediction
revenue         Revenue forecasting (MRR/ARR)
clv             Customer lifetime value prediction
payment         Payment failure risk scoring
segmentation    Customer segmentation (RFM)
expansion       Expansion/upsell prediction
refund          Refund and credit note analysis
anomaly         Billing anomaly detection
coupon          Coupon effectiveness analysis
cohort          Cohort retention analysis
events          Event sequence pattern mining
deduplication   Duplicate record detection
```

## Project Structure

```
chargebee_mapper/
  __main__.py              CLI entry point and analysis orchestration
  config.py                Configuration loading from environment variables
  client.py                Async HTTP client with rate limiting and retries
  fetcher.py               Fetch orchestrator for all entity types
  entities.py              Entity registry (36 Chargebee API resources)
  storage.py               JSON and SQLite output writers
  progress.py              Rich console progress display
  sync_state.py            Incremental sync state tracking
  data_cache.py            Shared data cache for analysis operations
  deduplication.py         Duplicate record detection
  churn_analyzer.py        Churn risk scoring
  revenue_forecaster.py    MRR/ARR forecasting
  clv_predictor.py         Customer lifetime value prediction
  payment_failure_predictor.py  Payment failure risk scoring
  customer_segmentation.py RFM customer segmentation
  expansion_predictor.py   Upsell/cross-sell detection
  refund_analyzer.py       Credit note analysis
  anomaly_detector.py      Billing anomaly detection
  coupon_analyzer.py       Coupon effectiveness analysis
  cohort_analyzer.py       Cohort retention analysis
  event_sequence_analyzer.py  Event pattern mining
  visualizations.py        Optional matplotlib chart generation

tests/
  conftest.py              Shared test fixtures
  test_client.py           Rate limiter and HTTP client tests
  test_config.py           Configuration loading tests
  test_entities.py         Entity registry tests
  test_storage.py          SQLite and JSON writer tests
  test_sync_state.py       Incremental sync state tests
  test_data_cache.py       Data cache tests
  test_deduplication.py    Deduplication tests
```

## Testing

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

## Chargebee Entities Fetched

The tool fetches 36 entity types organized as:

**Core Business:** Customer, Subscription, Invoice, Credit Note, Transaction, Order, Quote, Gift

**Product Catalog:** Item Family, Item, Item Price, Differential Price, Price Variant, Plan, Addon, Coupon, Coupon Set, Coupon Code

**Payments & Billing:** Payment Source, Virtual Bank Account, Unbilled Charge, Promotional Credit, Usage

**System:** Event, Comment, Hosted Page, Feature, Entitlement, Currency, Configuration, Site Migration Detail, Ramp, Webhook Endpoint

**Omnichannel:** Omnichannel Subscription, Omnichannel One-Time Order

**Dependent:** Attached Item (requires parent Item ID)

## License

MIT
