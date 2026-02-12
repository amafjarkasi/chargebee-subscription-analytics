"""Customer Lifetime Value (CLV) prediction.

Calculates and predicts customer lifetime value using historical data.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import math

logger = logging.getLogger("chargebee_mapper.clv_predictor")


@dataclass
class CustomerCLV:
    """CLV data for a single customer."""
    customer_id: str
    email: str | None
    company: str | None
    historical_revenue: float
    predicted_clv: float
    tenure_months: int
    avg_monthly_revenue: float
    purchase_frequency: float  # Purchases per month
    monetary_value: float  # Avg purchase value
    recency_days: int
    clv_segment: str  # "high", "medium", "low"
    expected_remaining_lifetime_months: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "email": self.email,
            "company": self.company,
            "historical_revenue": round(self.historical_revenue, 2),
            "predicted_clv": round(self.predicted_clv, 2),
            "tenure_months": self.tenure_months,
            "avg_monthly_revenue": round(self.avg_monthly_revenue, 2),
            "purchase_frequency": round(self.purchase_frequency, 3),
            "monetary_value": round(self.monetary_value, 2),
            "recency_days": self.recency_days,
            "clv_segment": self.clv_segment,
            "expected_remaining_lifetime_months": self.expected_remaining_lifetime_months,
        }


@dataclass
class CLVAnalysisResult:
    """Complete CLV analysis results."""
    analysis_timestamp: datetime
    total_customers: int
    total_historical_revenue: float
    total_predicted_clv: float
    avg_clv: float
    median_clv: float
    segment_distribution: dict[str, int]
    segment_revenue: dict[str, float]
    customer_clvs: list[CustomerCLV]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_customers": self.total_customers,
            "total_historical_revenue": round(self.total_historical_revenue, 2),
            "total_predicted_clv": round(self.total_predicted_clv, 2),
            "avg_clv": round(self.avg_clv, 2),
            "median_clv": round(self.median_clv, 2),
            "segment_distribution": self.segment_distribution,
            "segment_revenue": {k: round(v, 2) for k, v in self.segment_revenue.items()},
            "customer_clvs": [c.to_dict() for c in self.customer_clvs],
        }


def _parse_timestamp(ts: int | None) -> datetime | None:
    """Convert Unix timestamp to datetime."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, OSError):
        return None


def _days_since(dt: datetime | None) -> int:
    """Calculate days since a datetime."""
    if dt is None:
        return 0
    now = datetime.now(timezone.utc)
    return max(0, (now - dt).days)


def _months_since(dt: datetime | None) -> int:
    """Calculate months since a datetime."""
    if dt is None:
        return 0
    now = datetime.now(timezone.utc)
    return max(1, (now.year - dt.year) * 12 + (now.month - dt.month))


class CLVPredictor:
    """Predicts customer lifetime value using RFM-based analysis."""
    
    # Average customer lifetime assumptions (can be calibrated)
    DEFAULT_LIFETIME_MONTHS = 36
    DISCOUNT_RATE_MONTHLY = 0.01  # 1% monthly discount rate for NPV
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.json_dir = data_dir / "json"
        
        self._customers: list[dict] = []
        self._invoices: list[dict] = []
        self._subscriptions: list[dict] = []
        
    def load_data(self) -> bool:
        """Load required JSON data files."""
        logger.info("Loading data from %s", self.json_dir)
        
        self._customers = self._load_json("customers.json")
        self._invoices = self._load_json("invoices.json")
        self._subscriptions = self._load_json("subscriptions.json")
        
        if not self._customers:
            logger.error("No customer data found")
            return False
            
        logger.info(
            "Loaded: %d customers, %d invoices, %d subscriptions",
            len(self._customers), len(self._invoices), len(self._subscriptions)
        )
        return True
    
    def _load_json(self, filename: str) -> list[dict]:
        """Load a JSON file."""
        fpath = self.json_dir / filename
        if not fpath.exists():
            return []
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", fpath, e)
            return []
    
    def analyze(self) -> CLVAnalysisResult:
        """Run CLV analysis on all customers."""
        logger.info("Starting CLV analysis for %d customers", len(self._customers))
        
        # Index invoices by customer
        invoices_by_customer: dict[str, list[dict]] = defaultdict(list)
        for inv in self._invoices:
            cid = inv.get("customer_id")
            if cid and inv.get("status") in ("paid", "posted"):
                invoices_by_customer[cid].append(inv)
        
        # Index subscriptions by customer
        subs_by_customer: dict[str, list[dict]] = defaultdict(list)
        for sub in self._subscriptions:
            cid = sub.get("customer_id")
            if cid:
                subs_by_customer[cid].append(sub)
        
        customer_clvs: list[CustomerCLV] = []
        
        for customer in self._customers:
            if customer.get("deleted"):
                continue
            
            cid = customer.get("id")
            invoices = invoices_by_customer.get(cid, [])
            subscriptions = subs_by_customer.get(cid, [])
            
            clv = self._calculate_customer_clv(customer, invoices, subscriptions)
            customer_clvs.append(clv)
        
        # Sort by predicted CLV descending
        customer_clvs.sort(key=lambda x: x.predicted_clv, reverse=True)
        
        # Calculate aggregates
        total_historical = sum(c.historical_revenue for c in customer_clvs)
        total_predicted = sum(c.predicted_clv for c in customer_clvs)
        
        clv_values = [c.predicted_clv for c in customer_clvs if c.predicted_clv > 0]
        avg_clv = sum(clv_values) / len(clv_values) if clv_values else 0
        median_clv = sorted(clv_values)[len(clv_values) // 2] if clv_values else 0
        
        segment_dist = {"high": 0, "medium": 0, "low": 0}
        segment_rev = {"high": 0.0, "medium": 0.0, "low": 0.0}
        
        for c in customer_clvs:
            segment_dist[c.clv_segment] += 1
            segment_rev[c.clv_segment] += c.predicted_clv
        
        result = CLVAnalysisResult(
            analysis_timestamp=datetime.now(timezone.utc),
            total_customers=len(customer_clvs),
            total_historical_revenue=total_historical,
            total_predicted_clv=total_predicted,
            avg_clv=avg_clv,
            median_clv=median_clv,
            segment_distribution=segment_dist,
            segment_revenue=segment_rev,
            customer_clvs=customer_clvs,
        )
        
        logger.info(
            "CLV analysis complete: %d customers, avg CLV=$%.2f, total predicted=$%.2f",
            len(customer_clvs), avg_clv, total_predicted
        )
        
        return result
    
    def _calculate_customer_clv(
        self,
        customer: dict,
        invoices: list[dict],
        subscriptions: list[dict],
    ) -> CustomerCLV:
        """Calculate CLV for a single customer."""
        customer_id = customer.get("id", "unknown")
        
        # Calculate tenure
        created_at = _parse_timestamp(customer.get("created_at"))
        tenure_months = _months_since(created_at)
        
        # Calculate historical revenue
        historical_revenue = sum(
            (inv.get("total", 0) or 0) / 100
            for inv in invoices
        )
        
        # Calculate recency (days since last invoice)
        if invoices:
            last_invoice_date = max(
                _parse_timestamp(inv.get("date")) or datetime.min.replace(tzinfo=timezone.utc)
                for inv in invoices
            )
            recency_days = _days_since(last_invoice_date)
        else:
            recency_days = _days_since(created_at)
        
        # Calculate frequency (purchases per month)
        purchase_frequency = len(invoices) / max(tenure_months, 1)
        
        # Calculate monetary value (avg purchase)
        monetary_value = historical_revenue / max(len(invoices), 1)
        
        # Calculate avg monthly revenue
        avg_monthly_revenue = historical_revenue / max(tenure_months, 1)
        
        # Estimate remaining lifetime based on churn signals
        active_subs = [s for s in subscriptions if s.get("status") == "active"]
        has_active_sub = len(active_subs) > 0
        
        # Simple expected lifetime calculation
        if has_active_sub:
            # Active customers expected to stay longer
            if recency_days < 30:
                expected_remaining = 24
            elif recency_days < 90:
                expected_remaining = 18
            else:
                expected_remaining = 12
        else:
            expected_remaining = 3  # Churned/inactive customers
        
        # Calculate predicted CLV using simple model:
        # CLV = Historical + (Monthly Revenue * Expected Months * Discount Factor)
        future_value = 0.0
        for month in range(1, expected_remaining + 1):
            discount = 1 / ((1 + self.DISCOUNT_RATE_MONTHLY) ** month)
            future_value += avg_monthly_revenue * discount
        
        predicted_clv = historical_revenue + future_value
        
        # Determine segment based on percentile (will be normalized later)
        if predicted_clv >= 5000:
            segment = "high"
        elif predicted_clv >= 1000:
            segment = "medium"
        else:
            segment = "low"
        
        return CustomerCLV(
            customer_id=customer_id,
            email=customer.get("email"),
            company=customer.get("company"),
            historical_revenue=historical_revenue,
            predicted_clv=predicted_clv,
            tenure_months=tenure_months,
            avg_monthly_revenue=avg_monthly_revenue,
            purchase_frequency=purchase_frequency,
            monetary_value=monetary_value,
            recency_days=recency_days,
            clv_segment=segment,
            expected_remaining_lifetime_months=expected_remaining,
        )
    
    def save_results(self, result: CLVAnalysisResult, output_dir: Path) -> Path:
        """Save CLV results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "clv_analysis.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info("CLV analysis saved to %s", output_path)
        return output_path
