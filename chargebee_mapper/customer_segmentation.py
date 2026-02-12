"""Customer segmentation using RFM analysis and clustering.

Segments customers based on Recency, Frequency, and Monetary value.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import math

logger = logging.getLogger("chargebee_mapper.customer_segmentation")


@dataclass
class CustomerSegment:
    """Segment data for a single customer."""
    customer_id: str
    email: str | None
    company: str | None
    country: str | None
    # RFM scores (1-5, higher is better)
    recency_score: int
    frequency_score: int
    monetary_score: int
    rfm_score: int  # Combined R+F+M
    rfm_segment: str  # "Champions", "Loyal", "At Risk", etc.
    # Raw values
    days_since_last_purchase: int
    total_purchases: int
    total_revenue: float
    avg_order_value: float
    # Additional attributes
    subscription_status: str
    plan_type: str | None
    tenure_months: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "email": self.email,
            "company": self.company,
            "country": self.country,
            "recency_score": self.recency_score,
            "frequency_score": self.frequency_score,
            "monetary_score": self.monetary_score,
            "rfm_score": self.rfm_score,
            "rfm_segment": self.rfm_segment,
            "days_since_last_purchase": self.days_since_last_purchase,
            "total_purchases": self.total_purchases,
            "total_revenue": round(self.total_revenue, 2),
            "avg_order_value": round(self.avg_order_value, 2),
            "subscription_status": self.subscription_status,
            "plan_type": self.plan_type,
            "tenure_months": self.tenure_months,
        }


@dataclass
class SegmentationResult:
    """Complete customer segmentation results."""
    analysis_timestamp: datetime
    total_customers: int
    segment_distribution: dict[str, int]
    segment_revenue: dict[str, float]
    segment_descriptions: dict[str, str]
    geographic_distribution: dict[str, int]
    plan_distribution: dict[str, int]
    customer_segments: list[CustomerSegment]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_customers": self.total_customers,
            "segment_distribution": self.segment_distribution,
            "segment_revenue": {k: round(v, 2) for k, v in self.segment_revenue.items()},
            "segment_descriptions": self.segment_descriptions,
            "geographic_distribution": self.geographic_distribution,
            "plan_distribution": self.plan_distribution,
            "customer_segments": [c.to_dict() for c in self.customer_segments],
        }


# RFM segment definitions
RFM_SEGMENTS = {
    "Champions": {"min_rfm": 12, "description": "Best customers - high value, recent, frequent"},
    "Loyal Customers": {"min_rfm": 10, "description": "Consistent buyers with good value"},
    "Potential Loyalists": {"min_rfm": 8, "description": "Recent customers with growth potential"},
    "Recent Customers": {"min_rfm": 6, "description": "New customers, need nurturing"},
    "Promising": {"min_rfm": 5, "description": "Moderate engagement, room to grow"},
    "Need Attention": {"min_rfm": 4, "description": "Above average but slipping"},
    "About to Sleep": {"min_rfm": 3, "description": "Below average, at risk of churning"},
    "At Risk": {"min_rfm": 2, "description": "High value but haven't purchased recently"},
    "Hibernating": {"min_rfm": 0, "description": "Low engagement across all metrics"},
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
        return 9999
    now = datetime.now(timezone.utc)
    return max(0, (now - dt).days)


def _months_since(dt: datetime | None) -> int:
    """Calculate months since a datetime."""
    if dt is None:
        return 0
    now = datetime.now(timezone.utc)
    return max(1, (now.year - dt.year) * 12 + (now.month - dt.month))


def _quintile_score(value: float, quintiles: list[float], higher_is_better: bool = True) -> int:
    """Assign a 1-5 score based on quintile position."""
    for i, q in enumerate(quintiles):
        if value <= q:
            return (i + 1) if higher_is_better else (5 - i)
    return 5 if higher_is_better else 1


class CustomerSegmenter:
    """Segments customers using RFM analysis."""
    
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
    
    def analyze(self) -> SegmentationResult:
        """Run customer segmentation analysis."""
        logger.info("Starting customer segmentation for %d customers", len(self._customers))
        
        # Index data
        invoices_by_customer: dict[str, list[dict]] = defaultdict(list)
        for inv in self._invoices:
            cid = inv.get("customer_id")
            if cid and inv.get("status") in ("paid", "posted"):
                invoices_by_customer[cid].append(inv)
        
        subs_by_customer: dict[str, list[dict]] = defaultdict(list)
        for sub in self._subscriptions:
            cid = sub.get("customer_id")
            if cid:
                subs_by_customer[cid].append(sub)
        
        # Calculate raw RFM values for all customers
        raw_data = []
        for customer in self._customers:
            if customer.get("deleted"):
                continue
            
            cid = customer.get("id")
            invoices = invoices_by_customer.get(cid, [])
            subs = subs_by_customer.get(cid, [])
            
            # Recency: days since last invoice
            if invoices:
                last_date = max(_parse_timestamp(i.get("date")) for i in invoices)
                recency = _days_since(last_date)
            else:
                recency = _days_since(_parse_timestamp(customer.get("created_at")))
            
            # Frequency: number of invoices
            frequency = len(invoices)
            
            # Monetary: total revenue
            monetary = sum((inv.get("total", 0) or 0) / 100 for inv in invoices)
            
            raw_data.append({
                "customer": customer,
                "invoices": invoices,
                "subscriptions": subs,
                "recency": recency,
                "frequency": frequency,
                "monetary": monetary,
            })
        
        # Calculate quintiles for scoring
        recency_values = sorted([d["recency"] for d in raw_data])
        frequency_values = sorted([d["frequency"] for d in raw_data])
        monetary_values = sorted([d["monetary"] for d in raw_data])
        
        def get_quintiles(values: list) -> list[float]:
            if not values:
                return [0, 0, 0, 0, 0]
            n = len(values)
            # Use min to avoid index out of range for the last quintile
            return [values[min(int(n * p), n - 1)] for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
        
        r_quintiles = get_quintiles(recency_values)
        f_quintiles = get_quintiles(frequency_values)
        m_quintiles = get_quintiles(monetary_values)
        
        # Score each customer
        customer_segments: list[CustomerSegment] = []
        
        for data in raw_data:
            customer = data["customer"]
            invoices = data["invoices"]
            subs = data["subscriptions"]
            
            # Calculate RFM scores
            r_score = _quintile_score(data["recency"], r_quintiles, higher_is_better=False)
            f_score = _quintile_score(data["frequency"], f_quintiles, higher_is_better=True)
            m_score = _quintile_score(data["monetary"], m_quintiles, higher_is_better=True)
            rfm_score = r_score + f_score + m_score
            
            # Determine segment
            segment_name = "Hibernating"
            for name, criteria in RFM_SEGMENTS.items():
                if rfm_score >= criteria["min_rfm"]:
                    segment_name = name
                    break
            
            # Get subscription info
            active_subs = [s for s in subs if s.get("status") == "active"]
            sub_status = "active" if active_subs else ("churned" if subs else "none")
            
            plan_type = None
            if active_subs:
                items = active_subs[0].get("subscription_items", [])
                if items:
                    plan_type = items[0].get("item_price_id")
            
            # Get country
            billing = customer.get("billing_address", {})
            country = billing.get("country") if billing else None
            
            # Calculate tenure
            tenure = _months_since(_parse_timestamp(customer.get("created_at")))
            
            # Average order value
            aov = data["monetary"] / max(data["frequency"], 1)
            
            customer_segments.append(CustomerSegment(
                customer_id=customer.get("id", "unknown"),
                email=customer.get("email"),
                company=customer.get("company"),
                country=country,
                recency_score=r_score,
                frequency_score=f_score,
                monetary_score=m_score,
                rfm_score=rfm_score,
                rfm_segment=segment_name,
                days_since_last_purchase=data["recency"],
                total_purchases=data["frequency"],
                total_revenue=data["monetary"],
                avg_order_value=aov,
                subscription_status=sub_status,
                plan_type=plan_type,
                tenure_months=tenure,
            ))
        
        # Sort by RFM score
        customer_segments.sort(key=lambda x: x.rfm_score, reverse=True)
        
        # Calculate distributions
        segment_dist: dict[str, int] = defaultdict(int)
        segment_rev: dict[str, float] = defaultdict(float)
        geo_dist: dict[str, int] = defaultdict(int)
        plan_dist: dict[str, int] = defaultdict(int)
        
        for cs in customer_segments:
            segment_dist[cs.rfm_segment] += 1
            segment_rev[cs.rfm_segment] += cs.total_revenue
            if cs.country:
                geo_dist[cs.country] += 1
            if cs.plan_type:
                plan_dist[cs.plan_type] += 1
        
        segment_descriptions = {name: info["description"] for name, info in RFM_SEGMENTS.items()}
        
        result = SegmentationResult(
            analysis_timestamp=datetime.now(timezone.utc),
            total_customers=len(customer_segments),
            segment_distribution=dict(segment_dist),
            segment_revenue=dict(segment_rev),
            segment_descriptions=segment_descriptions,
            geographic_distribution=dict(geo_dist),
            plan_distribution=dict(plan_dist),
            customer_segments=customer_segments,
        )
        
        logger.info("Segmentation complete: %d customers across %d segments",
                   len(customer_segments), len(segment_dist))
        
        return result
    
    def save_results(self, result: SegmentationResult, output_dir: Path) -> Path:
        """Save segmentation results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "customer_segmentation.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info("Customer segmentation saved to %s", output_path)
        return output_path
