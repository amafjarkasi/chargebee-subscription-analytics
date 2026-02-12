"""Cohort analysis for customer retention and revenue.

Analyzes customer behavior by signup cohort.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("chargebee_mapper.cohort_analyzer")


@dataclass
class CohortMetrics:
    """Metrics for a single cohort."""
    cohort_period: str  # e.g., "2024-01"
    cohort_size: int
    retention_by_month: dict[int, float]  # Month offset -> retention rate
    revenue_by_month: dict[int, float]  # Month offset -> total revenue
    avg_revenue_per_customer: float
    churn_rate: float
    still_active: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_period": self.cohort_period,
            "cohort_size": self.cohort_size,
            "retention_by_month": {str(k): round(v, 4) for k, v in self.retention_by_month.items()},
            "revenue_by_month": {str(k): round(v, 2) for k, v in self.revenue_by_month.items()},
            "avg_revenue_per_customer": round(self.avg_revenue_per_customer, 2),
            "churn_rate": round(self.churn_rate, 4),
            "still_active": self.still_active,
        }


@dataclass
class CohortAnalysisResult:
    """Complete cohort analysis results."""
    analysis_timestamp: datetime
    total_cohorts: int
    total_customers: int
    cohorts: list[CohortMetrics]
    retention_matrix: dict[str, dict[int, float]]  # cohort -> month -> retention
    revenue_matrix: dict[str, dict[int, float]]  # cohort -> month -> revenue
    avg_retention_by_month: dict[int, float]
    insights: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_cohorts": self.total_cohorts,
            "total_customers": self.total_customers,
            "cohorts": [c.to_dict() for c in self.cohorts],
            "retention_matrix": {
                k: {str(m): round(r, 4) for m, r in v.items()}
                for k, v in self.retention_matrix.items()
            },
            "revenue_matrix": {
                k: {str(m): round(r, 2) for m, r in v.items()}
                for k, v in self.revenue_matrix.items()
            },
            "avg_retention_by_month": {str(k): round(v, 4) for k, v in self.avg_retention_by_month.items()},
            "insights": self.insights,
        }


def _parse_timestamp(ts: int | None) -> datetime | None:
    """Convert Unix timestamp to datetime."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, OSError):
        return None


def _get_month_key(dt: datetime) -> str:
    """Get YYYY-MM key from datetime."""
    return f"{dt.year}-{dt.month:02d}"


def _month_diff(dt1: datetime, dt2: datetime) -> int:
    """Calculate month difference between two dates."""
    return (dt2.year - dt1.year) * 12 + (dt2.month - dt1.month)


class CohortAnalyzer:
    """Analyzes customer cohorts for retention and revenue patterns."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.json_dir = data_dir / "json"
        
        self._customers: list[dict] = []
        self._subscriptions: list[dict] = []
        self._invoices: list[dict] = []
        
    def load_data(self) -> bool:
        """Load required JSON data files."""
        logger.info("Loading data from %s", self.json_dir)
        
        self._customers = self._load_json("customers.json")
        self._subscriptions = self._load_json("subscriptions.json")
        self._invoices = self._load_json("invoices.json")
        
        if not self._customers:
            logger.error("No customer data found")
            return False
            
        logger.info(
            "Loaded: %d customers, %d subscriptions, %d invoices",
            len(self._customers), len(self._subscriptions), len(self._invoices)
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
    
    def analyze(self) -> CohortAnalysisResult:
        """Run cohort analysis."""
        logger.info("Starting cohort analysis")
        
        now = datetime.now(timezone.utc)
        
        # Group customers by signup month
        cohort_customers: dict[str, list[dict]] = defaultdict(list)
        customer_cohort: dict[str, str] = {}  # customer_id -> cohort
        
        for customer in self._customers:
            if customer.get("deleted"):
                continue
            
            created = _parse_timestamp(customer.get("created_at"))
            if not created:
                continue
            
            cohort_key = _get_month_key(created)
            cohort_customers[cohort_key].append(customer)
            customer_cohort[customer.get("id")] = cohort_key
        
        # Index subscriptions and invoices by customer
        subs_by_customer: dict[str, list[dict]] = defaultdict(list)
        for sub in self._subscriptions:
            cid = sub.get("customer_id")
            if cid:
                subs_by_customer[cid].append(sub)
        
        invoices_by_customer: dict[str, list[dict]] = defaultdict(list)
        for inv in self._invoices:
            cid = inv.get("customer_id")
            if cid and inv.get("status") in ("paid", "posted"):
                invoices_by_customer[cid].append(inv)
        
        # Analyze each cohort
        cohorts: list[CohortMetrics] = []
        retention_matrix: dict[str, dict[int, float]] = {}
        revenue_matrix: dict[str, dict[int, float]] = {}
        
        for cohort_key in sorted(cohort_customers.keys()):
            customers = cohort_customers[cohort_key]
            cohort_date = datetime.strptime(cohort_key, "%Y-%m").replace(tzinfo=timezone.utc)
            max_months = _month_diff(cohort_date, now)
            
            if max_months < 0:
                continue
            
            cohort_size = len(customers)
            retention_by_month: dict[int, float] = {}
            revenue_by_month: dict[int, float] = {}
            
            # Track activity by month
            for month_offset in range(min(max_months + 1, 13)):  # Up to 12 months
                target_month = datetime(
                    cohort_date.year + (cohort_date.month + month_offset - 1) // 12,
                    (cohort_date.month + month_offset - 1) % 12 + 1,
                    1,
                    tzinfo=timezone.utc
                )
                target_key = _get_month_key(target_month)
                
                active_count = 0
                month_revenue = 0.0
                
                for customer in customers:
                    cid = customer.get("id")
                    
                    # Check if customer was active (has invoice in this month)
                    customer_invoices = invoices_by_customer.get(cid, [])
                    month_invoices = [
                        inv for inv in customer_invoices
                        if _parse_timestamp(inv.get("date")) and 
                        _get_month_key(_parse_timestamp(inv.get("date"))) == target_key
                    ]
                    
                    if month_invoices:
                        active_count += 1
                        month_revenue += sum(
                            (inv.get("total", 0) or 0) / 100 for inv in month_invoices
                        )
                
                retention_by_month[month_offset] = active_count / cohort_size if cohort_size > 0 else 0
                revenue_by_month[month_offset] = month_revenue
            
            # Count still active customers
            still_active = 0
            for customer in customers:
                cid = customer.get("id")
                subs = subs_by_customer.get(cid, [])
                if any(s.get("status") == "active" for s in subs):
                    still_active += 1
            
            # Calculate total revenue and churn
            total_revenue = sum(
                (inv.get("total", 0) or 0) / 100
                for customer in customers
                for inv in invoices_by_customer.get(customer.get("id"), [])
            )
            
            churn_rate = 1 - (still_active / cohort_size) if cohort_size > 0 else 0
            avg_revenue = total_revenue / cohort_size if cohort_size > 0 else 0
            
            cohorts.append(CohortMetrics(
                cohort_period=cohort_key,
                cohort_size=cohort_size,
                retention_by_month=retention_by_month,
                revenue_by_month=revenue_by_month,
                avg_revenue_per_customer=avg_revenue,
                churn_rate=churn_rate,
                still_active=still_active,
            ))
            
            retention_matrix[cohort_key] = retention_by_month
            revenue_matrix[cohort_key] = revenue_by_month
        
        # Calculate average retention by month across cohorts
        avg_retention: dict[int, list[float]] = defaultdict(list)
        for cohort in cohorts:
            for month, rate in cohort.retention_by_month.items():
                avg_retention[month].append(rate)
        
        avg_retention_by_month = {
            month: sum(rates) / len(rates)
            for month, rates in avg_retention.items()
            if rates
        }
        
        # Generate insights
        insights = self._generate_insights(cohorts, avg_retention_by_month)
        
        result = CohortAnalysisResult(
            analysis_timestamp=now,
            total_cohorts=len(cohorts),
            total_customers=sum(c.cohort_size for c in cohorts),
            cohorts=cohorts,
            retention_matrix=retention_matrix,
            revenue_matrix=revenue_matrix,
            avg_retention_by_month=avg_retention_by_month,
            insights=insights,
        )
        
        logger.info(
            "Cohort analysis complete: %d cohorts, %d total customers",
            len(cohorts), result.total_customers
        )
        
        return result
    
    def _generate_insights(
        self, cohorts: list[CohortMetrics], avg_retention: dict[int, float]
    ) -> list[str]:
        """Generate insights from cohort analysis."""
        insights = []
        
        # Month 1 retention (immediate churn)
        if 1 in avg_retention:
            m1_retention = avg_retention[1]
            if m1_retention < 0.7:
                insights.append(f"Low Month 1 retention ({m1_retention:.0%}) - investigate onboarding")
            else:
                insights.append(f"Good Month 1 retention ({m1_retention:.0%})")
        
        # Month 3 and 6 retention
        if 3 in avg_retention:
            insights.append(f"Month 3 retention: {avg_retention[3]:.0%}")
        if 6 in avg_retention:
            insights.append(f"Month 6 retention: {avg_retention[6]:.0%}")
        
        # Best and worst cohorts
        if len(cohorts) >= 3:
            sorted_by_retention = sorted(
                cohorts, 
                key=lambda c: c.retention_by_month.get(3, 0), 
                reverse=True
            )
            best = sorted_by_retention[0]
            worst = sorted_by_retention[-1]
            
            if best.retention_by_month.get(3, 0) > 0:
                insights.append(f"Best performing cohort: {best.cohort_period}")
            if worst.retention_by_month.get(3, 0) < best.retention_by_month.get(3, 0):
                insights.append(f"Needs attention: {worst.cohort_period} cohort")
        
        return insights
    
    def save_results(self, result: CohortAnalysisResult, output_dir: Path) -> Path:
        """Save analysis results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "cohort_analysis.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info("Cohort analysis saved to %s", output_path)
        return output_path
