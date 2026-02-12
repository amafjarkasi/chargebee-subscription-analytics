"""Refund and credit note analysis.

Analyzes refund patterns, reasons, and identifies potential issues.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("chargebee_mapper.refund_analyzer")


@dataclass
class RefundPattern:
    """Refund pattern analysis for a reason code."""
    reason_code: str
    count: int
    total_amount: float
    avg_amount: float
    pct_of_total: float
    affected_customers: int
    trend: str  # "increasing", "stable", "decreasing"


@dataclass
class CustomerRefundProfile:
    """Refund profile for a single customer."""
    customer_id: str
    email: str | None
    company: str | None
    total_refunds: int
    total_refund_amount: float
    refund_rate: float  # Refunds / Total invoices
    reason_codes: list[str]
    risk_flag: str  # "normal", "elevated", "high"
    last_refund_date: datetime | None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "email": self.email,
            "company": self.company,
            "total_refunds": self.total_refunds,
            "total_refund_amount": round(self.total_refund_amount, 2),
            "refund_rate": round(self.refund_rate, 4),
            "reason_codes": self.reason_codes,
            "risk_flag": self.risk_flag,
            "last_refund_date": self.last_refund_date.isoformat() if self.last_refund_date else None,
        }


@dataclass
class RefundAnalysisResult:
    """Complete refund analysis results."""
    analysis_timestamp: datetime
    total_credit_notes: int
    total_refund_amount: float
    total_invoice_amount: float
    overall_refund_rate: float
    refund_by_reason: list[RefundPattern]
    refund_by_month: dict[str, dict]
    high_refund_customers: list[CustomerRefundProfile]
    all_customer_profiles: list[CustomerRefundProfile]
    insights: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_credit_notes": self.total_credit_notes,
            "total_refund_amount": round(self.total_refund_amount, 2),
            "total_invoice_amount": round(self.total_invoice_amount, 2),
            "overall_refund_rate": round(self.overall_refund_rate, 4),
            "refund_by_reason": [
                {
                    "reason_code": r.reason_code,
                    "count": r.count,
                    "total_amount": round(r.total_amount, 2),
                    "avg_amount": round(r.avg_amount, 2),
                    "pct_of_total": round(r.pct_of_total, 4),
                    "affected_customers": r.affected_customers,
                    "trend": r.trend,
                }
                for r in self.refund_by_reason
            ],
            "refund_by_month": self.refund_by_month,
            "high_refund_customers": [c.to_dict() for c in self.high_refund_customers],
            "all_customer_profiles": [c.to_dict() for c in self.all_customer_profiles],
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


class RefundAnalyzer:
    """Analyzes refund and credit note patterns."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.json_dir = data_dir / "json"
        
        self._credit_notes: list[dict] = []
        self._invoices: list[dict] = []
        self._customers: list[dict] = []
        
    def load_data(self) -> bool:
        """Load required JSON data files."""
        logger.info("Loading data from %s", self.json_dir)
        
        self._credit_notes = self._load_json("credit_notes.json")
        self._invoices = self._load_json("invoices.json")
        self._customers = self._load_json("customers.json")
        
        logger.info(
            "Loaded: %d credit notes, %d invoices, %d customers",
            len(self._credit_notes), len(self._invoices), len(self._customers)
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
    
    def analyze(self) -> RefundAnalysisResult:
        """Run refund analysis."""
        logger.info("Starting refund analysis")
        
        # Calculate total invoice amount
        total_invoice_amount = sum(
            (inv.get("total", 0) or 0) / 100
            for inv in self._invoices
            if inv.get("status") in ("paid", "posted")
        )
        
        # Analyze credit notes by reason
        reason_stats: dict[str, dict] = defaultdict(lambda: {
            "count": 0, "amount": 0.0, "customers": set()
        })
        
        monthly_stats: dict[str, dict] = defaultdict(lambda: {
            "count": 0, "amount": 0.0
        })
        
        customer_refunds: dict[str, dict] = defaultdict(lambda: {
            "count": 0, "amount": 0.0, "reasons": [], "last_date": None
        })
        
        total_refund_amount = 0.0
        
        for cn in self._credit_notes:
            if cn.get("status") == "voided":
                continue
            
            amount = (cn.get("total", 0) or 0) / 100
            reason = cn.get("reason_code", "unknown")
            customer_id = cn.get("customer_id")
            date = _parse_timestamp(cn.get("date"))
            
            total_refund_amount += amount
            
            # By reason
            reason_stats[reason]["count"] += 1
            reason_stats[reason]["amount"] += amount
            if customer_id:
                reason_stats[reason]["customers"].add(customer_id)
            
            # By month
            if date:
                month_key = f"{date.year}-{date.month:02d}"
                monthly_stats[month_key]["count"] += 1
                monthly_stats[month_key]["amount"] += amount
            
            # By customer
            if customer_id:
                customer_refunds[customer_id]["count"] += 1
                customer_refunds[customer_id]["amount"] += amount
                if reason not in customer_refunds[customer_id]["reasons"]:
                    customer_refunds[customer_id]["reasons"].append(reason)
                if date and (customer_refunds[customer_id]["last_date"] is None or 
                           date > customer_refunds[customer_id]["last_date"]):
                    customer_refunds[customer_id]["last_date"] = date
        
        # Build reason patterns
        refund_patterns = []
        for reason, stats in sorted(reason_stats.items(), key=lambda x: x[1]["amount"], reverse=True):
            pct = stats["amount"] / total_refund_amount if total_refund_amount > 0 else 0
            refund_patterns.append(RefundPattern(
                reason_code=reason,
                count=stats["count"],
                total_amount=stats["amount"],
                avg_amount=stats["amount"] / max(stats["count"], 1),
                pct_of_total=pct,
                affected_customers=len(stats["customers"]),
                trend="stable",  # Would need historical data for trend
            ))
        
        # Build customer profiles
        customer_map = {c.get("id"): c for c in self._customers}
        invoices_by_customer: dict[str, int] = defaultdict(int)
        for inv in self._invoices:
            cid = inv.get("customer_id")
            if cid:
                invoices_by_customer[cid] += 1
        
        customer_profiles = []
        for cid, stats in customer_refunds.items():
            customer = customer_map.get(cid, {})
            invoice_count = invoices_by_customer.get(cid, 1)
            refund_rate = stats["count"] / max(invoice_count, 1)
            
            # Determine risk flag
            if refund_rate > 0.3 or stats["count"] >= 5:
                risk_flag = "high"
            elif refund_rate > 0.1 or stats["count"] >= 3:
                risk_flag = "elevated"
            else:
                risk_flag = "normal"
            
            customer_profiles.append(CustomerRefundProfile(
                customer_id=cid,
                email=customer.get("email"),
                company=customer.get("company"),
                total_refunds=stats["count"],
                total_refund_amount=stats["amount"],
                refund_rate=refund_rate,
                reason_codes=stats["reasons"],
                risk_flag=risk_flag,
                last_refund_date=stats["last_date"],
            ))
        
        # Sort by refund amount
        customer_profiles.sort(key=lambda x: x.total_refund_amount, reverse=True)
        high_refund_customers = [c for c in customer_profiles if c.risk_flag in ("high", "elevated")]
        
        # Generate insights
        insights = self._generate_insights(
            refund_patterns, total_refund_amount, total_invoice_amount, high_refund_customers
        )
        
        overall_refund_rate = total_refund_amount / total_invoice_amount if total_invoice_amount > 0 else 0
        
        result = RefundAnalysisResult(
            analysis_timestamp=datetime.now(timezone.utc),
            total_credit_notes=len([cn for cn in self._credit_notes if cn.get("status") != "voided"]),
            total_refund_amount=total_refund_amount,
            total_invoice_amount=total_invoice_amount,
            overall_refund_rate=overall_refund_rate,
            refund_by_reason=refund_patterns,
            refund_by_month={k: {"count": v["count"], "amount": round(v["amount"], 2)} 
                           for k, v in sorted(monthly_stats.items())},
            high_refund_customers=high_refund_customers,
            all_customer_profiles=customer_profiles,
            insights=insights,
        )
        
        logger.info(
            "Refund analysis complete: %d credit notes, $%.2f total refunds, %.1f%% refund rate",
            result.total_credit_notes, total_refund_amount, overall_refund_rate * 100
        )
        
        return result
    
    def _generate_insights(
        self,
        patterns: list[RefundPattern],
        total_refunds: float,
        total_invoices: float,
        high_risk_customers: list[CustomerRefundProfile],
    ) -> list[str]:
        """Generate actionable insights from refund data."""
        insights = []
        
        refund_rate = total_refunds / total_invoices if total_invoices > 0 else 0
        
        if refund_rate > 0.1:
            insights.append(f"High overall refund rate ({refund_rate:.1%}) - investigate root causes")
        elif refund_rate > 0.05:
            insights.append(f"Moderate refund rate ({refund_rate:.1%}) - monitor closely")
        else:
            insights.append(f"Healthy refund rate ({refund_rate:.1%})")
        
        # Top reason analysis
        if patterns:
            top_reason = patterns[0]
            insights.append(
                f"Top refund reason: '{top_reason.reason_code}' "
                f"({top_reason.count} refunds, ${top_reason.total_amount:.0f}, "
                f"{top_reason.pct_of_total:.0%} of total)"
            )
        
        # High-risk customer analysis
        if high_risk_customers:
            total_high_risk_amount = sum(c.total_refund_amount for c in high_risk_customers)
            insights.append(
                f"{len(high_risk_customers)} customers flagged as elevated/high refund risk "
                f"(${total_high_risk_amount:.0f} in refunds)"
            )
        
        return insights
    
    def save_results(self, result: RefundAnalysisResult, output_dir: Path) -> Path:
        """Save analysis results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "refund_analysis.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info("Refund analysis saved to %s", output_path)
        return output_path
