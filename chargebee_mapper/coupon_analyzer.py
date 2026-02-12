"""Coupon and discount effectiveness analysis.

Analyzes the ROI and effectiveness of coupons and discounts.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("chargebee_mapper.coupon_analyzer")


@dataclass
class CouponPerformance:
    """Performance metrics for a single coupon."""
    coupon_id: str
    coupon_name: str
    discount_type: str
    discount_value: float
    redemption_count: int
    total_discount_given: float
    revenue_from_users: float
    roi: float  # Revenue / Discount given
    avg_customer_value: float
    retention_rate: float | None
    status: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "coupon_id": self.coupon_id,
            "coupon_name": self.coupon_name,
            "discount_type": self.discount_type,
            "discount_value": self.discount_value,
            "redemption_count": self.redemption_count,
            "total_discount_given": round(self.total_discount_given, 2),
            "revenue_from_users": round(self.revenue_from_users, 2),
            "roi": round(self.roi, 2),
            "avg_customer_value": round(self.avg_customer_value, 2),
            "retention_rate": round(self.retention_rate, 4) if self.retention_rate else None,
            "status": self.status,
        }


@dataclass
class DiscountAnalysisSummary:
    """Summary of discount analysis."""
    total_discounts_given: float
    total_revenue_from_discounted: float
    overall_roi: float
    discount_rate: float  # Discount / Revenue
    top_performing_coupons: list[str]
    underperforming_coupons: list[str]


@dataclass
class CouponAnalysisResult:
    """Complete coupon analysis results."""
    analysis_timestamp: datetime
    total_coupons: int
    active_coupons: int
    total_redemptions: int
    total_discount_amount: float
    coupon_performances: list[CouponPerformance]
    discount_by_type: dict[str, dict]
    monthly_redemptions: dict[str, int]
    insights: list[str]
    summary: DiscountAnalysisSummary
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_coupons": self.total_coupons,
            "active_coupons": self.active_coupons,
            "total_redemptions": self.total_redemptions,
            "total_discount_amount": round(self.total_discount_amount, 2),
            "coupon_performances": [c.to_dict() for c in self.coupon_performances],
            "discount_by_type": self.discount_by_type,
            "monthly_redemptions": self.monthly_redemptions,
            "insights": self.insights,
            "summary": {
                "total_discounts_given": round(self.summary.total_discounts_given, 2),
                "total_revenue_from_discounted": round(self.summary.total_revenue_from_discounted, 2),
                "overall_roi": round(self.summary.overall_roi, 2),
                "discount_rate": round(self.summary.discount_rate, 4),
                "top_performing_coupons": self.summary.top_performing_coupons,
                "underperforming_coupons": self.summary.underperforming_coupons,
            },
        }


def _parse_timestamp(ts: int | None) -> datetime | None:
    """Convert Unix timestamp to datetime."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, OSError):
        return None


class CouponAnalyzer:
    """Analyzes coupon and discount effectiveness."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.json_dir = data_dir / "json"
        
        self._coupons: list[dict] = []
        self._invoices: list[dict] = []
        self._subscriptions: list[dict] = []
        self._customers: list[dict] = []
        
    def load_data(self) -> bool:
        """Load required JSON data files."""
        logger.info("Loading data from %s", self.json_dir)
        
        self._coupons = self._load_json("coupons.json")
        self._invoices = self._load_json("invoices.json")
        self._subscriptions = self._load_json("subscriptions.json")
        self._customers = self._load_json("customers.json")
        
        logger.info(
            "Loaded: %d coupons, %d invoices, %d subscriptions",
            len(self._coupons), len(self._invoices), len(self._subscriptions)
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
    
    def analyze(self) -> CouponAnalysisResult:
        """Run coupon effectiveness analysis."""
        logger.info("Starting coupon analysis")
        
        # Track coupon usage from invoices
        coupon_usage: dict[str, dict] = defaultdict(lambda: {
            "redemptions": 0,
            "discount_amount": 0.0,
            "customers": set(),
            "revenue": 0.0,
        })
        
        monthly_redemptions: dict[str, int] = defaultdict(int)
        
        # Analyze invoice discounts
        for inv in self._invoices:
            if inv.get("status") not in ("paid", "posted"):
                continue
            
            discounts = inv.get("discounts", [])
            customer_id = inv.get("customer_id")
            invoice_total = (inv.get("total", 0) or 0) / 100
            date = _parse_timestamp(inv.get("date"))
            
            for discount in discounts:
                entity_id = discount.get("entity_id")
                if not entity_id:
                    continue
                
                discount_amount = (discount.get("amount", 0) or 0) / 100
                
                coupon_usage[entity_id]["redemptions"] += 1
                coupon_usage[entity_id]["discount_amount"] += discount_amount
                if customer_id:
                    coupon_usage[entity_id]["customers"].add(customer_id)
                coupon_usage[entity_id]["revenue"] += invoice_total
                
                if date:
                    month_key = f"{date.year}-{date.month:02d}"
                    monthly_redemptions[month_key] += 1
        
        # Also check subscription coupons
        for sub in self._subscriptions:
            coupons = sub.get("coupons", [])
            customer_id = sub.get("customer_id")
            mrr = (sub.get("mrr", 0) or 0) / 100
            
            for coupon in coupons:
                coupon_id = coupon.get("coupon_id")
                if coupon_id:
                    coupon_usage[coupon_id]["customers"].add(customer_id)
        
        # Build coupon performance list
        coupon_map = {c.get("id"): c for c in self._coupons}
        performances: list[CouponPerformance] = []
        
        # Calculate total revenue from discounted customers
        customer_revenue: dict[str, float] = defaultdict(float)
        for inv in self._invoices:
            cid = inv.get("customer_id")
            if cid and inv.get("status") in ("paid", "posted"):
                customer_revenue[cid] += (inv.get("total", 0) or 0) / 100
        
        for coupon_id, usage in coupon_usage.items():
            coupon = coupon_map.get(coupon_id, {})
            
            # Calculate total revenue from customers who used this coupon
            total_customer_revenue = sum(
                customer_revenue.get(cid, 0) for cid in usage["customers"]
            )
            
            discount_given = usage["discount_amount"]
            roi = total_customer_revenue / discount_given if discount_given > 0 else 0
            
            num_customers = len(usage["customers"])
            avg_value = total_customer_revenue / num_customers if num_customers > 0 else 0
            
            # Calculate retention (customers still active)
            active_count = 0
            for cid in usage["customers"]:
                for sub in self._subscriptions:
                    if sub.get("customer_id") == cid and sub.get("status") == "active":
                        active_count += 1
                        break
            
            retention = active_count / num_customers if num_customers > 0 else None
            
            # Get discount details
            discount_type = coupon.get("discount_type", "unknown")
            if discount_type == "percentage":
                discount_value = coupon.get("discount_percentage", 0)
            else:
                discount_value = (coupon.get("discount_amount", 0) or 0) / 100
            
            performances.append(CouponPerformance(
                coupon_id=coupon_id,
                coupon_name=coupon.get("name", coupon_id),
                discount_type=discount_type,
                discount_value=discount_value,
                redemption_count=usage["redemptions"],
                total_discount_given=discount_given,
                revenue_from_users=total_customer_revenue,
                roi=roi,
                avg_customer_value=avg_value,
                retention_rate=retention,
                status=coupon.get("status", "unknown"),
            ))
        
        # Sort by ROI
        performances.sort(key=lambda x: x.roi, reverse=True)
        
        # Analyze by discount type
        by_type: dict[str, dict] = defaultdict(lambda: {
            "count": 0, "total_discount": 0.0, "total_revenue": 0.0
        })
        for p in performances:
            by_type[p.discount_type]["count"] += 1
            by_type[p.discount_type]["total_discount"] += p.total_discount_given
            by_type[p.discount_type]["total_revenue"] += p.revenue_from_users
        
        # Calculate summary
        total_discounts = sum(p.total_discount_given for p in performances)
        total_revenue = sum(p.revenue_from_users for p in performances)
        
        top_performing = [p.coupon_id for p in performances[:3] if p.roi > 2]
        underperforming = [p.coupon_id for p in performances if p.roi < 1 and p.redemption_count > 0]
        
        summary = DiscountAnalysisSummary(
            total_discounts_given=total_discounts,
            total_revenue_from_discounted=total_revenue,
            overall_roi=total_revenue / total_discounts if total_discounts > 0 else 0,
            discount_rate=total_discounts / total_revenue if total_revenue > 0 else 0,
            top_performing_coupons=top_performing,
            underperforming_coupons=underperforming[:5],
        )
        
        # Generate insights
        insights = self._generate_insights(performances, summary)
        
        result = CouponAnalysisResult(
            analysis_timestamp=datetime.now(timezone.utc),
            total_coupons=len(self._coupons),
            active_coupons=sum(1 for c in self._coupons if c.get("status") == "active"),
            total_redemptions=sum(p.redemption_count for p in performances),
            total_discount_amount=total_discounts,
            coupon_performances=performances,
            discount_by_type={k: dict(v) for k, v in by_type.items()},
            monthly_redemptions=dict(sorted(monthly_redemptions.items())),
            insights=insights,
            summary=summary,
        )
        
        logger.info(
            "Coupon analysis complete: %d coupons, $%.2f total discounts, %.1fx ROI",
            len(performances), total_discounts, summary.overall_roi
        )
        
        return result
    
    def _generate_insights(
        self, performances: list[CouponPerformance], summary: DiscountAnalysisSummary
    ) -> list[str]:
        """Generate insights from coupon analysis."""
        insights = []
        
        if summary.overall_roi > 3:
            insights.append(f"Excellent coupon ROI ({summary.overall_roi:.1f}x) - discounts driving strong revenue")
        elif summary.overall_roi > 1:
            insights.append(f"Positive coupon ROI ({summary.overall_roi:.1f}x) - discounts are profitable")
        else:
            insights.append(f"Low coupon ROI ({summary.overall_roi:.1f}x) - review discount strategy")
        
        if summary.top_performing_coupons:
            insights.append(f"Top performing coupons: {', '.join(summary.top_performing_coupons)}")
        
        if summary.underperforming_coupons:
            insights.append(f"Consider retiring: {', '.join(summary.underperforming_coupons[:3])}")
        
        # Check retention
        high_retention = [p for p in performances if p.retention_rate and p.retention_rate > 0.8]
        if high_retention:
            insights.append(f"{len(high_retention)} coupons have >80% customer retention")
        
        return insights
    
    def save_results(self, result: CouponAnalysisResult, output_dir: Path) -> Path:
        """Save analysis results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "coupon_analysis.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info("Coupon analysis saved to %s", output_path)
        return output_path
