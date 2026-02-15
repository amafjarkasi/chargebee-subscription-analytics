"""Churn prediction analyzer using rule-based scoring.

This module analyzes Chargebee customer data to predict churn risk
using a weighted scoring system based on:
- Payment failures and dunning status
- Subscription downgrades and changes
- Billing recency and renewal timing
- Customer tenure and lifetime value
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("chargebee_mapper.churn_analyzer")

# Scoring weights (0-100 scale, higher = more churn risk)
WEIGHTS = {
    # Payment failures (max 30 points)
    "payment_failed_recent": 15,      # Failed payment in last 30 days
    "payment_failed_multiple": 10,    # Multiple failed payments
    "dunning_active": 5,              # Currently in dunning
    
    # Subscription downgrades (max 25 points)
    "plan_downgraded": 15,            # Downgraded plan recently
    "quantity_reduced": 10,           # Reduced quantity/seats
    
    # Billing recency (max 25 points)
    "no_recent_invoice": 10,          # No invoice in 60+ days
    "renewal_soon_no_activity": 15,   # Renewal in 30 days, low engagement
    
    # Customer tenure/value (max 20 points)
    "new_customer": 10,               # Customer < 90 days old
    "low_lifetime_value": 5,          # Below median LTV
    "single_subscription": 5,         # Only one subscription (less sticky)
}


@dataclass
class ChurnSignal:
    """A detected churn risk signal."""
    signal_type: str
    description: str
    points: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CustomerChurnScore:
    """Churn risk assessment for a single customer."""
    customer_id: str
    email: str | None
    company: str | None
    total_score: int  # 0-100, higher = more risk
    risk_level: str   # "low", "medium", "high", "critical"
    signals: list[ChurnSignal] = field(default_factory=list)
    subscription_count: int = 0
    total_mrr: float = 0.0
    customer_since: datetime | None = None
    last_payment_date: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "email": self.email,
            "company": self.company,
            "total_score": self.total_score,
            "risk_level": self.risk_level,
            "signals": [
                {
                    "type": s.signal_type,
                    "description": s.description,
                    "points": s.points,
                    "details": s.details,
                }
                for s in self.signals
            ],
            "subscription_count": self.subscription_count,
            "total_mrr": self.total_mrr,
            "customer_since": self.customer_since.isoformat() if self.customer_since else None,
            "last_payment_date": self.last_payment_date.isoformat() if self.last_payment_date else None,
        }


@dataclass
class ChurnAnalysisResult:
    """Complete churn analysis results."""
    analysis_timestamp: datetime
    prediction_window_days: int
    total_customers_analyzed: int
    customers_at_risk: int
    risk_distribution: dict[str, int]  # risk_level -> count
    total_mrr_at_risk: float
    customer_scores: list[CustomerChurnScore]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "prediction_window_days": self.prediction_window_days,
            "total_customers_analyzed": self.total_customers_analyzed,
            "customers_at_risk": self.customers_at_risk,
            "risk_distribution": self.risk_distribution,
            "total_mrr_at_risk": round(self.total_mrr_at_risk, 2),
            "customer_scores": [c.to_dict() for c in self.customer_scores],
        }


def _parse_timestamp(ts: int | None) -> datetime | None:
    """Convert Unix timestamp to datetime."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, OSError):
        return None


def _days_ago(dt: datetime | None) -> int | None:
    """Calculate days since a given datetime."""
    if dt is None:
        return None
    now = datetime.now(timezone.utc)
    return (now - dt).days


def _get_risk_level(score: int) -> str:
    """Convert numeric score to risk level."""
    if score >= 70:
        return "critical"
    elif score >= 50:
        return "high"
    elif score >= 30:
        return "medium"
    return "low"


class ChurnAnalyzer:
    """Analyzes customer data to predict churn risk using rule-based scoring."""
    
    def __init__(self, data_dir: Path, prediction_window_days: int = 90):
        self.data_dir = data_dir
        self.prediction_window_days = prediction_window_days
        self.json_dir = data_dir / "json"
        
        # Data stores
        self._customers: list[dict] = []
        self._subscriptions: list[dict] = []
        self._invoices: list[dict] = []
        self._transactions: list[dict] = []
        self._credit_notes: list[dict] = []
        
        # Indexed lookups
        self._subs_by_customer: dict[str, list[dict]] = {}
        self._invoices_by_customer: dict[str, list[dict]] = {}
        self._transactions_by_customer: dict[str, list[dict]] = {}

    def load_data(self) -> bool:
        """Load required JSON data files. Returns True if successful."""
        logger.info("Loading data from %s", self.json_dir)
        
        required_files = ["customers.json"]
        optional_files = [
            "subscriptions.json",
            "invoices.json",
            "transactions.json",
            "credit_notes.json",
        ]
        
        # Check required files exist
        for fname in required_files:
            fpath = self.json_dir / fname
            if not fpath.exists():
                logger.error("Required file not found: %s", fpath)
                return False

        # Load data files
        self._customers = self._load_json("customers.json")
        self._subscriptions = self._load_json("subscriptions.json")
        self._invoices = self._load_json("invoices.json")
        self._transactions = self._load_json("transactions.json")
        self._credit_notes = self._load_json("credit_notes.json")
        
        logger.info(
            "Loaded: %d customers, %d subscriptions, %d invoices, %d transactions",
            len(self._customers),
            len(self._subscriptions),
            len(self._invoices),
            len(self._transactions),
        )
        
        # Build indexes
        self._build_indexes()
        
        return True
    
    def _load_json(self, filename: str) -> list[dict]:
        """Load a JSON file, returning empty list if not found."""
        fpath = self.json_dir / filename
        if not fpath.exists():
            logger.debug("File not found (optional): %s", fpath)
            return []
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load %s: %s", fpath, e)
            return []

    def _build_indexes(self) -> None:
        """Build lookup indexes by customer ID."""
        # Index subscriptions by customer_id
        for sub in self._subscriptions:
            cid = sub.get("customer_id")
            if cid:
                self._subs_by_customer.setdefault(cid, []).append(sub)
        
        # Index invoices by customer_id
        for inv in self._invoices:
            cid = inv.get("customer_id")
            if cid:
                self._invoices_by_customer.setdefault(cid, []).append(inv)
        
        # Index transactions by customer_id
        for txn in self._transactions:
            cid = txn.get("customer_id")
            if cid:
                self._transactions_by_customer.setdefault(cid, []).append(txn)
    
    def analyze(self) -> ChurnAnalysisResult:
        """Run churn analysis on all customers."""
        logger.info("Starting churn analysis for %d customers", len(self._customers))
        
        scores: list[CustomerChurnScore] = []
        
        for customer in self._customers:
            # Skip deleted customers
            if customer.get("deleted"):
                continue
            
            score = self._analyze_customer(customer)
            scores.append(score)
        
        # Sort by risk score (highest first)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Calculate aggregates
        risk_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        mrr_at_risk = 0.0
        at_risk_count = 0
        
        for s in scores:
            risk_dist[s.risk_level] = risk_dist.get(s.risk_level, 0) + 1
            if s.risk_level in ("high", "critical"):
                at_risk_count += 1
                mrr_at_risk += s.total_mrr
        
        result = ChurnAnalysisResult(
            analysis_timestamp=datetime.now(timezone.utc),
            prediction_window_days=self.prediction_window_days,
            total_customers_analyzed=len(scores),
            customers_at_risk=at_risk_count,
            risk_distribution=risk_dist,
            total_mrr_at_risk=mrr_at_risk,
            customer_scores=scores,
        )
        
        logger.info(
            "Analysis complete: %d customers, %d at risk (%.1f%%), $%.2f MRR at risk",
            result.total_customers_analyzed,
            result.customers_at_risk,
            (result.customers_at_risk / max(result.total_customers_analyzed, 1)) * 100,
            result.total_mrr_at_risk,
        )
        
        return result
    
    def _analyze_customer(self, customer: dict) -> CustomerChurnScore:
        """Analyze a single customer and generate their churn score."""
        customer_id = customer.get("id", "unknown")
        signals: list[ChurnSignal] = []
        
        # Get related data
        subscriptions = self._subs_by_customer.get(customer_id, [])
        invoices = self._invoices_by_customer.get(customer_id, [])
        transactions = self._transactions_by_customer.get(customer_id, [])
        
        # Calculate customer metrics
        customer_since = _parse_timestamp(customer.get("created_at"))
        active_subs = [s for s in subscriptions if s.get("status") == "active"]
        total_mrr = self._calculate_mrr(active_subs)
        
        # Find last successful payment
        successful_txns = [
            t for t in transactions 
            if t.get("status") == "success" and t.get("type") == "payment"
        ]
        last_payment = None
        if successful_txns:
            latest = max(successful_txns, key=lambda x: x.get("date", 0))
            last_payment = _parse_timestamp(latest.get("date"))
        
        # ===== CHECK PAYMENT FAILURE SIGNALS =====
        signals.extend(self._check_payment_signals(transactions, invoices))
        
        # ===== CHECK SUBSCRIPTION DOWNGRADE SIGNALS =====
        signals.extend(self._check_subscription_signals(subscriptions))
        
        # ===== CHECK BILLING RECENCY SIGNALS =====
        signals.extend(self._check_billing_signals(invoices, active_subs))
        
        # ===== CHECK TENURE/VALUE SIGNALS =====
        signals.extend(self._check_tenure_signals(customer, subscriptions, total_mrr))
        
        # Calculate total score (cap at 100)
        total_score = min(100, sum(s.points for s in signals))
        risk_level = _get_risk_level(total_score)
        
        return CustomerChurnScore(
            customer_id=customer_id,
            email=customer.get("email"),
            company=customer.get("company"),
            total_score=total_score,
            risk_level=risk_level,
            signals=signals,
            subscription_count=len(active_subs),
            total_mrr=total_mrr,
            customer_since=customer_since,
            last_payment_date=last_payment,
        )
    
    def _check_payment_signals(
        self, transactions: list[dict], invoices: list[dict]
    ) -> list[ChurnSignal]:
        """Check for payment failure signals."""
        signals = []
        
        # Count failed transactions
        failed_txns = [
            t for t in transactions 
            if t.get("status") in ("failure", "voided")
        ]
        recent_failures = [
            t for t in failed_txns
            if _days_ago(_parse_timestamp(t.get("date"))) is not None
            and _days_ago(_parse_timestamp(t.get("date"))) <= 30
        ]
        
        if recent_failures:
            signals.append(ChurnSignal(
                signal_type="payment_failed_recent",
                description=f"Payment failed in last 30 days ({len(recent_failures)} failures)",
                points=WEIGHTS["payment_failed_recent"],
                details={"failure_count": len(recent_failures)},
            ))
        
        if len(failed_txns) >= 3:
            signals.append(ChurnSignal(
                signal_type="payment_failed_multiple",
                description=f"Multiple payment failures ({len(failed_txns)} total)",
                points=WEIGHTS["payment_failed_multiple"],
                details={"total_failures": len(failed_txns)},
            ))
        
        # Check for dunning status on invoices
        dunning_invoices = [
            i for i in invoices 
            if i.get("dunning_status") in ("in_progress", "exhausted", "stopped")
        ]
        if dunning_invoices:
            signals.append(ChurnSignal(
                signal_type="dunning_active",
                description=f"Account in dunning ({len(dunning_invoices)} invoices)",
                points=WEIGHTS["dunning_active"],
                details={"dunning_invoice_count": len(dunning_invoices)},
            ))
        
        return signals
    
    def _check_subscription_signals(self, subscriptions: list[dict]) -> list[ChurnSignal]:
        """Check for subscription downgrade signals."""
        signals = []
        
        # Look for cancelled or paused subscriptions (indicates churn behavior)
        cancelled_subs = [
            s for s in subscriptions 
            if s.get("status") in ("cancelled", "non_renewing")
        ]
        
        if cancelled_subs:
            # Check if cancelled recently
            recent_cancels = [
                s for s in cancelled_subs
                if _days_ago(_parse_timestamp(s.get("cancelled_at"))) is not None
                and _days_ago(_parse_timestamp(s.get("cancelled_at"))) <= 90
            ]
            if recent_cancels:
                signals.append(ChurnSignal(
                    signal_type="plan_downgraded",
                    description=f"Subscription cancelled/non-renewing recently ({len(recent_cancels)})",
                    points=WEIGHTS["plan_downgraded"],
                    details={"cancelled_count": len(recent_cancels)},
                ))
        
        # Check for reduced quantity (comparing subscription_items if available)
        for sub in subscriptions:
            items = sub.get("subscription_items", [])
            for item in items:
                if item.get("quantity", 1) == 1 and item.get("unit_price", 0) == 0:
                    # Minimal/free plan indicator
                    signals.append(ChurnSignal(
                        signal_type="quantity_reduced",
                        description="Subscription at minimal quantity/free tier",
                        points=WEIGHTS["quantity_reduced"],
                        details={"subscription_id": sub.get("id")},
                    ))
                    break
        
        return signals
    
    def _check_billing_signals(
        self, invoices: list[dict], active_subs: list[dict]
    ) -> list[ChurnSignal]:
        """Check for billing recency signals."""
        signals = []
        
        # Check for recent invoice activity
        if invoices:
            latest_invoice = max(invoices, key=lambda x: x.get("date", 0))
            days_since_invoice = _days_ago(_parse_timestamp(latest_invoice.get("date")))
            
            if days_since_invoice is not None and days_since_invoice > 60:
                signals.append(ChurnSignal(
                    signal_type="no_recent_invoice",
                    description=f"No invoice in {days_since_invoice} days",
                    points=WEIGHTS["no_recent_invoice"],
                    details={"days_since_invoice": days_since_invoice},
                ))
        elif not invoices:
            # No invoices at all
            signals.append(ChurnSignal(
                signal_type="no_recent_invoice",
                description="No invoices found for customer",
                points=WEIGHTS["no_recent_invoice"],
                details={},
            ))
        
        # Check for upcoming renewal with no recent engagement
        for sub in active_subs:
            next_billing = _parse_timestamp(sub.get("next_billing_at"))
            if next_billing:
                days_until_renewal = -_days_ago(next_billing) if _days_ago(next_billing) else None
                if days_until_renewal is not None and 0 < days_until_renewal <= 30:
                    # Renewal coming up - check for low engagement signals
                    signals.append(ChurnSignal(
                        signal_type="renewal_soon_no_activity",
                        description=f"Renewal in {days_until_renewal} days",
                        points=WEIGHTS["renewal_soon_no_activity"],
                        details={
                            "days_until_renewal": days_until_renewal,
                            "subscription_id": sub.get("id"),
                        },
                    ))
                    break  # Only count once
        
        return signals
    
    def _check_tenure_signals(
        self, customer: dict, subscriptions: list[dict], total_mrr: float
    ) -> list[ChurnSignal]:
        """Check for customer tenure and value signals."""
        signals = []
        
        # Check if new customer (< 90 days)
        customer_since = _parse_timestamp(customer.get("created_at"))
        if customer_since:
            days_as_customer = _days_ago(customer_since)
            if days_as_customer is not None and days_as_customer < 90:
                signals.append(ChurnSignal(
                    signal_type="new_customer",
                    description=f"New customer ({days_as_customer} days)",
                    points=WEIGHTS["new_customer"],
                    details={"days_as_customer": days_as_customer},
                ))
        
        # Check for low LTV (simplified: MRR below threshold)
        if total_mrr < 50:  # $50 MRR threshold
            signals.append(ChurnSignal(
                signal_type="low_lifetime_value",
                description=f"Low MRR (${total_mrr:.2f}/month)",
                points=WEIGHTS["low_lifetime_value"],
                details={"mrr": total_mrr},
            ))
        
        # Check for single subscription (less sticky)
        active_subs = [s for s in subscriptions if s.get("status") == "active"]
        if len(active_subs) == 1:
            signals.append(ChurnSignal(
                signal_type="single_subscription",
                description="Single subscription (lower stickiness)",
                points=WEIGHTS["single_subscription"],
                details={"subscription_count": 1},
            ))
        
        return signals
    
    def _calculate_mrr(self, active_subscriptions: list[dict]) -> float:
        """Calculate total MRR from active subscriptions."""
        total = 0.0
        for sub in active_subscriptions:
            # Try to get MRR from subscription data
            mrr = sub.get("mrr", 0)
            if mrr:
                total += mrr / 100  # Chargebee stores amounts in cents
            else:
                # Fallback: sum up item prices
                items = sub.get("subscription_items", [])
                for item in items:
                    amount = item.get("amount", 0) or item.get("unit_price", 0)
                    quantity = item.get("quantity", 1)
                    total += (amount * quantity) / 100
        return total

    def save_results(self, result: ChurnAnalysisResult, output_dir: Path) -> Path:
        """Save analysis results to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "churn_analysis.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info("Churn analysis saved to %s", output_path)
        return output_path
