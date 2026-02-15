"""Payment failure prediction using historical transaction data.

Identifies customers and payment sources at risk of payment failure.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import days_ago, load_json_file, parse_timestamp

logger = logging.getLogger("chargebee_mapper.payment_failure_predictor")


@dataclass
class PaymentRiskScore:
    """Payment failure risk assessment for a customer."""
    customer_id: str
    email: str | None
    company: str | None
    risk_score: int  # 0-100
    risk_level: str  # "low", "medium", "high", "critical"
    total_transactions: int
    failed_transactions: int
    failure_rate: float
    recent_failures: int  # Last 90 days
    payment_method: str | None
    card_expiring_soon: bool
    last_success_days_ago: int | None
    risk_factors: list[str]
    recommended_actions: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "email": self.email,
            "company": self.company,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "total_transactions": self.total_transactions,
            "failed_transactions": self.failed_transactions,
            "failure_rate": round(self.failure_rate, 4),
            "recent_failures": self.recent_failures,
            "payment_method": self.payment_method,
            "card_expiring_soon": self.card_expiring_soon,
            "last_success_days_ago": self.last_success_days_ago,
            "risk_factors": self.risk_factors,
            "recommended_actions": self.recommended_actions,
        }


@dataclass
class PaymentFailureAnalysisResult:
    """Complete payment failure analysis results."""
    analysis_timestamp: datetime
    total_customers_analyzed: int
    customers_at_risk: int
    total_transactions: int
    failed_transactions: int
    overall_failure_rate: float
    failure_by_payment_method: dict[str, dict]
    failure_by_gateway: dict[str, dict]
    risk_distribution: dict[str, int]
    customer_scores: list[PaymentRiskScore]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_customers_analyzed": self.total_customers_analyzed,
            "customers_at_risk": self.customers_at_risk,
            "total_transactions": self.total_transactions,
            "failed_transactions": self.failed_transactions,
            "overall_failure_rate": round(self.overall_failure_rate, 4),
            "failure_by_payment_method": self.failure_by_payment_method,
            "failure_by_gateway": self.failure_by_gateway,
            "risk_distribution": self.risk_distribution,
            "customer_scores": [c.to_dict() for c in self.customer_scores],
        }


def _get_risk_level(score: int) -> str:
    """Convert numeric score to risk level."""
    if score >= 70:
        return "critical"
    elif score >= 50:
        return "high"
    elif score >= 30:
        return "medium"
    return "low"


class PaymentFailurePredictor:
    """Predicts payment failure risk using transaction history."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.json_dir = data_dir / "json"
        
        self._customers: list[dict] = []
        self._transactions: list[dict] = []
        self._payment_sources: list[dict] = []
        self._invoices: list[dict] = []
        self._loaded = False

    def load_data(self) -> bool:
        """Load required JSON data files."""
        logger.info("Loading data from %s", self.json_dir)
        
        self._customers = load_json_file(self.json_dir / "customers.json")
        self._transactions = load_json_file(self.json_dir / "transactions.json")
        self._payment_sources = load_json_file(self.json_dir / "payment_sources.json")
        self._invoices = load_json_file(self.json_dir / "invoices.json")
        
        if not self._transactions:
            logger.error("No transaction data found")
            return False
            
        logger.info(
            "Loaded: %d customers, %d transactions, %d payment sources",
            len(self._customers), len(self._transactions), len(self._payment_sources)
        )
        self._loaded = True
        return True
    
    def analyze(self) -> PaymentFailureAnalysisResult:
        """Run payment failure analysis."""
        if not self._loaded:
            if not self.load_data():
                 raise RuntimeError("Cannot proceed without data")

        logger.info("Starting payment failure analysis")
        
        # Index data by customer
        txns_by_customer: dict[str, list[dict]] = defaultdict(list)
        for txn in self._transactions:
            cid = txn.get("customer_id")
            if cid:
                txns_by_customer[cid].append(txn)
        
        ps_by_customer: dict[str, list[dict]] = defaultdict(list)
        for ps in self._payment_sources:
            cid = ps.get("customer_id")
            if cid:
                ps_by_customer[cid].append(ps)
        
        # Analyze failure rates by payment method and gateway
        method_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "failed": 0})
        gateway_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "failed": 0})
        
        for txn in self._transactions:
            if txn.get("type") != "payment":
                continue
            
            method = txn.get("payment_method", "unknown")
            gateway = txn.get("gateway", "unknown")
            status = txn.get("status")
            
            method_stats[method]["total"] += 1
            gateway_stats[gateway]["total"] += 1
            
            if status in ("failure", "voided"):
                method_stats[method]["failed"] += 1
                gateway_stats[gateway]["failed"] += 1
        
        # Calculate failure rates
        for stats in method_stats.values():
            stats["failure_rate"] = stats["failed"] / max(stats["total"], 1)
        for stats in gateway_stats.values():
            stats["failure_rate"] = stats["failed"] / max(stats["total"], 1)
        
        # Analyze each customer
        customer_scores: list[PaymentRiskScore] = []
        customer_map = {c.get("id"): c for c in self._customers}
        
        for cid, txns in txns_by_customer.items():
            customer = customer_map.get(cid, {})
            if customer.get("deleted"):
                continue
            
            payment_sources = ps_by_customer.get(cid, [])
            score = self._analyze_customer(customer, txns, payment_sources, method_stats)
            customer_scores.append(score)
        
        # Sort by risk score
        customer_scores.sort(key=lambda x: x.risk_score, reverse=True)
        
        # Calculate aggregates
        total_txns = sum(1 for t in self._transactions if t.get("type") == "payment")
        failed_txns = sum(
            1 for t in self._transactions 
            if t.get("type") == "payment" and t.get("status") in ("failure", "voided")
        )
        
        risk_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for s in customer_scores:
            risk_dist[s.risk_level] += 1
        
        at_risk = sum(1 for s in customer_scores if s.risk_level in ("high", "critical"))
        
        result = PaymentFailureAnalysisResult(
            analysis_timestamp=datetime.now(timezone.utc),
            total_customers_analyzed=len(customer_scores),
            customers_at_risk=at_risk,
            total_transactions=total_txns,
            failed_transactions=failed_txns,
            overall_failure_rate=failed_txns / max(total_txns, 1),
            failure_by_payment_method=dict(method_stats),
            failure_by_gateway=dict(gateway_stats),
            risk_distribution=risk_dist,
            customer_scores=customer_scores,
        )
        
        logger.info(
            "Payment failure analysis complete: %d customers, %d at risk, %.1f%% overall failure rate",
            len(customer_scores), at_risk, (failed_txns / max(total_txns, 1)) * 100
        )
        
        return result
    
    def _analyze_customer(
        self,
        customer: dict,
        transactions: list[dict],
        payment_sources: list[dict],
        method_stats: dict[str, dict],
    ) -> PaymentRiskScore:
        """Analyze payment failure risk for a single customer."""
        customer_id = customer.get("id", "unknown")
        risk_factors = []
        recommended_actions = []
        risk_score = 0
        
        # Filter to payment transactions only
        payment_txns = [t for t in transactions if t.get("type") == "payment"]
        failed_txns = [t for t in payment_txns if t.get("status") in ("failure", "voided")]
        success_txns = [t for t in payment_txns if t.get("status") == "success"]
        
        total_count = len(payment_txns)
        failed_count = len(failed_txns)
        failure_rate = failed_count / max(total_count, 1)
        
        # Check recent failures (last 90 days)
        recent_failures = sum(
            1 for t in failed_txns
            if days_ago(parse_timestamp(t.get("date"))) is not None
            and days_ago(parse_timestamp(t.get("date"))) <= 90
        )
        
        # Last successful payment
        last_success_days = None
        if success_txns:
            latest = max(success_txns, key=lambda x: x.get("date", 0))
            last_success_days = days_ago(parse_timestamp(latest.get("date")))
        
        # Get primary payment method
        primary_method = None
        if payment_txns:
            methods = [t.get("payment_method") for t in payment_txns if t.get("payment_method")]
            if methods:
                primary_method = max(set(methods), key=methods.count)
        
        # Check if card is expiring soon
        card_expiring = False
        for ps in payment_sources:
            if ps.get("type") == "card":
                card = ps.get("card", {})
                exp_month = card.get("expiry_month")
                exp_year = card.get("expiry_year")
                if exp_month and exp_year:
                    now = datetime.now(timezone.utc)
                    if exp_year < now.year or (exp_year == now.year and exp_month <= now.month + 2):
                        card_expiring = True
                        break
        
        # === SCORING RULES ===
        
        # High failure rate
        if failure_rate > 0.3:
            risk_score += 30
            risk_factors.append(f"High failure rate: {failure_rate:.1%}")
            recommended_actions.append("Contact customer to update payment method")
        elif failure_rate > 0.1:
            risk_score += 15
            risk_factors.append(f"Elevated failure rate: {failure_rate:.1%}")
        
        # Recent failures
        if recent_failures >= 3:
            risk_score += 25
            risk_factors.append(f"Multiple recent failures: {recent_failures} in 90 days")
            recommended_actions.append("Implement smart retry schedule")
        elif recent_failures >= 1:
            risk_score += 10
            risk_factors.append(f"Recent failure: {recent_failures} in 90 days")
        
        # Card expiring soon
        if card_expiring:
            risk_score += 20
            risk_factors.append("Payment card expiring soon")
            recommended_actions.append("Send card update reminder")
        
        # No recent successful payment
        if last_success_days is not None and last_success_days > 60:
            risk_score += 15
            risk_factors.append(f"No successful payment in {last_success_days} days")
        
        # Payment method with high failure rate
        if primary_method and primary_method in method_stats:
            method_failure_rate = method_stats[primary_method].get("failure_rate", 0)
            if method_failure_rate > 0.15:
                risk_score += 10
                risk_factors.append(f"Payment method ({primary_method}) has high failure rate")
                recommended_actions.append("Suggest alternative payment method")
        
        # No payment sources on file
        if not payment_sources:
            risk_score += 15
            risk_factors.append("No payment source on file")
            recommended_actions.append("Request payment method setup")
        
        risk_score = min(100, risk_score)
        risk_level = _get_risk_level(risk_score)
        
        return PaymentRiskScore(
            customer_id=customer_id,
            email=customer.get("email"),
            company=customer.get("company"),
            risk_score=risk_score,
            risk_level=risk_level,
            total_transactions=total_count,
            failed_transactions=failed_count,
            failure_rate=failure_rate,
            recent_failures=recent_failures,
            payment_method=primary_method,
            card_expiring_soon=card_expiring,
            last_success_days_ago=last_success_days,
            risk_factors=risk_factors,
            recommended_actions=recommended_actions,
        )
    
    def save_results(self, result: PaymentFailureAnalysisResult, output_dir: Path) -> Path:
        """Save analysis results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "payment_failure_analysis.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info("Payment failure analysis saved to %s", output_path)
        return output_path
