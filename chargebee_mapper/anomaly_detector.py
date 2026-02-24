"""Billing anomaly detection.

Detects unusual patterns in invoices, transactions, and billing data.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import math

logger = logging.getLogger("chargebee_mapper.anomaly_detector")


@dataclass
class BillingAnomaly:
    """A detected billing anomaly."""
    anomaly_id: str
    anomaly_type: str
    severity: str  # "low", "medium", "high", "critical"
    entity_type: str  # "invoice", "transaction", "customer"
    entity_id: str
    customer_id: str | None
    description: str
    expected_value: float | None
    actual_value: float | None
    deviation_pct: float | None
    detected_at: datetime
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "customer_id": self.customer_id,
            "description": self.description,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "deviation_pct": round(self.deviation_pct, 2) if self.deviation_pct else None,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class AnomalyDetectionResult:
    """Complete anomaly detection results."""
    analysis_timestamp: datetime
    total_records_analyzed: int
    total_anomalies_detected: int
    anomalies_by_type: dict[str, int]
    anomalies_by_severity: dict[str, int]
    anomalies: list[BillingAnomaly]
    statistical_summary: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_records_analyzed": self.total_records_analyzed,
            "total_anomalies_detected": self.total_anomalies_detected,
            "anomalies_by_type": self.anomalies_by_type,
            "anomalies_by_severity": self.anomalies_by_severity,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "statistical_summary": self.statistical_summary,
        }


def _parse_timestamp(ts: int | None) -> datetime | None:
    """Convert Unix timestamp to datetime."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, OSError):
        return None


class AnomalyDetector:
    """Detects billing anomalies using statistical methods."""
    
    # Thresholds for anomaly detection
    ZSCORE_THRESHOLD = 3.0  # Standard deviations from mean
    HIGH_VALUE_MULTIPLIER = 5.0  # Multiple of median for high value alert
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.json_dir = data_dir / "json"
        
        self._invoices: list[dict] = []
        self._transactions: list[dict] = []
        self._customers: list[dict] = []
        
        self._anomaly_counter = 0
        
    def load_data(self) -> bool:
        """Load required JSON data files."""
        logger.info("Loading data from %s", self.json_dir)
        
        self._invoices = self._load_json("invoices.json")
        self._transactions = self._load_json("transactions.json")
        self._customers = self._load_json("customers.json")
        
        logger.info(
            "Loaded: %d invoices, %d transactions, %d customers",
            len(self._invoices), len(self._transactions), len(self._customers)
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
    
    def _next_anomaly_id(self) -> str:
        """Generate next anomaly ID."""
        self._anomaly_counter += 1
        return f"ANO-{self._anomaly_counter:04d}"
    
    def analyze(self) -> AnomalyDetectionResult:
        """Run anomaly detection analysis."""
        logger.info("Starting anomaly detection")
        
        anomalies: list[BillingAnomaly] = []
        
        # Detect invoice anomalies
        invoice_anomalies = self._detect_invoice_anomalies()
        anomalies.extend(invoice_anomalies)
        
        # Detect transaction anomalies
        txn_anomalies = self._detect_transaction_anomalies()
        anomalies.extend(txn_anomalies)
        
        # Detect customer billing anomalies
        customer_anomalies = self._detect_customer_anomalies()
        anomalies.extend(customer_anomalies)
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(key=lambda x: severity_order.get(x.severity, 4))
        
        # Calculate statistics
        by_type: dict[str, int] = defaultdict(int)
        by_severity: dict[str, int] = defaultdict(int)
        
        for a in anomalies:
            by_type[a.anomaly_type] += 1
            by_severity[a.severity] += 1
        
        # Calculate invoice statistics
        invoice_amounts = [(inv.get("total", 0) or 0) / 100 for inv in self._invoices]
        stats = self._calculate_statistics(invoice_amounts)
        
        result = AnomalyDetectionResult(
            analysis_timestamp=datetime.now(timezone.utc),
            total_records_analyzed=len(self._invoices) + len(self._transactions),
            total_anomalies_detected=len(anomalies),
            anomalies_by_type=dict(by_type),
            anomalies_by_severity=dict(by_severity),
            anomalies=anomalies,
            statistical_summary={
                "invoice_stats": stats,
                "thresholds": {
                    "zscore_threshold": self.ZSCORE_THRESHOLD,
                    "high_value_multiplier": self.HIGH_VALUE_MULTIPLIER,
                }
            },
        )
        
        logger.info(
            "Anomaly detection complete: %d anomalies detected (%d critical, %d high)",
            len(anomalies),
            by_severity.get("critical", 0),
            by_severity.get("high", 0)
        )
        
        return result
    
    def _calculate_statistics(self, values: list[float]) -> dict[str, float]:
        """Calculate statistical measures for a list of values."""
        if not values:
            return {"mean": 0, "median": 0, "std_dev": 0, "min": 0, "max": 0}
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        mean = sum(values) / n
        median = sorted_vals[n // 2]
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = math.sqrt(variance)
        
        return {
            "mean": round(mean, 2),
            "median": round(median, 2),
            "std_dev": round(std_dev, 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "count": n,
        }
    
    def _detect_invoice_anomalies(self) -> list[BillingAnomaly]:
        """Detect anomalies in invoice data."""
        anomalies = []
        
        # Calculate invoice statistics
        amounts = [(inv.get("total", 0) or 0) / 100 for inv in self._invoices if inv.get("total")]
        
        if not amounts:
            return anomalies
        
        mean = sum(amounts) / len(amounts)
        sorted_amounts = sorted(amounts)
        median = sorted_amounts[len(amounts) // 2]
        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        std_dev = math.sqrt(variance) if variance > 0 else 1
        
        for inv in self._invoices:
            amount = (inv.get("total", 0) or 0) / 100
            invoice_id = inv.get("id", "unknown")
            customer_id = inv.get("customer_id")
            
            # Check for unusually high amount (z-score)
            if std_dev > 0:
                zscore = (amount - mean) / std_dev
                
                if zscore > self.ZSCORE_THRESHOLD:
                    severity = "critical" if zscore > 5 else "high"
                    anomalies.append(BillingAnomaly(
                        anomaly_id=self._next_anomaly_id(),
                        anomaly_type="unusually_high_invoice",
                        severity=severity,
                        entity_type="invoice",
                        entity_id=invoice_id,
                        customer_id=customer_id,
                        description=f"Invoice amount ${amount:.2f} is {zscore:.1f} std devs above mean",
                        expected_value=mean,
                        actual_value=amount,
                        deviation_pct=((amount - mean) / mean) * 100 if mean > 0 else None,
                        detected_at=datetime.now(timezone.utc),
                    ))
            
            # Check for zero or negative amounts (should be rare)
            if amount <= 0 and inv.get("status") in ("paid", "posted"):
                anomalies.append(BillingAnomaly(
                    anomaly_id=self._next_anomaly_id(),
                    anomaly_type="zero_or_negative_invoice",
                    severity="medium",
                    entity_type="invoice",
                    entity_id=invoice_id,
                    customer_id=customer_id,
                    description=f"Invoice has zero or negative amount: ${amount:.2f}",
                    expected_value=median,
                    actual_value=amount,
                    deviation_pct=None,
                    detected_at=datetime.now(timezone.utc),
                ))
            
            # Check for very old unpaid invoices
            if inv.get("status") == "payment_due":
                due_date = _parse_timestamp(inv.get("due_date"))
                if due_date:
                    days_overdue = (datetime.now(timezone.utc) - due_date).days
                    if days_overdue > 90:
                        anomalies.append(BillingAnomaly(
                            anomaly_id=self._next_anomaly_id(),
                            anomaly_type="severely_overdue_invoice",
                            severity="high" if days_overdue > 180 else "medium",
                            entity_type="invoice",
                            entity_id=invoice_id,
                            customer_id=customer_id,
                            description=f"Invoice is {days_overdue} days overdue (${amount:.2f})",
                            expected_value=0,
                            actual_value=days_overdue,
                            deviation_pct=None,
                            detected_at=datetime.now(timezone.utc),
                        ))
        
        return anomalies
    
    def _detect_transaction_anomalies(self) -> list[BillingAnomaly]:
        """Detect anomalies in transaction data."""
        anomalies = []
        
        for txn in self._transactions:
            txn_id = txn.get("id", "unknown")
            customer_id = txn.get("customer_id")
            amount = (txn.get("amount", 0) or 0) / 100
            
            # Check for fraud flags
            fraud_flag = txn.get("fraud_flag")
            if fraud_flag and fraud_flag != "safe":
                anomalies.append(BillingAnomaly(
                    anomaly_id=self._next_anomaly_id(),
                    anomaly_type="fraud_flagged_transaction",
                    severity="critical" if fraud_flag == "fraudulent" else "high",
                    entity_type="transaction",
                    entity_id=txn_id,
                    customer_id=customer_id,
                    description=f"Transaction flagged as '{fraud_flag}' (${amount:.2f})",
                    expected_value=None,
                    actual_value=amount,
                    deviation_pct=None,
                    detected_at=datetime.now(timezone.utc),
                ))
            
            # Check for unusual failure patterns
            if txn.get("status") == "failure":
                auth_reason = txn.get("authorization_reason", "")
                if "fraud" in auth_reason.lower() or "stolen" in auth_reason.lower():
                    anomalies.append(BillingAnomaly(
                        anomaly_id=self._next_anomaly_id(),
                        anomaly_type="suspicious_failure_reason",
                        severity="high",
                        entity_type="transaction",
                        entity_id=txn_id,
                        customer_id=customer_id,
                        description=f"Transaction failed with suspicious reason: {auth_reason}",
                        expected_value=None,
                        actual_value=amount,
                        deviation_pct=None,
                        detected_at=datetime.now(timezone.utc),
                    ))
        
        return anomalies
    
    def _detect_customer_anomalies(self) -> list[BillingAnomaly]:
        """Detect customer-level billing anomalies."""
        anomalies = []
        
        # Group invoices by customer
        invoices_by_customer: dict[str, list[dict]] = defaultdict(list)
        for inv in self._invoices:
            cid = inv.get("customer_id")
            if cid:
                invoices_by_customer[cid].append(inv)
        
        for cid, invoices in invoices_by_customer.items():
            if len(invoices) < 3:
                continue
            
            # Check for sudden large changes
            for i, inv in enumerate(invoices[1:], 1):
                current = (inv.get("total", 0) or 0) / 100
                prev_inv = invoices[i - 1]
                previous = (prev_inv.get("total", 0) or 0) / 100
                
                if previous > 0 and current > 0:
                    change_pct = ((current - previous) / previous) * 100
                    
                    # Flag >500% increase
                    if change_pct > 500:
                        anomalies.append(BillingAnomaly(
                            anomaly_id=self._next_anomaly_id(),
                            anomaly_type="sudden_billing_spike",
                            severity="medium",
                            entity_type="customer",
                            entity_id=cid,
                            customer_id=cid,
                            description=f"Invoice jumped from ${previous:.2f} to ${current:.2f} ({change_pct:.0f}% increase)",
                            expected_value=previous,
                            actual_value=current,
                            deviation_pct=change_pct,
                            detected_at=datetime.now(timezone.utc),
                        ))
        
        return anomalies
    
    def save_results(self, result: AnomalyDetectionResult, output_dir: Path) -> Path:
        """Save analysis results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "anomaly_detection.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info("Anomaly detection saved to %s", output_path)
        return output_path
