"""Event sequence analysis and pattern mining.

Analyzes event sequences to identify common customer journeys and patterns.
"""

import json
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .utils import parse_timestamp

logger = logging.getLogger("chargebee_mapper.event_sequence_analyzer")


@dataclass
class EventSequencePattern:
    """A common event sequence pattern."""
    pattern: list[str]
    frequency: int
    pct_of_customers: float
    avg_time_between_events: float  # Hours
    outcome: str | None  # "churned", "upgraded", "active", etc.
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "frequency": self.frequency,
            "pct_of_customers": round(self.pct_of_customers, 4),
            "avg_time_between_events": round(self.avg_time_between_events, 2),
            "outcome": self.outcome,
        }


@dataclass
class CustomerJourney:
    """Event journey for a single customer."""
    customer_id: str
    email: str | None
    events: list[dict]  # Simplified event data
    journey_length_days: int
    key_milestones: list[str]
    current_status: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "email": self.email,
            "event_count": len(self.events),
            "events": self.events[:20],  # Limit to first 20
            "journey_length_days": self.journey_length_days,
            "key_milestones": self.key_milestones,
            "current_status": self.current_status,
        }


@dataclass
class EventSequenceAnalysisResult:
    """Complete event sequence analysis results."""
    analysis_timestamp: datetime
    total_events: int
    total_customers_with_events: int
    event_type_distribution: dict[str, int]
    common_patterns: list[EventSequencePattern]
    pre_churn_patterns: list[EventSequencePattern]
    pre_upgrade_patterns: list[EventSequencePattern]
    sample_journeys: list[CustomerJourney]
    insights: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_events": self.total_events,
            "total_customers_with_events": self.total_customers_with_events,
            "event_type_distribution": self.event_type_distribution,
            "common_patterns": [p.to_dict() for p in self.common_patterns],
            "pre_churn_patterns": [p.to_dict() for p in self.pre_churn_patterns],
            "pre_upgrade_patterns": [p.to_dict() for p in self.pre_upgrade_patterns],
            "sample_journeys": [j.to_dict() for j in self.sample_journeys],
            "insights": self.insights,
        }


class EventSequenceAnalyzer:
    """Analyzes event sequences to find patterns."""
    
    # Key event types to track
    KEY_EVENTS = {
        "customer_created", "customer_changed", "customer_deleted",
        "subscription_created", "subscription_activated", "subscription_changed",
        "subscription_cancelled", "subscription_renewed", "subscription_paused",
        "invoice_generated", "invoice_updated", "invoice_deleted",
        "payment_succeeded", "payment_failed", "payment_refunded",
        "card_added", "card_updated", "card_deleted", "card_expired",
        "plan_changed",
    }
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.json_dir = data_dir / "json"
        
        self._events: list[dict] = []
        self._customers: list[dict] = []
        self._subscriptions: list[dict] = []
        
    def load_data(self) -> bool:
        """Load required JSON data files."""
        logger.info("Loading data from %s", self.json_dir)
        
        self._events = self._load_json("events.json")
        self._customers = self._load_json("customers.json")
        self._subscriptions = self._load_json("subscriptions.json")
        
        logger.info(
            "Loaded: %d events, %d customers, %d subscriptions",
            len(self._events), len(self._customers), len(self._subscriptions)
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
    
    def analyze(self) -> EventSequenceAnalysisResult:
        """Run event sequence analysis."""
        logger.info("Starting event sequence analysis")
        
        # Group events by customer
        events_by_customer: dict[str, list[dict]] = defaultdict(list)
        event_type_counts: Counter = Counter()
        
        for event in self._events:
            event_type = event.get("event_type", "unknown")
            event_type_counts[event_type] += 1
            
            # Extract customer ID from event content
            content = event.get("content", {})
            customer_id = None
            
            # Try to find customer ID in content
            for key in ["customer", "subscription", "invoice", "transaction"]:
                if key in content:
                    customer_id = content[key].get("customer_id") or content[key].get("id")
                    if key == "customer":
                        customer_id = content[key].get("id")
                    break
            
            if customer_id:
                events_by_customer[customer_id].append({
                    "type": event_type,
                    "occurred_at": event.get("occurred_at"),
                    "source": event.get("source"),
                })
        
        # Sort events by time for each customer
        for cid in events_by_customer:
            events_by_customer[cid].sort(key=lambda x: x.get("occurred_at", 0))
        
        # Build subscription status map
        customer_status: dict[str, str] = {}
        for sub in self._subscriptions:
            cid = sub.get("customer_id")
            status = sub.get("status", "unknown")
            if cid:
                # Use most recent/important status
                if status == "active" or cid not in customer_status:
                    customer_status[cid] = status
        
        # Extract sequences and find patterns
        sequences: list[tuple[list[str], str]] = []  # (event_types, outcome)
        
        for cid, events in events_by_customer.items():
            if len(events) < 2:
                continue
            
            event_types = [e["type"] for e in events]
            outcome = customer_status.get(cid, "unknown")
            sequences.append((event_types, outcome))
        
        # Find common 2-event and 3-event patterns
        pattern_2_counts: Counter = Counter()
        pattern_3_counts: Counter = Counter()
        
        for event_types, _ in sequences:
            # 2-event patterns
            for i in range(len(event_types) - 1):
                pattern = (event_types[i], event_types[i + 1])
                pattern_2_counts[pattern] += 1
            
            # 3-event patterns
            for i in range(len(event_types) - 2):
                pattern = (event_types[i], event_types[i + 1], event_types[i + 2])
                pattern_3_counts[pattern] += 1
        
        # Build common patterns
        total_customers = len(events_by_customer)
        common_patterns: list[EventSequencePattern] = []
        
        for pattern, count in pattern_2_counts.most_common(15):
            if count < 3:
                continue
            common_patterns.append(EventSequencePattern(
                pattern=list(pattern),
                frequency=count,
                pct_of_customers=count / total_customers if total_customers > 0 else 0,
                avg_time_between_events=0,  # Would need to calculate
                outcome=None,
            ))
        
        # Find pre-churn patterns
        pre_churn_patterns = self._find_outcome_patterns(sequences, "cancelled")
        pre_upgrade_patterns = self._find_outcome_patterns(sequences, "active", look_for="subscription_changed")
        
        # Build sample journeys
        sample_journeys = self._build_sample_journeys(
            events_by_customer, customer_status
        )
        
        # Generate insights
        insights = self._generate_insights(
            event_type_counts, common_patterns, pre_churn_patterns
        )
        
        result = EventSequenceAnalysisResult(
            analysis_timestamp=datetime.now(timezone.utc),
            total_events=len(self._events),
            total_customers_with_events=total_customers,
            event_type_distribution=dict(event_type_counts.most_common(30)),
            common_patterns=common_patterns,
            pre_churn_patterns=pre_churn_patterns,
            pre_upgrade_patterns=pre_upgrade_patterns,
            sample_journeys=sample_journeys,
            insights=insights,
        )
        
        logger.info(
            "Event sequence analysis complete: %d events, %d patterns found",
            len(self._events), len(common_patterns)
        )
        
        return result
    
    def _find_outcome_patterns(
        self, sequences: list[tuple[list[str], str]], outcome: str,
        look_for: str | None = None
    ) -> list[EventSequencePattern]:
        """Find patterns that precede a specific outcome."""
        patterns: Counter = Counter()
        
        for event_types, seq_outcome in sequences:
            if seq_outcome != outcome:
                continue
            
            # Look at last 5 events before outcome
            if len(event_types) >= 3:
                last_events = tuple(event_types[-5:])
                patterns[last_events] += 1
                
                # Also 3-event patterns
                for i in range(len(event_types) - 2):
                    pattern = (event_types[i], event_types[i + 1], event_types[i + 2])
                    if look_for is None or look_for in pattern:
                        patterns[pattern] += 1
        
        result = []
        total = sum(1 for _, o in sequences if o == outcome)
        
        for pattern, count in patterns.most_common(10):
            if count < 2:
                continue
            result.append(EventSequencePattern(
                pattern=list(pattern),
                frequency=count,
                pct_of_customers=count / total if total > 0 else 0,
                avg_time_between_events=0,
                outcome=outcome,
            ))
        
        return result
    
    def _build_sample_journeys(
        self,
        events_by_customer: dict[str, list[dict]],
        customer_status: dict[str, str],
    ) -> list[CustomerJourney]:
        """Build sample customer journeys."""
        customer_map = {c.get("id"): c for c in self._customers}
        journeys = []
        
        # Get customers with most events
        sorted_customers = sorted(
            events_by_customer.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for cid, events in sorted_customers[:10]:
            customer = customer_map.get(cid, {})
            
            if not events:
                continue
            
            first_event = parse_timestamp(events[0].get("occurred_at"))
            last_event = parse_timestamp(events[-1].get("occurred_at"))
            
            journey_days = 0
            if first_event and last_event:
                journey_days = (last_event - first_event).days
            
            # Extract key milestones
            milestones = []
            milestone_types = {
                "customer_created", "subscription_created", "subscription_activated",
                "payment_succeeded", "subscription_cancelled",
            }
            for e in events:
                if e["type"] in milestone_types and e["type"] not in milestones:
                    milestones.append(e["type"])
            
            journeys.append(CustomerJourney(
                customer_id=cid,
                email=customer.get("email"),
                events=[
                    {
                        "type": e["type"],
                        "date": dt.isoformat() if (dt := parse_timestamp(e.get("occurred_at"))) else None,
                    }
                    for e in events
                ],
                journey_length_days=journey_days,
                key_milestones=milestones,
                current_status=customer_status.get(cid, "unknown"),
            ))
        
        return journeys
    
    def _generate_insights(
        self,
        event_counts: dict[str, int],
        common_patterns: list[EventSequencePattern],
        pre_churn: list[EventSequencePattern],
    ) -> list[str]:
        """Generate insights from event analysis."""
        insights = []
        
        # Top event types
        top_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_events:
            insights.append(f"Most common events: {', '.join(e[0] for e in top_events)}")
        
        # Payment failure patterns
        failure_patterns = [
            p for p in common_patterns 
            if "payment_failed" in p.pattern
        ]
        if failure_patterns:
            insights.append(f"Found {len(failure_patterns)} patterns involving payment failures")
        
        # Pre-churn warning
        if pre_churn:
            top_churn = pre_churn[0]
            insights.append(
                f"Common pre-churn sequence: {' -> '.join(top_churn.pattern[:3])}"
            )
        
        return insights
    
    def save_results(self, result: EventSequenceAnalysisResult, output_dir: Path) -> Path:
        """Save analysis results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "event_sequence_analysis.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info("Event sequence analysis saved to %s", output_path)
        return output_path
