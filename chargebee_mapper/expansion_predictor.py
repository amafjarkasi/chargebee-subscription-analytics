"""Expansion revenue prediction - upsell/cross-sell opportunities.

Identifies customers likely to upgrade and recommends products.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("chargebee_mapper.expansion_predictor")


@dataclass
class ExpansionOpportunity:
    """Expansion opportunity for a single customer."""
    customer_id: str
    email: str | None
    company: str | None
    expansion_score: int  # 0-100
    expansion_likelihood: str  # "high", "medium", "low"
    current_mrr: float
    potential_uplift: float
    current_plan: str | None
    recommended_plan: str | None
    expansion_signals: list[str]
    recommended_actions: list[str]
    tenure_months: int
    growth_trajectory: str  # "growing", "stable", "declining"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "email": self.email,
            "company": self.company,
            "expansion_score": self.expansion_score,
            "expansion_likelihood": self.expansion_likelihood,
            "current_mrr": round(self.current_mrr, 2),
            "potential_uplift": round(self.potential_uplift, 2),
            "current_plan": self.current_plan,
            "recommended_plan": self.recommended_plan,
            "expansion_signals": self.expansion_signals,
            "recommended_actions": self.recommended_actions,
            "tenure_months": self.tenure_months,
            "growth_trajectory": self.growth_trajectory,
        }


@dataclass
class ExpansionAnalysisResult:
    """Complete expansion revenue analysis results."""
    analysis_timestamp: datetime
    total_customers_analyzed: int
    expansion_candidates: int
    total_potential_uplift: float
    likelihood_distribution: dict[str, int]
    plan_upgrade_paths: dict[str, list[str]]
    top_opportunities: list[ExpansionOpportunity]
    all_opportunities: list[ExpansionOpportunity]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_customers_analyzed": self.total_customers_analyzed,
            "expansion_candidates": self.expansion_candidates,
            "total_potential_uplift": round(self.total_potential_uplift, 2),
            "likelihood_distribution": self.likelihood_distribution,
            "plan_upgrade_paths": self.plan_upgrade_paths,
            "top_opportunities": [o.to_dict() for o in self.top_opportunities[:20]],
            "all_opportunities": [o.to_dict() for o in self.all_opportunities],
        }


def _parse_timestamp(ts: int | None) -> datetime | None:
    """Convert Unix timestamp to datetime."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, OSError):
        return None


def _months_since(dt: datetime | None) -> int:
    """Calculate months since a datetime."""
    if dt is None:
        return 0
    now = datetime.now(timezone.utc)
    return max(1, (now.year - dt.year) * 12 + (now.month - dt.month))


def _get_likelihood(score: int) -> str:
    """Convert score to likelihood level."""
    if score >= 70:
        return "high"
    elif score >= 40:
        return "medium"
    return "low"


class ExpansionPredictor:
    """Predicts expansion revenue opportunities."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.json_dir = data_dir / "json"
        
        self._customers: list[dict] = []
        self._subscriptions: list[dict] = []
        self._invoices: list[dict] = []
        self._items: list[dict] = []
        self._item_prices: list[dict] = []
        self._events: list[dict] = []
        
    def load_data(self) -> bool:
        """Load required JSON data files."""
        logger.info("Loading data from %s", self.json_dir)
        
        self._customers = self._load_json("customers.json")
        self._subscriptions = self._load_json("subscriptions.json")
        self._invoices = self._load_json("invoices.json")
        self._items = self._load_json("items.json")
        self._item_prices = self._load_json("item_prices.json")
        self._events = self._load_json("events.json")
        
        if not self._customers or not self._subscriptions:
            logger.error("Missing required data (customers or subscriptions)")
            return False
            
        logger.info(
            "Loaded: %d customers, %d subscriptions, %d item prices",
            len(self._customers), len(self._subscriptions), len(self._item_prices)
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
    
    def analyze(self) -> ExpansionAnalysisResult:
        """Run expansion prediction analysis."""
        logger.info("Starting expansion prediction analysis")
        
        # Build price hierarchy (lower to higher tiers)
        price_hierarchy = self._build_price_hierarchy()
        
        # Index data
        subs_by_customer: dict[str, list[dict]] = defaultdict(list)
        for sub in self._subscriptions:
            cid = sub.get("customer_id")
            if cid:
                subs_by_customer[cid].append(sub)
        
        invoices_by_customer: dict[str, list[dict]] = defaultdict(list)
        for inv in self._invoices:
            cid = inv.get("customer_id")
            if cid:
                invoices_by_customer[cid].append(inv)
        
        # Analyze each customer
        opportunities: list[ExpansionOpportunity] = []
        
        for customer in self._customers:
            if customer.get("deleted"):
                continue
            
            cid = customer.get("id")
            subs = subs_by_customer.get(cid, [])
            invoices = invoices_by_customer.get(cid, [])
            
            # Only analyze customers with active subscriptions
            active_subs = [s for s in subs if s.get("status") == "active"]
            if not active_subs:
                continue
            
            opp = self._analyze_customer(customer, active_subs, invoices, price_hierarchy)
            if opp:
                opportunities.append(opp)
        
        # Sort by expansion score
        opportunities.sort(key=lambda x: x.expansion_score, reverse=True)
        
        # Calculate aggregates
        likelihood_dist = {"high": 0, "medium": 0, "low": 0}
        total_uplift = 0.0
        candidates = 0
        
        for opp in opportunities:
            likelihood_dist[opp.expansion_likelihood] += 1
            total_uplift += opp.potential_uplift
            if opp.expansion_likelihood in ("high", "medium"):
                candidates += 1
        
        result = ExpansionAnalysisResult(
            analysis_timestamp=datetime.now(timezone.utc),
            total_customers_analyzed=len(opportunities),
            expansion_candidates=candidates,
            total_potential_uplift=total_uplift,
            likelihood_distribution=likelihood_dist,
            plan_upgrade_paths=price_hierarchy,
            top_opportunities=opportunities[:20],
            all_opportunities=opportunities,
        )
        
        logger.info(
            "Expansion analysis complete: %d customers, %d candidates, $%.2f potential uplift",
            len(opportunities), candidates, total_uplift
        )
        
        return result
    
    def _build_price_hierarchy(self) -> dict[str, list[str]]:
        """Build upgrade paths from lower to higher priced plans."""
        # Group prices by item
        prices_by_item: dict[str, list[dict]] = defaultdict(list)
        for price in self._item_prices:
            item_id = price.get("item_id")
            if item_id and price.get("status") == "active":
                prices_by_item[item_id].append(price)
        
        # Build upgrade paths (simple: sort by price)
        upgrade_paths: dict[str, list[str]] = {}
        
        for item_id, prices in prices_by_item.items():
            sorted_prices = sorted(prices, key=lambda x: x.get("price", 0) or 0)
            price_ids = [p.get("id") for p in sorted_prices if p.get("id")]
            
            # Map each price to higher-tier options
            for i, price_id in enumerate(price_ids[:-1]):
                upgrade_paths[price_id] = price_ids[i + 1:]
        
        return upgrade_paths
    
    def _analyze_customer(
        self,
        customer: dict,
        subscriptions: list[dict],
        invoices: list[dict],
        price_hierarchy: dict[str, list[str]],
    ) -> ExpansionOpportunity | None:
        """Analyze expansion opportunity for a single customer."""
        customer_id = customer.get("id", "unknown")
        expansion_signals = []
        recommended_actions = []
        score = 0
        
        # Get current subscription info
        primary_sub = subscriptions[0]
        current_mrr = sum((s.get("mrr", 0) or 0) / 100 for s in subscriptions)
        
        # Get current plan
        current_plan = None
        sub_items = primary_sub.get("subscription_items", [])
        if sub_items:
            current_plan = sub_items[0].get("item_price_id")
        
        # Calculate tenure
        created_at = _parse_timestamp(customer.get("created_at"))
        tenure_months = _months_since(created_at)
        
        # Analyze growth trajectory from invoice history
        growth = self._calculate_growth_trajectory(invoices)
        
        # === EXPANSION SIGNALS ===
        
        # 1. Growing revenue trajectory
        if growth == "growing":
            score += 25
            expansion_signals.append("Revenue growing month-over-month")
            recommended_actions.append("Proactively discuss expansion options")
        
        # 2. Long tenure (established relationship)
        if tenure_months >= 12:
            score += 15
            expansion_signals.append(f"Established customer ({tenure_months} months)")
        elif tenure_months >= 6:
            score += 10
            expansion_signals.append(f"Maturing relationship ({tenure_months} months)")
        
        # 3. High current MRR (valuable customer)
        if current_mrr >= 1000:
            score += 15
            expansion_signals.append(f"High-value customer (${current_mrr:.0f} MRR)")
        elif current_mrr >= 500:
            score += 10
            expansion_signals.append(f"Significant revenue (${current_mrr:.0f} MRR)")
        
        # 4. Multiple subscriptions (engaged)
        if len(subscriptions) > 1:
            score += 15
            expansion_signals.append(f"Multiple subscriptions ({len(subscriptions)})")
            recommended_actions.append("Consider bundle pricing")
        
        # 5. On lower-tier plan with upgrade path available
        recommended_plan = None
        potential_uplift = 0.0
        
        if current_plan and current_plan in price_hierarchy:
            upgrades = price_hierarchy[current_plan]
            if upgrades:
                recommended_plan = upgrades[0]  # Next tier up
                
                # Estimate uplift (would need price lookup in production)
                potential_uplift = current_mrr * 0.5  # Assume 50% uplift potential
                
                score += 20
                expansion_signals.append("Upgrade path available")
                recommended_actions.append(f"Present {recommended_plan} upgrade option")
        
        # 6. Payment reliability (no failures = good candidate)
        paid_invoices = [i for i in invoices if i.get("status") == "paid"]
        if len(paid_invoices) >= 6:
            score += 10
            expansion_signals.append("Strong payment history")
        
        # Determine likelihood
        likelihood = _get_likelihood(score)
        
        # Add general recommendations based on likelihood
        if likelihood == "high":
            recommended_actions.append("Schedule expansion conversation")
            recommended_actions.append("Prepare ROI analysis for upgrade")
        elif likelihood == "medium":
            recommended_actions.append("Send case study/success story")
            recommended_actions.append("Monitor for usage growth signals")
        
        return ExpansionOpportunity(
            customer_id=customer_id,
            email=customer.get("email"),
            company=customer.get("company"),
            expansion_score=min(100, score),
            expansion_likelihood=likelihood,
            current_mrr=current_mrr,
            potential_uplift=potential_uplift,
            current_plan=current_plan,
            recommended_plan=recommended_plan,
            expansion_signals=expansion_signals,
            recommended_actions=recommended_actions,
            tenure_months=tenure_months,
            growth_trajectory=growth,
        )
    
    def _calculate_growth_trajectory(self, invoices: list[dict]) -> str:
        """Calculate revenue growth trajectory from invoices."""
        if len(invoices) < 3:
            return "stable"
        
        # Group invoices by month
        monthly_revenue: dict[str, float] = defaultdict(float)
        for inv in invoices:
            if inv.get("status") not in ("paid", "posted"):
                continue
            date = _parse_timestamp(inv.get("date"))
            if date:
                key = f"{date.year}-{date.month:02d}"
                monthly_revenue[key] += (inv.get("total", 0) or 0) / 100
        
        if len(monthly_revenue) < 3:
            return "stable"
        
        # Get last 6 months
        sorted_months = sorted(monthly_revenue.keys())[-6:]
        values = [monthly_revenue[m] for m in sorted_months]
        
        # Simple trend detection
        if len(values) >= 3:
            first_half_avg = sum(values[:len(values)//2]) / (len(values)//2)
            second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
            
            if first_half_avg == 0:
                return "stable"
            
            growth_rate = (second_half_avg - first_half_avg) / first_half_avg
            
            if growth_rate > 0.1:
                return "growing"
            elif growth_rate < -0.1:
                return "declining"
        
        return "stable"
    
    def save_results(self, result: ExpansionAnalysisResult, output_dir: Path) -> Path:
        """Save analysis results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "expansion_analysis.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info("Expansion analysis saved to %s", output_path)
        return output_path
