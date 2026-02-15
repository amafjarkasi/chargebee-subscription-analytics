"""Revenue forecasting using time series analysis.

Predicts MRR/ARR trends using statistical methods.
Supports multiple forecasting approaches when dependencies are available.
"""

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .utils import calculate_mrr, days_ago, load_json_file, parse_timestamp

logger = logging.getLogger("chargebee_mapper.revenue_forecaster")


@dataclass
class MonthlyRevenue:
    """Revenue data for a single month."""
    year: int
    month: int
    mrr: float  # Represents "Billed Revenue" in this context
    invoice_count: int
    new_customers: int
    churned_customers: int
    
    @property
    def period(self) -> str:
        return f"{self.year}-{self.month:02d}"


@dataclass
class ForecastPoint:
    """A single forecast data point."""
    period: str
    predicted_mrr: float
    lower_bound: float
    upper_bound: float
    confidence: float = 0.95


@dataclass
class RevenueForecastResult:
    """Complete revenue forecast results."""
    analysis_timestamp: datetime
    historical_months: int
    forecast_months: int
    current_mrr: float
    current_arr: float
    historical_data: list[MonthlyRevenue]
    forecast_data: list[ForecastPoint]
    growth_rate_monthly: float
    growth_rate_annual: float
    seasonality_detected: bool
    trend_direction: str  # "growing", "stable", "declining"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "historical_months": self.historical_months,
            "forecast_months": self.forecast_months,
            "current_mrr": round(self.current_mrr, 2),
            "current_arr": round(self.current_arr, 2),
            "growth_rate_monthly": round(self.growth_rate_monthly, 4),
            "growth_rate_annual": round(self.growth_rate_annual, 4),
            "seasonality_detected": self.seasonality_detected,
            "trend_direction": self.trend_direction,
            "historical_data": [
                {
                    "period": h.period,
                    "mrr": round(h.mrr, 2),
                    "invoice_count": h.invoice_count,
                    "new_customers": h.new_customers,
                    "churned_customers": h.churned_customers,
                }
                for h in self.historical_data
            ],
            "forecast_data": [
                {
                    "period": f.period,
                    "predicted_mrr": round(f.predicted_mrr, 2),
                    "lower_bound": round(f.lower_bound, 2),
                    "upper_bound": round(f.upper_bound, 2),
                    "confidence": f.confidence,
                }
                for f in self.forecast_data
            ],
        }


class RevenueForecaster:
    """Forecasts MRR/ARR using time series analysis."""
    
    def __init__(self, data_dir: Path, forecast_months: int = 6):
        self.data_dir = data_dir
        self.forecast_months = forecast_months
        self.json_dir = data_dir / "json"
        
        self._invoices: list[dict] = []
        self._subscriptions: list[dict] = []
        self._customers: list[dict] = []
        self._loaded = False
        
    def load_data(self) -> bool:
        """Load required JSON data files."""
        logger.info("Loading data from %s", self.json_dir)
        
        # We need invoices at minimum
        if not (self.json_dir / "invoices.json").exists():
            logger.error("No invoice data found (required for historical revenue)")
            return False

        self._invoices = load_json_file(self.json_dir / "invoices.json")
        self._subscriptions = load_json_file(self.json_dir / "subscriptions.json")
        self._customers = load_json_file(self.json_dir / "customers.json")

        logger.info(
            "Loaded: %d invoices, %d subscriptions, %d customers",
            len(self._invoices), len(self._subscriptions), len(self._customers)
        )
        self._loaded = True
        return True
    
    def analyze(self) -> RevenueForecastResult:
        """Run revenue forecasting analysis."""
        if not self._loaded:
             if not self.load_data():
                 raise RuntimeError("Cannot proceed without data")

        logger.info("Starting revenue forecast analysis")
        
        # Build monthly revenue data
        monthly_data = self._build_monthly_revenue()
        
        if len(monthly_data) < 3:
            logger.warning("Insufficient historical data for reliable forecasting (need 3+ months)")
        
        # Calculate current MRR from active subscriptions
        current_mrr = self._calculate_current_mrr()
        
        # Calculate growth rates
        growth_monthly, growth_annual = self._calculate_growth_rates(monthly_data)
        
        # Detect trend
        trend = self._detect_trend(monthly_data)
        
        # Check for seasonality
        seasonality = self._detect_seasonality(monthly_data)
        
        # Generate forecast
        forecast = self._generate_forecast(monthly_data, growth_monthly, current_mrr)
        
        result = RevenueForecastResult(
            analysis_timestamp=datetime.now(timezone.utc),
            historical_months=len(monthly_data),
            forecast_months=self.forecast_months,
            current_mrr=current_mrr,
            current_arr=current_mrr * 12,
            historical_data=monthly_data,
            forecast_data=forecast,
            growth_rate_monthly=growth_monthly,
            growth_rate_annual=growth_annual,
            seasonality_detected=seasonality,
            trend_direction=trend,
        )
        
        logger.info(
            "Forecast complete: MRR=$%.2f, trend=%s, growth=%.1f%%/month",
            current_mrr, trend, growth_monthly * 100
        )
        
        return result
    
    def _build_monthly_revenue(self) -> list[MonthlyRevenue]:
        """Aggregate invoice data into monthly revenue."""
        monthly: dict[str, dict] = defaultdict(lambda: {
            "mrr": 0.0, "invoice_count": 0, "customers": set()
        })
        
        for inv in self._invoices:
            # Include paid and posted (posted = finalized but potentially unpaid yet, still counts as revenue usually)
            if inv.get("status") not in ("paid", "posted"):
                continue
            
            date = parse_timestamp(inv.get("date"))
            if not date:
                continue
            
            key = f"{date.year}-{date.month:02d}"

            # Use sub_total (pre-tax) or total (post-tax). Usually revenue means Net Revenue (pre-tax).
            # Chargebee sub_total excludes tax.
            amount_cents = inv.get("sub_total", 0)
            if amount_cents is None:
                 amount_cents = inv.get("total", 0) # Fallback

            amount = amount_cents / 100.0  # Convert cents to dollars
            
            monthly[key]["mrr"] += amount
            monthly[key]["invoice_count"] += 1
            if inv.get("customer_id"):
                monthly[key]["customers"].add(inv["customer_id"])
        
        # Build customer churn data
        customer_first_month: dict[str, str] = {}
        customer_last_month: dict[str, str] = {}
        
        for cust in self._customers:
            cid = cust.get("id")
            created = parse_timestamp(cust.get("created_at"))
            if cid and created:
                key = f"{created.year}-{created.month:02d}"
                customer_first_month[cid] = key
        
        for sub in self._subscriptions:
            cid = sub.get("customer_id")
            if sub.get("status") == "cancelled" and sub.get("cancelled_at"):
                cancelled = parse_timestamp(sub.get("cancelled_at"))
                if cancelled and cid:
                    key = f"{cancelled.year}-{cancelled.month:02d}"
                    customer_last_month[cid] = key
        
        # Convert to sorted list
        result = []
        for period in sorted(monthly.keys()):
            data = monthly[period]
            year, month = map(int, period.split("-"))
            
            new_customers = sum(1 for c, p in customer_first_month.items() if p == period)
            churned = sum(1 for c, p in customer_last_month.items() if p == period)
            
            result.append(MonthlyRevenue(
                year=year,
                month=month,
                mrr=data["mrr"],
                invoice_count=data["invoice_count"],
                new_customers=new_customers,
                churned_customers=churned,
            ))
        
        return result
    
    def _calculate_current_mrr(self) -> float:
        """Calculate current MRR from active subscriptions."""
        # Use utility from utils.py logic?
        # Actually utils.calculate_mrr logic is identical to what we want.
        # But we need to make sure self._subscriptions is populated.
        if not self._subscriptions:
            return 0.0
        return calculate_mrr(self._subscriptions)
    
    def _calculate_growth_rates(
        self, monthly_data: list[MonthlyRevenue]
    ) -> tuple[float, float]:
        """Calculate monthly and annual growth rates."""
        if len(monthly_data) < 2:
            return 0.0, 0.0
        
        # Use last 6 months or all available
        recent = monthly_data[-6:] if len(monthly_data) >= 6 else monthly_data
        
        if len(recent) < 2:
             return 0.0, 0.0

        start_mrr = recent[0].mrr
        end_mrr = recent[-1].mrr
        n_months = len(recent) - 1
        
        if start_mrr <= 0:
            # Cannot calculate growth from 0 or negative base
            return 0.0, 0.0
        
        if end_mrr < 0:
             end_mrr = 0

        # Compound monthly growth rate
        try:
            monthly_rate = (end_mrr / start_mrr) ** (1 / n_months) - 1
        except ValueError:
            # Handle negative root issues if they arise
            return 0.0, 0.0

        annual_rate = (1 + monthly_rate) ** 12 - 1
        
        return monthly_rate, annual_rate
    
    def _detect_trend(self, monthly_data: list[MonthlyRevenue]) -> str:
        """Detect overall revenue trend."""
        if len(monthly_data) < 3:
            return "stable"
        
        recent = monthly_data[-6:] if len(monthly_data) >= 6 else monthly_data
        mrr_values = [m.mrr for m in recent]
        
        # Simple linear regression slope
        n = len(mrr_values)
        if n == 0:
             return "stable"

        x_mean = (n - 1) / 2
        y_mean = sum(mrr_values) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(mrr_values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # Normalize slope by mean revenue
        if y_mean == 0:
            return "stable"
        
        normalized_slope = slope / y_mean
        
        if normalized_slope > 0.02:
            return "growing"
        elif normalized_slope < -0.02:
            return "declining"
        return "stable"
    
    def _detect_seasonality(self, monthly_data: list[MonthlyRevenue]) -> bool:
        """Simple seasonality detection."""
        if len(monthly_data) < 24: # Need at least 2 full years for real seasonality check
            return False
        
        # Compare same months across years
        by_month: dict[int, list[float]] = defaultdict(list)
        for m in monthly_data:
            by_month[m.month].append(m.mrr)
        
        # If we have consistent patterns (e.g., specific months always higher than average)
        # Calculate variance of means of months vs mean of all data?
        # Simplified: Check if at least 2 years exist
        years = set(m.year for m in monthly_data)
        if len(years) < 2:
            return False

        # Very naive check: Do we have enough data points for monthly comparison?
        months_with_history = sum(1 for v in by_month.values() if len(v) >= 2)
        return months_with_history >= 6
    
    def _generate_forecast(
        self, monthly_data: list[MonthlyRevenue], growth_rate: float, current_mrr: float
    ) -> list[ForecastPoint]:
        """Generate future revenue forecasts."""
        
        # Base the forecast on the calculated Current MRR from active subscriptions
        # rather than the last invoice month, as invoice timing can be irregular.
        # However, if current_mrr is 0 (no subs loaded), fallback to last month invoice.
        start_mrr = current_mrr
        if start_mrr == 0 and monthly_data:
            start_mrr = monthly_data[-1].mrr

        if start_mrr == 0:
            return []
        
        # Calculate standard deviation for confidence intervals based on historical volatility
        if len(monthly_data) >= 3:
            mrr_values = [m.mrr for m in monthly_data[-12:]]
            mean_mrr = sum(mrr_values) / len(mrr_values)
            # Variance calculation handling empty list checked by len check
            variance = sum((x - mean_mrr) ** 2 for x in mrr_values) / len(mrr_values)
            std_dev = math.sqrt(variance)
        else:
            std_dev = start_mrr * 0.1  # Default 10% uncertainty
        
        forecasts = []

        # Determine start date (next month after now)
        now = datetime.now(timezone.utc)
        current_year = now.year
        current_month = now.month
        
        for i in range(1, self.forecast_months + 1):
            # Move to next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
            
            # Project MRR with growth rate
            projected_mrr = start_mrr * ((1 + growth_rate) ** i)
            
            # Widen confidence interval over time
            # Uncertainty grows with sqrt(time) standard assumption in random walk
            uncertainty = std_dev * math.sqrt(i) * 1.96  # 95% CI (approx)
            
            forecasts.append(ForecastPoint(
                period=f"{current_year}-{current_month:02d}",
                predicted_mrr=projected_mrr,
                lower_bound=max(0, projected_mrr - uncertainty),
                upper_bound=projected_mrr + uncertainty,
                confidence=0.95,
            ))
        
        return forecasts
    
    def save_results(self, result: RevenueForecastResult, output_dir: Path) -> Path:
        """Save forecast results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "revenue_forecast.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info("Revenue forecast saved to %s", output_path)
        return output_path
