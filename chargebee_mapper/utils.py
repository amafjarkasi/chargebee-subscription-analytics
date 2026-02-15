import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("chargebee_mapper.utils")

def parse_timestamp(ts: int | None) -> datetime | None:
    """Convert Unix timestamp to UTC datetime."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, OSError):
        return None

def days_ago(dt: datetime | None) -> int | None:
    """Calculate days since a given datetime (positive for past, negative for future)."""
    if dt is None:
        return None
    now = datetime.now(timezone.utc)
    return (now - dt).days

def months_ago(dt: datetime | None) -> int | None:
    """Calculate months since a given datetime."""
    if dt is None:
        return None
    now = datetime.now(timezone.utc)
    return (now.year - dt.year) * 12 + (now.month - dt.month)

def calculate_mrr(active_subscriptions: list[dict[str, Any]]) -> float:
    """Calculate total MRR from active subscriptions (in dollars)."""
    total = 0.0
    for sub in active_subscriptions:
        if sub.get("status") != "active":
            continue

        # Try to get MRR from subscription data
        mrr = sub.get("mrr", 0)
        if mrr:
            total += mrr / 100  # Chargebee stores amounts in cents
        else:
            # Fallback: sum up item prices
            items = sub.get("subscription_items", [])
            for item in items:
                # amount (for addons) or unit_price (for plans)
                amount = item.get("amount", 0) or item.get("unit_price", 0)
                quantity = item.get("quantity", 1)

                # Check billing period? Assuming monthly for fallback simplicity
                # Ideally check plan interval, but complex without plan data join
                total += (amount * quantity) / 100
    return total

def load_json_file(file_path: Path) -> list[dict[str, Any]]:
    """Load a JSON file, returning empty list if not found or invalid."""
    if not file_path.exists():
        logger.debug("File not found: %s", file_path)
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load %s: %s", file_path, e)
        return []
