"""Data deduplication detection and reporting.

This module identifies potential duplicate records in the fetched data,
particularly focusing on customer duplicates which can cause inaccurate
analysis results.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logger = logging.getLogger("chargebee_mapper.deduplication")


@dataclass
class DuplicateGroup:
    """A group of potentially duplicate records."""
    
    match_type: str  # "exact_email", "similar_company", "same_phone", etc.
    match_key: str   # The matching value (email, normalized company name, etc.)
    confidence: float  # 0.0 to 1.0
    records: list[dict[str, Any]] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.records)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "match_type": self.match_type,
            "match_key": self.match_key,
            "confidence": self.confidence,
            "count": self.count,
            "records": [
                {
                    "id": r.get("id"),
                    "email": r.get("email"),
                    "company": r.get("company"),
                    "first_name": r.get("first_name"),
                    "last_name": r.get("last_name"),
                }
                for r in self.records
            ],
        }


@dataclass
class DeduplicationReport:
    """Report of detected duplicates."""
    
    analysis_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_records: int = 0
    unique_records: int = 0
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    
    @property
    def total_duplicates(self) -> int:
        return sum(g.count - 1 for g in self.duplicate_groups)
    
    @property
    def duplication_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.total_duplicates / self.total_records
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_records": self.total_records,
            "unique_records": self.unique_records,
            "total_duplicates": self.total_duplicates,
            "duplication_rate": self.duplication_rate,
            "duplicate_groups_by_type": self._group_by_type(),
            "duplicate_groups": [g.to_dict() for g in self.duplicate_groups],
        }
    
    def _group_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for group in self.duplicate_groups:
            counts[group.match_type] += 1
        return dict(counts)


class DeduplicationAnalyzer:
    """Analyzes data for potential duplicates."""
    
    # Minimum similarity threshold for company name matching
    COMPANY_SIMILARITY_THRESHOLD = 0.85
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.json_dir = data_dir / "json"
        self.customers: list[dict[str, Any]] = []
    
    def load_data(self) -> bool:
        """Load customer data for analysis."""
        customers_file = self.json_dir / "customers.json"
        if not customers_file.exists():
            logger.error("Customers file not found: %s", customers_file)
            return False
        
        try:
            with open(customers_file, "r", encoding="utf-8") as f:
                self.customers = json.load(f)
            logger.info("Loaded %d customers for deduplication analysis", len(self.customers))
            return True
        except Exception as e:
            logger.error("Failed to load customers: %s", e)
            return False
    
    def analyze(self) -> DeduplicationReport:
        """Run deduplication analysis."""
        report = DeduplicationReport(total_records=len(self.customers))
        
        # Find duplicates by different criteria
        email_duplicates = self._find_email_duplicates()
        company_duplicates = self._find_company_duplicates()
        
        # Combine results
        report.duplicate_groups.extend(email_duplicates)
        report.duplicate_groups.extend(company_duplicates)
        
        # Calculate unique records (approximation)
        duplicate_ids = set()
        for group in report.duplicate_groups:
            # All but the first record in each group are "duplicates"
            for record in group.records[1:]:
                duplicate_ids.add(record.get("id"))
        
        report.unique_records = report.total_records - len(duplicate_ids)
        
        logger.info(
            "Deduplication analysis complete: %d groups, %d potential duplicates",
            len(report.duplicate_groups),
            report.total_duplicates,
        )
        
        return report
    
    def _normalize_email(self, email: str | None) -> str | None:
        """Normalize an email address for comparison."""
        if not email:
            return None
        
        # Lowercase and strip whitespace
        email = email.lower().strip()
        
        # Remove "DONOTEMAIL", "NO_EMAILS", etc. markers
        email = re.sub(r"donotemail[-_]?", "", email, flags=re.IGNORECASE)
        email = re.sub(r"no_?emails?[-_]?", "", email, flags=re.IGNORECASE)
        email = re.sub(r"[-_]?donotemail", "", email, flags=re.IGNORECASE)
        
        # Validate email format
        if "@" not in email or "." not in email.split("@")[-1]:
            return None
        
        return email
    
    def _normalize_company(self, company: str | None) -> str:
        """Normalize a company name for comparison."""
        if not company:
            return ""
        
        # Lowercase and strip
        company = company.lower().strip()
        
        # Remove parenthetical content first (e.g., "(o/a Other Name)")
        company = re.sub(r"\s*\(.*?\)", "", company)
        
        # Remove "operating as" suffixes
        company = re.sub(r"\s*(o/a|dba|d/b/a|c/o)\s+.*$", "", company, flags=re.IGNORECASE)
        
        # Remove common suffixes (must be separate patterns for proper matching)
        company = re.sub(r",?\s*inc\.?$", "", company, flags=re.IGNORECASE)
        company = re.sub(r",?\s*llc\.?$", "", company, flags=re.IGNORECASE)
        company = re.sub(r",?\s*ltd\.?$", "", company, flags=re.IGNORECASE)
        company = re.sub(r",?\s*corp\.?$", "", company, flags=re.IGNORECASE)
        company = re.sub(r",?\s*corporation$", "", company, flags=re.IGNORECASE)
        company = re.sub(r",?\s*company$", "", company, flags=re.IGNORECASE)
        company = re.sub(r",?\s*co\.?$", "", company, flags=re.IGNORECASE)
        
        # Remove special characters
        company = re.sub(r"[^\w\s]", "", company)
        
        # Collapse whitespace
        company = re.sub(r"\s+", " ", company).strip()
        
        return company
    
    def _find_email_duplicates(self) -> list[DuplicateGroup]:
        """Find customers with the same email address."""
        email_to_customers: dict[str, list[dict]] = defaultdict(list)
        
        for customer in self.customers:
            email = self._normalize_email(customer.get("email"))
            if email:
                email_to_customers[email].append(customer)
        
        # Find groups with more than one customer
        groups = []
        for email, customers in email_to_customers.items():
            if len(customers) > 1:
                groups.append(DuplicateGroup(
                    match_type="exact_email",
                    match_key=email,
                    confidence=1.0,
                    records=customers,
                ))
        
        logger.debug("Found %d email duplicate groups", len(groups))
        return groups
    
    def _find_company_duplicates(self) -> list[DuplicateGroup]:
        """Find customers with similar company names."""
        # Build normalized company name index
        company_customers: list[tuple[str, str, dict]] = []  # (normalized, original, customer)
        
        for customer in self.customers:
            company = customer.get("company")
            if company:
                normalized = self._normalize_company(company)
                if normalized and len(normalized) >= 3:  # Skip very short names
                    company_customers.append((normalized, company, customer))
        
        # Find similar companies using n-gram comparison
        # Group exact matches first
        exact_groups: dict[str, list[dict]] = defaultdict(list)
        for normalized, original, customer in company_customers:
            exact_groups[normalized].append(customer)
        
        groups = []
        
        # Find exact normalized matches
        for normalized, customers in exact_groups.items():
            if len(customers) > 1:
                # Check if they have different IDs (not already grouped by email)
                unique_ids = {c.get("id") for c in customers}
                if len(unique_ids) > 1:
                    groups.append(DuplicateGroup(
                        match_type="exact_company",
                        match_key=normalized,
                        confidence=0.95,
                        records=customers,
                    ))
        
        # Find fuzzy matches (more expensive)
        normalized_names = list(exact_groups.keys())
        matched_pairs: set[tuple[str, str]] = set()
        
        for i, name1 in enumerate(normalized_names):
            if len(exact_groups[name1]) > 1:
                continue  # Already an exact match group
            
            for name2 in normalized_names[i + 1:]:
                if len(exact_groups[name2]) > 1:
                    continue
                
                # Skip if already matched
                pair_key = tuple(sorted([name1, name2]))
                if pair_key in matched_pairs:
                    continue
                
                # Check similarity
                similarity = SequenceMatcher(None, name1, name2).ratio()
                if similarity >= self.COMPANY_SIMILARITY_THRESHOLD:
                    matched_pairs.add(pair_key)
                    
                    combined = exact_groups[name1] + exact_groups[name2]
                    unique_ids = {c.get("id") for c in combined}
                    
                    if len(unique_ids) > 1:
                        groups.append(DuplicateGroup(
                            match_type="similar_company",
                            match_key=f"{name1} ~ {name2}",
                            confidence=similarity,
                            records=combined,
                        ))
        
        logger.debug("Found %d company duplicate groups", len(groups))
        return groups
    
    def save_results(self, report: DeduplicationReport, output_dir: Path) -> Path:
        """Save deduplication report to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "deduplication_report.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        logger.info("Deduplication report saved to %s", output_file)
        return output_file
    
    def print_summary(self, report: DeduplicationReport) -> None:
        """Print a summary of the deduplication analysis."""
        print("\n=== Deduplication Report ===")
        print(f"Total records analyzed: {report.total_records}")
        print(f"Unique records: {report.unique_records}")
        print(f"Potential duplicates: {report.total_duplicates}")
        print(f"Duplication rate: {report.duplication_rate:.1%}")
        print()
        
        if report.duplicate_groups:
            print("Duplicate groups by type:")
            for match_type, count in report._group_by_type().items():
                print(f"  {match_type}: {count} groups")
            print()
            
            print("Top duplicate groups:")
            sorted_groups = sorted(
                report.duplicate_groups,
                key=lambda g: g.count,
                reverse=True,
            )[:10]
            
            for group in sorted_groups:
                print(f"  [{group.match_type}] {group.match_key}")
                print(f"    Count: {group.count}, Confidence: {group.confidence:.0%}")
                for record in group.records[:3]:
                    print(f"      - {record.get('id')}: {record.get('email', 'N/A')}")
                if group.count > 3:
                    print(f"      ... and {group.count - 3} more")
        else:
            print("No duplicates detected.")


def run_deduplication(data_dir: Path) -> DeduplicationReport:
    """Run deduplication analysis and return report."""
    analyzer = DeduplicationAnalyzer(data_dir)
    if not analyzer.load_data():
        raise RuntimeError("Failed to load data for deduplication")
    
    report = analyzer.analyze()
    
    # Save report
    output_dir = data_dir / "analysis"
    analyzer.save_results(report, output_dir)
    
    return report
