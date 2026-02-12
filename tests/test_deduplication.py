"""Unit tests for chargebee_mapper.deduplication module."""

import json
from pathlib import Path

import pytest

from chargebee_mapper.deduplication import (
    DeduplicationAnalyzer,
    DeduplicationReport,
    DuplicateGroup,
)


class TestDuplicateGroup:
    """Tests for DuplicateGroup dataclass."""

    def test_init(self):
        """Test basic initialization."""
        group = DuplicateGroup(
            match_type="exact_email",
            match_key="test@example.com",
            confidence=1.0,
        )
        
        assert group.match_type == "exact_email"
        assert group.match_key == "test@example.com"
        assert group.confidence == 1.0
        assert group.records == []
        assert group.count == 0

    def test_with_records(self):
        """Test with records."""
        group = DuplicateGroup(
            match_type="exact_email",
            match_key="test@example.com",
            confidence=1.0,
            records=[{"id": "1"}, {"id": "2"}],
        )
        
        assert group.count == 2

    def test_to_dict(self):
        """Test serialization to dict."""
        group = DuplicateGroup(
            match_type="exact_email",
            match_key="test@example.com",
            confidence=1.0,
            records=[{"id": "1", "email": "test@example.com"}],
        )
        
        result = group.to_dict()
        
        assert result["match_type"] == "exact_email"
        assert result["count"] == 1
        assert len(result["records"]) == 1


class TestDeduplicationReport:
    """Tests for DeduplicationReport dataclass."""

    def test_init(self):
        """Test basic initialization."""
        report = DeduplicationReport(total_records=100)
        
        assert report.total_records == 100
        assert report.unique_records == 0
        assert report.duplicate_groups == []

    def test_total_duplicates(self):
        """Test total_duplicates calculation."""
        report = DeduplicationReport(total_records=100)
        report.duplicate_groups = [
            DuplicateGroup("exact_email", "a@b.com", 1.0, [{"id": "1"}, {"id": "2"}]),
            DuplicateGroup("exact_email", "c@d.com", 1.0, [{"id": "3"}, {"id": "4"}, {"id": "5"}]),
        ]
        
        # 2-1 + 3-1 = 3 duplicates (all but one in each group)
        assert report.total_duplicates == 3

    def test_duplication_rate(self):
        """Test duplication_rate calculation."""
        report = DeduplicationReport(total_records=100, unique_records=97)
        report.duplicate_groups = [
            DuplicateGroup("exact_email", "a@b.com", 1.0, [{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}]),
        ]
        
        # 3 duplicates out of 100 records
        assert report.duplication_rate == 0.03

    def test_duplication_rate_zero_records(self):
        """Test duplication_rate with zero records."""
        report = DeduplicationReport(total_records=0)
        
        assert report.duplication_rate == 0.0


class TestDeduplicationAnalyzer:
    """Tests for DeduplicationAnalyzer class."""

    @pytest.fixture
    def analyzer_with_data(self, temp_dir):
        """Create an analyzer with sample customer data."""
        json_dir = temp_dir / "json"
        json_dir.mkdir()
        
        customers = [
            {"id": "1", "email": "alice@example.com", "company": "Acme Inc."},
            {"id": "2", "email": "bob@example.com", "company": "Bob's Shop"},
            {"id": "3", "email": "alice@example.com", "company": "Acme Corp"},  # Duplicate email
            {"id": "4", "email": "charlie@test.com", "company": "Acme Inc"},  # Similar company
        ]
        
        with open(json_dir / "customers.json", "w") as f:
            json.dump(customers, f)
        
        analyzer = DeduplicationAnalyzer(temp_dir)
        analyzer.load_data()
        return analyzer

    def test_normalize_email(self, temp_dir):
        """Test email normalization."""
        analyzer = DeduplicationAnalyzer(temp_dir)
        
        assert analyzer._normalize_email("Test@Example.COM") == "test@example.com"
        assert analyzer._normalize_email("  test@example.com  ") == "test@example.com"
        assert analyzer._normalize_email("DONOTEMAIL-test@example.com") == "test@example.com"
        assert analyzer._normalize_email("NO_EMAILS_test@example.com") == "test@example.com"
        assert analyzer._normalize_email(None) is None
        assert analyzer._normalize_email("invalid") is None

    def test_normalize_company(self, temp_dir):
        """Test company name normalization."""
        analyzer = DeduplicationAnalyzer(temp_dir)
        
        assert analyzer._normalize_company("Acme Inc.") == "acme"
        assert analyzer._normalize_company("Acme LLC") == "acme"
        assert analyzer._normalize_company("Acme Corporation") == "acme"
        assert analyzer._normalize_company("Acme (o/a Other Name)") == "acme"
        assert analyzer._normalize_company("") == ""
        assert analyzer._normalize_company(None) == ""

    def test_find_email_duplicates(self, analyzer_with_data):
        """Test finding email duplicates."""
        duplicates = analyzer_with_data._find_email_duplicates()
        
        assert len(duplicates) == 1
        assert duplicates[0].match_key == "alice@example.com"
        assert duplicates[0].count == 2

    def test_analyze(self, analyzer_with_data):
        """Test full analysis."""
        report = analyzer_with_data.analyze()
        
        assert report.total_records == 4
        assert len(report.duplicate_groups) >= 1  # At least email duplicates

    def test_save_results(self, analyzer_with_data, temp_dir):
        """Test saving analysis results."""
        report = analyzer_with_data.analyze()
        output_dir = temp_dir / "analysis"
        
        output_file = analyzer_with_data.save_results(report, output_dir)
        
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert "total_records" in data
        assert "duplicate_groups" in data
