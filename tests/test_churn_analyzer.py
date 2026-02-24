import pytest
from chargebee_mapper.churn_analyzer import _get_risk_level

@pytest.mark.parametrize("score,expected_risk", [
    (100, "critical"),
    (71, "critical"),
    (70, "critical"),
    (69, "high"),
    (51, "high"),
    (50, "high"),
    (49, "medium"),
    (31, "medium"),
    (30, "medium"),
    (29, "low"),
    (1, "low"),
    (0, "low"),
    (-1, "low"),
])
def test_get_risk_level_boundaries(score, expected_risk):
    """Test _get_risk_level with boundary values."""
    assert _get_risk_level(score) == expected_risk
