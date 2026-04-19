"""
Shared test fixtures.
"""


def pytest_configure(config):
    """Register custom markers."""
    # slow: bootstrap-heavy statistical tests that take several seconds each.
    # Skip with: pytest -m "not slow"
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running (bootstrap-heavy statistical tests)"
    )
