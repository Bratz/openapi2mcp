"""Placeholder integration smoke test — replaced by real graph tests in M5.

Exists so `pytest tests/integration -q` (a CLAUDE.md must-pass command)
collects at least one test on a fresh clone (git does not track empty dirs).
"""

def test_package_importable():
    import hermes

    assert hermes.__version__
