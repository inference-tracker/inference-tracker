import pytest

from inference_tracker import _attach


@pytest.fixture(autouse=True)
def _attached_tracking_off():
    """init() attaches to the SDKs for the whole process; each test starts without it (test_attach turns it on)."""
    _attach.off()
    yield
    _attach.off()
