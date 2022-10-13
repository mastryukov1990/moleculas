import pytest


@pytest.fixture
def molecules():
    return [
        "FC1CCNCC1",
        "CCCCON",
        "N#CN1CCC1",
    ]
