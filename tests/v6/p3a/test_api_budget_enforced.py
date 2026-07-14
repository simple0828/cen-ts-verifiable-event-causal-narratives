import pytest
from concurrent.futures import ThreadPoolExecutor

from cen_ts.api_config import APIConfig, BudgetExceeded, BudgetLedger


def test_api_budget_enforced(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json", APIConfig(api_key="x", base_url="u", model="m", max_calls=1))
    ledger.reserve_call()
    with pytest.raises(BudgetExceeded):
        ledger.reserve_call()


def test_concurrent_budget_reservations_are_atomic(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json", APIConfig(api_key="x", base_url="u", model="m", max_calls=20))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: ledger.reserve_call(), range(20)))
    assert ledger.read()["calls"] == 20
