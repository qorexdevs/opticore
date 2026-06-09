"""Tests for the IBKR provider, fully offline.

ib_async only talks to a live TWS/Gateway, so CI never ran more than the
import. Here we stub the slice of ib_async the adapter uses (IB.connect,
reqSecDefOptParams, qualifyContracts, reqTickers) with deterministic data
and exercise the chain-building logic: expiry/strike filtering, mid, and
the DataFrame schema contract shared with ``fetch_chain`` / ``enrich``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _make_fake_ibkr(
    *,
    price=100.0,
    expirations=("20260515", "20260619", "20260717"),
    strikes=(90.0, 95.0, 100.0, 105.0, 110.0),
    exchange="SMART",
    qualify_conid=12345,
):
    """Return a fake ``ib_async`` module shaped enough for the adapter."""

    def _stock(symbol, exch, currency):
        return SimpleNamespace(
            symbol=symbol, exchange=exch, currency=currency, secType="STK", conId=0
        )

    def _option(symbol, exp, strike, right, exch, currency="USD"):
        return SimpleNamespace(
            symbol=symbol,
            lastTradeDateOrContractMonth=exp,
            strike=strike,
            right=right,
            exchange=exch,
            currency=currency,
            secType="OPT",
            conId=0,
        )

    class FakeIB:
        def __init__(self):
            self.connected = False
            self.RequestTimeout = None

        def connect(self, host, port, clientId, timeout, **kw):
            self.connected = True

        def reqMarketDataType(self, market_data_type):
            pass

        def qualifyContracts(self, *contracts):
            for c in contracts:
                c.conId = qualify_conid
            return list(contracts)

        def reqSecDefOptParams(self, symbol, fut_fop, sec_type, con_id):
            if not expirations or not strikes:
                return []
            return [
                SimpleNamespace(
                    exchange=exchange,
                    expirations=set(expirations),
                    strikes=set(strikes),
                )
            ]

        def reqTickers(self, *contracts):
            out = []
            for c in contracts:
                if getattr(c, "secType", "") == "STK":
                    out.append(
                        SimpleNamespace(
                            contract=c,
                            marketPrice=lambda: price,
                            last=price,
                            close=price,
                        )
                    )
                else:
                    out.append(
                        SimpleNamespace(
                            contract=c,
                            bid=1.0,
                            ask=1.2,
                            last=1.1,
                            volume=10,
                            open_interest=100,
                        )
                    )
            return out

        def sleep(self, seconds):
            pass

        def isConnected(self):
            return self.connected

        def disconnect(self):
            self.connected = False

    return SimpleNamespace(IB=FakeIB, Stock=_stock, Option=_option)


def test_fetch_ibkr_returns_expected_schema(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "ib_async", _make_fake_ibkr())
    from opticore.data.ibkr import fetch_ibkr_chain

    df = fetch_ibkr_chain("AAPL", max_expiries=2, strike_count=2)

    expected = {
        "symbol",
        "strike",
        "expiry",
        "kind",
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "underlying_price",
        "mid",
    }
    assert set(df.columns) == expected
    assert (df["symbol"] == "AAPL").all()
    assert set(df["kind"].unique()) <= {"call", "put"}
    assert pd.api.types.is_datetime64_any_dtype(df["expiry"])
    assert df["expiry"].iloc[0].tz is not None
    np.testing.assert_allclose(df["mid"].iloc[0], (1.0 + 1.2) / 2)


def test_fetch_ibkr_filters_strikes_around_atm(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "ib_async", _make_fake_ibkr())
    from opticore.data.ibkr import fetch_ibkr_chain

    # underlying 100, strikes [90, 95, 100, 105, 110], ATM index = 2.
    # strike_count=1 keeps one strike either side of ATM -> [95, 100, 105].
    df = fetch_ibkr_chain("AAPL", max_expiries=1, strike_count=1)

    assert sorted(df["strike"].unique()) == [95.0, 100.0, 105.0]
    # one expiry, 3 strikes, call+put
    assert len(df) == 3 * 2


def test_fetch_ibkr_empty_option_params_raises(monkeypatch):
    fake = _make_fake_ibkr(expirations=(), strikes=())
    monkeypatch.setitem(__import__("sys").modules, "ib_async", fake)
    from opticore.data.ibkr import fetch_ibkr_chain

    with pytest.raises(ValueError, match="No option chain"):
        fetch_ibkr_chain("AAPL")


def test_fetch_ibkr_no_qualified_contracts_raises(monkeypatch):
    # qualify leaves conId at 0, so every option is filtered out
    fake = _make_fake_ibkr(qualify_conid=0)
    monkeypatch.setitem(__import__("sys").modules, "ib_async", fake)
    from opticore.data.ibkr import fetch_ibkr_chain

    with pytest.raises(ValueError, match="No valid option contracts"):
        fetch_ibkr_chain("AAPL")


def test_ibkr_output_compatible_with_enrich(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "ib_async", _make_fake_ibkr())
    from opticore import enrich
    from opticore.data.ibkr import fetch_ibkr_chain

    chain = fetch_ibkr_chain("AAPL", max_expiries=1, strike_count=2)
    enriched = enrich(chain, rate=0.05)

    for col in ("iv", "delta", "gamma", "theta", "vega", "rho", "mid", "tte"):
        assert col in enriched.columns, f"missing {col}"
    assert pd.api.types.is_numeric_dtype(enriched["delta"])
