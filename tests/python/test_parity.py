"""Tests for oc.parity_check and oc.implied_forward (Issues #28, #29)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import opticore as oc
import pandas as pd
import pytest


def _synthetic_chain(
    underlying: float = 100.0,
    rate: float = 0.05,
    div_yield: float = 0.0,
    vol: float = 0.20,
    n_strikes: int = 11,
    expiry_days: tuple[int, ...] = (30, 60, 120),
    spread_bps: float = 0.0,
) -> pd.DataFrame:
    """Build a put-call-parity-clean chain by pricing both sides with BSM.

    With matched (rate, div_yield, vol), parity holds to ~1e-12. ``spread_bps``
    optionally widens bid/ask around the model price.
    """
    now = datetime.now(timezone.utc)
    strikes = np.linspace(85.0, 115.0, n_strikes)
    rows = []
    for d in expiry_days:
        # Keep full timestamp (no normalize) so tte matches the BSM price exactly
        exp_dt = now + timedelta(days=d)
        exp_ts = pd.Timestamp(exp_dt)
        tte = (exp_dt - now).total_seconds() / (365.25 * 24 * 3600)
        for k in strikes:
            for kind in ("call", "put"):
                p = oc.price(
                    spot=underlying,
                    strike=k,
                    expiry=tte,
                    rate=rate,
                    vol=vol,
                    kind=kind,
                    div_yield=div_yield,
                )
                half = p * spread_bps / 1e4
                rows.append(
                    {
                        "symbol": "TEST",
                        "expiry": exp_ts,
                        "strike": float(k),
                        "kind": kind,
                        "bid": max(p - half, 0.01),
                        "ask": p + half,
                        "last": p,
                        "mid": p,
                        "volume": 100,
                        "open_interest": 500,
                        "underlying_price": underlying,
                    }
                )
    return pd.DataFrame(rows)


# ── parity_check ────────────────────────────────────────────────────────────


class TestParityCheck:
    def test_clean_chain_residuals_near_zero(self):
        chain = _synthetic_chain(rate=0.05, div_yield=0.02)
        diag = oc.parity_check(chain, rate=0.05, div_yield=0.02)
        assert not diag.empty
        # parity_check recomputes tte from wall clock, so time elapsed since
        # the chain was built leaks into the residual (~1.2e-8 per 100ms).
        # 1e-6 allows a few seconds of drift while still catching real errors.
        assert np.abs(diag["parity_residual"]).max() < 1e-6

    def test_returns_expected_columns(self):
        chain = _synthetic_chain()
        diag = oc.parity_check(chain, rate=0.05)
        assert set(diag.columns) == {
            "expiry",
            "strike",
            "call_mid",
            "put_mid",
            "parity_residual",
            "residual_pct",
        }

    def test_flags_bad_row(self):
        """Corrupting one call price must produce a clearly-flagged outlier."""
        chain = _synthetic_chain(rate=0.05, div_yield=0.0)
        # Bump one call's mid by $5
        mask = (chain["kind"] == "call") & np.isclose(chain["strike"], 100.0)
        # Pick the first matching row
        idx = chain[mask].index[0]
        chain.loc[idx, "mid"] = chain.loc[idx, "mid"] + 5.0

        diag = oc.parity_check(chain, rate=0.05, div_yield=0.0)
        worst = diag.loc[diag["parity_residual"].abs().idxmax()]
        assert np.isclose(worst["strike"], 100.0)
        # Residual is approximately the $5 we injected
        assert abs(worst["parity_residual"] - 5.0) < 1e-6

    def test_handles_missing_mid_via_bid_ask(self):
        """When 'mid' is absent but bid/ask present, parity_check computes it."""
        chain = _synthetic_chain()
        chain = chain.drop(columns=["mid"])
        diag = oc.parity_check(chain, rate=0.05)
        assert not diag.empty

    def test_empty_chain_returns_empty_frame(self):
        empty = pd.DataFrame(columns=["expiry", "strike", "kind", "underlying_price", "mid"])
        diag = oc.parity_check(empty)
        assert diag.empty
        assert "parity_residual" in diag.columns

    def test_unpaired_strikes_dropped(self):
        """Strikes with only a call (no put) shouldn't appear in output."""
        chain = _synthetic_chain(n_strikes=5)
        # Drop all puts at strike=85 — that strike should disappear
        chain = chain.drop(
            chain[(chain["kind"] == "put") & np.isclose(chain["strike"], 85.0)].index
        )
        diag = oc.parity_check(chain, rate=0.05)
        assert not (np.isclose(diag["strike"], 85.0)).any()


# ── implied_forward ─────────────────────────────────────────────────────────


class TestImpliedForward:
    def test_recovers_known_div_yield(self):
        """Build a chain with q=0.025; recover q within ~1bp."""
        rate = 0.05
        q_true = 0.025
        chain = _synthetic_chain(rate=rate, div_yield=q_true)
        out = oc.implied_forward(chain, rate=rate)
        assert not out.empty
        # Each expiry's recovered q should be within 1bp of truth
        err = (out["implied_div_yield"] - q_true).abs()
        assert err.max() < 1e-4, f"max err = {err.max():.6f}"

    def test_zero_div_yield(self):
        rate = 0.05
        chain = _synthetic_chain(rate=rate, div_yield=0.0)
        out = oc.implied_forward(chain, rate=rate)
        assert (out["implied_div_yield"].abs() < 1e-4).all()

    def test_forward_equals_S_exp_minus_qT(self):
        """F should equal S*exp((r-q)*T) per BSM."""
        rate = 0.05
        q = 0.03
        chain = _synthetic_chain(rate=rate, div_yield=q, expiry_days=(30, 90))
        out = oc.implied_forward(chain, rate=rate)
        S = 100.0
        for _, row in out.iterrows():
            expected = S * np.exp((rate - q) * row["tte"])
            assert abs(row["forward"] - expected) < 0.02

    def test_returns_expected_columns(self):
        chain = _synthetic_chain()
        out = oc.implied_forward(chain, rate=0.05)
        assert set(out.columns) == {
            "expiry",
            "tte",
            "forward",
            "implied_div_yield",
            "n_strikes_used",
        }

    def test_one_row_per_expiry(self):
        chain = _synthetic_chain(expiry_days=(30, 60, 120))
        out = oc.implied_forward(chain, rate=0.05)
        assert len(out) == 3
        assert out["expiry"].is_unique

    def test_n_atm_strikes_respected(self):
        chain = _synthetic_chain(n_strikes=11)
        out = oc.implied_forward(chain, rate=0.05, n_atm_strikes=5)
        assert (out["n_strikes_used"] <= 5).all()

    def test_empty_chain_returns_empty_frame(self):
        empty = pd.DataFrame(columns=["expiry", "strike", "kind", "underlying_price", "mid"])
        out = oc.implied_forward(empty)
        assert out.empty
        assert "forward" in out.columns


# ── atm_iv ──────────────────────────────────────────────────────────────────


class TestAtmIv:
    def test_recovers_flat_vol(self):
        """A chain priced at a flat 0.20 vol should give atm_iv ~ 0.20."""
        chain = _synthetic_chain(vol=0.20, rate=0.05)
        out = oc.atm_iv(chain, rate=0.05)
        assert not out.empty
        assert (out["atm_iv"] - 0.20).abs().max() < 1e-3

    def test_one_row_per_expiry_sorted(self):
        chain = _synthetic_chain(expiry_days=(120, 30, 60))
        out = oc.atm_iv(chain, rate=0.05)
        assert len(out) == 3
        assert out["expiry"].is_unique
        assert out["tte"].is_monotonic_increasing

    def test_atm_strike_is_nearest_spot(self):
        chain = _synthetic_chain(n_strikes=11)  # strikes 85..115, spot 100
        out = oc.atm_iv(chain, rate=0.05)
        assert (out["atm_strike"] == 100.0).all()

    def test_returns_expected_columns(self):
        chain = _synthetic_chain()
        out = oc.atm_iv(chain, rate=0.05)
        assert set(out.columns) == {
            "expiry",
            "tte",
            "atm_strike",
            "atm_iv",
            "underlying_price",
        }

    def test_empty_chain_returns_empty_frame(self):
        empty = pd.DataFrame(columns=["expiry", "strike", "kind", "underlying_price", "mid"])
        out = oc.atm_iv(empty)
        assert out.empty
        assert "atm_iv" in out.columns


# ── term_slope ───────────────────────────────────────────────────────────────


def _atm_frame(pairs):
    """Build a minimal atm_iv-shaped frame from (tte, iv) pairs."""
    return pd.DataFrame(
        {
            "expiry": [f"e{i}" for i in range(len(pairs))],
            "tte": [t for t, _ in pairs],
            "atm_strike": [100.0] * len(pairs),
            "atm_iv": [v for _, v in pairs],
            "underlying_price": [100.0] * len(pairs),
        }
    )


class TestTermSlope:
    def test_flat_vol_chain_is_flat(self):
        out = oc.term_slope(oc.atm_iv(_synthetic_chain(vol=0.20), rate=0.05))
        assert out.shape == "flat"
        assert abs(out.slope) < 1e-3

    def test_rising_curve_is_contango(self):
        out = oc.term_slope(_atm_frame([(0.1, 0.18), (0.3, 0.22), (0.6, 0.26)]))
        assert out.shape == "contango"
        assert out.slope > 0
        assert out.front_iv == 0.18
        assert out.back_iv == 0.26

    def test_falling_curve_is_backwardation(self):
        out = oc.term_slope(_atm_frame([(0.1, 0.30), (0.3, 0.24), (0.6, 0.20)]))
        assert out.shape == "backwardation"
        assert out.slope < 0

    def test_unsorted_input_uses_tenor_order(self):
        out = oc.term_slope(_atm_frame([(0.6, 0.26), (0.1, 0.18)]))
        assert out.front_tte == 0.1
        assert out.back_tte == 0.6

    def test_single_expiry_raises(self):
        with pytest.raises(ValueError):
            oc.term_slope(_atm_frame([(0.2, 0.2)]))

    def test_missing_columns_raises(self):
        with pytest.raises(KeyError):
            oc.term_slope(pd.DataFrame({"foo": [1, 2]}))


# ── Smoke tests for public API surface ──────────────────────────────────────


def test_parity_check_in_public_api():
    assert hasattr(oc, "parity_check")
    assert callable(oc.parity_check)


def test_implied_forward_in_public_api():
    assert hasattr(oc, "implied_forward")
    assert callable(oc.implied_forward)


def test_atm_iv_in_public_api():
    assert hasattr(oc, "atm_iv")
    assert callable(oc.atm_iv)


def test_term_slope_in_public_api():
    assert hasattr(oc, "term_slope")
    assert callable(oc.term_slope)
