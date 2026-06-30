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


# ── iv_skew ──────────────────────────────────────────────────────────────────


def _skewed_chain(slope_per_lm=-0.5, vol_atm=0.20, underlying=100.0, expiry_days=(30,)):
    """Chain priced with a vol that varies linearly in log-moneyness.

    ``slope_per_lm`` is d(vol)/d(ln(K/S)); negative gives the equity-style
    negative skew (low strikes carry more vol). iv_skew should recover it.
    """
    now = datetime.now(timezone.utc)
    strikes = np.linspace(85.0, 115.0, 11)
    rows = []
    for d in expiry_days:
        exp_dt = now + timedelta(days=d)
        exp_ts = pd.Timestamp(exp_dt)
        tte = (exp_dt - now).total_seconds() / (365.25 * 24 * 3600)
        for k in strikes:
            vol = vol_atm + slope_per_lm * np.log(k / underlying)
            for kind in ("call", "put"):
                p = oc.price(spot=underlying, strike=k, expiry=tte, rate=0.05, vol=vol, kind=kind)
                rows.append(
                    {
                        "symbol": "TEST",
                        "expiry": exp_ts,
                        "strike": float(k),
                        "kind": kind,
                        "mid": p,
                        "underlying_price": underlying,
                    }
                )
    return pd.DataFrame(rows)


class TestIvSkew:
    def test_negative_skew_recovered(self):
        out = oc.iv_skew(_skewed_chain(slope_per_lm=-0.5), rate=0.05)
        assert len(out) == 1
        assert out["skew"].iloc[0] < 0
        # the fit should land near the slope we priced in
        assert out["skew"].iloc[0] == pytest.approx(-0.5, abs=0.05)
        assert out["put_wing_iv"].iloc[0] > out["call_wing_iv"].iloc[0]

    def test_flat_smile_is_near_zero(self):
        out = oc.iv_skew(_skewed_chain(slope_per_lm=0.0), rate=0.05)
        assert abs(out["skew"].iloc[0]) < 1e-2

    def test_one_row_per_expiry_sorted(self):
        out = oc.iv_skew(_skewed_chain(expiry_days=(120, 30, 60)), rate=0.05)
        assert len(out) == 3
        assert list(out["tte"]) == sorted(out["tte"])

    def test_returns_expected_columns(self):
        out = oc.iv_skew(_skewed_chain(), rate=0.05)
        for col in ("expiry", "tte", "atm_iv", "skew", "put_wing_iv", "call_wing_iv", "n_strikes"):
            assert col in out.columns

    def test_empty_chain_returns_empty_frame(self):
        empty = pd.DataFrame(columns=["expiry", "strike", "kind", "underlying_price", "mid"])
        out = oc.iv_skew(empty)
        assert out.empty
        assert "skew" in out.columns


class TestRrBf:
    def test_negative_skew_gives_negative_rr(self):
        out = oc.rr_bf(_skewed_chain(slope_per_lm=-0.5), rate=0.05)
        assert len(out) == 1
        # equity put skew: puts richer than calls, so call - put wing is negative
        assert out["rr"].iloc[0] < 0

    def test_flat_smile_rr_near_zero(self):
        out = oc.rr_bf(_skewed_chain(slope_per_lm=0.0), rate=0.05)
        assert abs(out["rr"].iloc[0]) < 1e-2

    def test_matches_iv_skew_wings(self):
        chain = _skewed_chain(slope_per_lm=-0.3)
        skew = oc.iv_skew(chain, rate=0.05)
        out = oc.rr_bf(chain, rate=0.05)
        put_w = skew["put_wing_iv"].iloc[0]
        call_w = skew["call_wing_iv"].iloc[0]
        rr = call_w - put_w
        bf = (put_w + call_w) / 2 - skew["atm_iv"].iloc[0]
        assert out["rr"].iloc[0] == pytest.approx(rr, abs=1e-6)
        assert out["bf"].iloc[0] == pytest.approx(bf, abs=1e-6)

    def test_returns_expected_columns(self):
        out = oc.rr_bf(_skewed_chain(), rate=0.05)
        for col in ("expiry", "tte", "atm_iv", "rr", "bf", "n_strikes"):
            assert col in out.columns

    def test_empty_chain_returns_empty_frame(self):
        empty = pd.DataFrame(columns=["expiry", "strike", "kind", "underlying_price", "mid"])
        out = oc.rr_bf(empty)
        assert out.empty
        assert "rr" in out.columns


class TestStraddle:
    def test_atm_strike_is_nearest_spot(self):
        out = oc.straddle(_synthetic_chain(underlying=100.0))
        assert (out["atm_strike"] == 100.0).all()

    def test_price_equals_call_plus_put(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        out = oc.straddle(chain)
        row = chain[(chain["strike"] == 100.0) & (chain["expiry"] == chain["expiry"].iloc[0])]
        call = float(row[row["kind"] == "call"]["mid"].iloc[0])
        put = float(row[row["kind"] == "put"]["mid"].iloc[0])
        assert out["straddle_price"].iloc[0] == pytest.approx(call + put, abs=1e-9)

    def test_breakevens_one_width_either_side(self):
        out = oc.straddle(_synthetic_chain())
        r = out.iloc[0]
        assert r["breakeven_high"] == pytest.approx(r["atm_strike"] + r["straddle_price"])
        assert r["breakeven_low"] == pytest.approx(r["atm_strike"] - r["straddle_price"])

    def test_implied_move_is_straddle_over_spot(self):
        out = oc.straddle(_synthetic_chain(underlying=100.0))
        r = out.iloc[0]
        assert r["implied_move"] == pytest.approx(r["straddle_price"] / r["underlying_price"])

    def test_one_row_per_expiry_sorted(self):
        out = oc.straddle(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["tte"]) == sorted(out["tte"])

    def test_returns_expected_columns(self):
        out = oc.straddle(_synthetic_chain())
        for col in (
            "expiry",
            "tte",
            "atm_strike",
            "underlying_price",
            "straddle_price",
            "breakeven_low",
            "breakeven_high",
            "implied_move",
        ):
            assert col in out.columns

    def test_empty_chain_returns_empty_frame(self):
        empty = pd.DataFrame(columns=["expiry", "strike", "kind", "underlying_price", "mid"])
        out = oc.straddle(empty)
        assert out.empty
        assert "straddle_price" in out.columns


class TestStrangle:
    def test_legs_are_first_otm_pair(self):
        # strikes 85..115 step 3, spot 100 -> nearest OTM call 103, put 97
        out = oc.strangle(_synthetic_chain(underlying=100.0))
        assert (out["call_strike"] == 103.0).all()
        assert (out["put_strike"] == 97.0).all()

    def test_width_two_steps_further_out(self):
        out = oc.strangle(_synthetic_chain(underlying=100.0), width=2)
        assert (out["call_strike"] == 106.0).all()
        assert (out["put_strike"] == 94.0).all()

    def test_price_equals_call_plus_put(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        out = oc.strangle(chain)
        exp = chain["expiry"].iloc[0]
        call = float(
            chain[
                (chain["strike"] == 103.0) & (chain["kind"] == "call") & (chain["expiry"] == exp)
            ]["mid"].iloc[0]
        )
        put = float(
            chain[(chain["strike"] == 97.0) & (chain["kind"] == "put") & (chain["expiry"] == exp)][
                "mid"
            ].iloc[0]
        )
        assert out["strangle_price"].iloc[0] == pytest.approx(call + put, abs=1e-9)

    def test_breakevens_one_width_outside_each_strike(self):
        out = oc.strangle(_synthetic_chain())
        r = out.iloc[0]
        assert r["breakeven_high"] == pytest.approx(r["call_strike"] + r["strangle_price"])
        assert r["breakeven_low"] == pytest.approx(r["put_strike"] - r["strangle_price"])

    def test_implied_move_is_cost_over_spot(self):
        out = oc.strangle(_synthetic_chain(underlying=100.0))
        r = out.iloc[0]
        assert r["implied_move"] == pytest.approx(r["strangle_price"] / r["underlying_price"])

    def test_cheaper_than_straddle(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        strad = oc.straddle(chain)["straddle_price"].iloc[0]
        strang = oc.strangle(chain)["strangle_price"].iloc[0]
        assert strang < strad

    def test_one_row_per_expiry_sorted(self):
        out = oc.strangle(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["tte"]) == sorted(out["tte"])

    def test_width_too_far_drops_expiry(self):
        # 5 strikes each side of spot; width 6 has nothing that far out
        out = oc.strangle(_synthetic_chain(underlying=100.0), width=6)
        assert out.empty

    def test_width_below_one_raises(self):
        with pytest.raises(ValueError):
            oc.strangle(_synthetic_chain(), width=0)

    def test_returns_expected_columns(self):
        out = oc.strangle(_synthetic_chain())
        for col in (
            "expiry",
            "tte",
            "put_strike",
            "call_strike",
            "underlying_price",
            "strangle_price",
            "breakeven_low",
            "breakeven_high",
            "implied_move",
        ):
            assert col in out.columns

    def test_empty_chain_returns_empty_frame(self):
        empty = pd.DataFrame(columns=["expiry", "strike", "kind", "underlying_price", "mid"])
        out = oc.strangle(empty)
        assert out.empty
        assert "strangle_price" in out.columns


class TestVertical:
    def test_legs_bracket_spot(self):
        # strikes step 3 (85..115), spot 100 -> low leg 100, high leg 103
        out = oc.vertical(_synthetic_chain(underlying=100.0, expiry_days=(30,)))
        r = out.iloc[0]
        assert r["long_strike"] == 100.0
        assert r["short_strike"] == 103.0

    def test_bull_call_is_debit_with_capped_profit(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        out = oc.vertical(chain, kind="call", side="bull")
        r = out.iloc[0]
        assert r["net_debit"] > 0  # pay to open
        # max profit = strike width - debit, max loss = -debit
        assert r["max_profit"] == pytest.approx(3.0 - r["net_debit"])
        assert r["max_loss"] == pytest.approx(-r["net_debit"])
        assert r["breakeven"] == pytest.approx(100.0 + r["net_debit"])

    def test_bear_call_is_credit_mirror_of_bull(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        bull = oc.vertical(chain, kind="call", side="bull").iloc[0]
        bear = oc.vertical(chain, kind="call", side="bear").iloc[0]
        # same two strikes, opposite sign cost and flipped profit/loss
        assert bear["net_debit"] == pytest.approx(-bull["net_debit"])
        assert bear["max_profit"] == pytest.approx(-bull["max_loss"])
        assert bear["max_loss"] == pytest.approx(-bull["max_profit"])
        assert bear["breakeven"] == pytest.approx(bull["breakeven"])

    def test_bull_put_is_credit(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        out = oc.vertical(chain, kind="put", side="bull")
        r = out.iloc[0]
        assert r["net_debit"] < 0  # collect premium
        assert r["max_profit"] == pytest.approx(-r["net_debit"])
        assert r["max_loss"] == pytest.approx(-(3.0 + r["net_debit"]))

    def test_width_widens_the_spread(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        w1 = oc.vertical(chain, width=1).iloc[0]
        w2 = oc.vertical(chain, width=2).iloc[0]
        assert w2["short_strike"] - w2["long_strike"] == pytest.approx(6.0)
        # wider debit spread costs more but can earn more
        assert w2["net_debit"] > w1["net_debit"]
        assert w2["max_profit"] > w1["max_profit"]

    def test_one_row_per_expiry_sorted(self):
        out = oc.vertical(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["tte"]) == sorted(out["tte"])

    def test_returns_expected_columns(self):
        out = oc.vertical(_synthetic_chain())
        for col in (
            "expiry",
            "tte",
            "kind",
            "side",
            "long_strike",
            "short_strike",
            "underlying_price",
            "net_debit",
            "max_profit",
            "max_loss",
            "breakeven",
        ):
            assert col in out.columns

    def test_bad_args_raise(self):
        with pytest.raises(ValueError):
            oc.vertical(_synthetic_chain(), width=0)
        with pytest.raises(ValueError):
            oc.vertical(_synthetic_chain(), kind="straddle")
        with pytest.raises(ValueError):
            oc.vertical(_synthetic_chain(), side="sideways")

    def test_empty_chain_returns_empty_frame(self):
        empty = pd.DataFrame(columns=["expiry", "strike", "kind", "underlying_price", "mid"])
        out = oc.vertical(empty)
        assert out.empty
        assert "net_debit" in out.columns


class TestButterfly:
    def test_wings_bracket_the_body(self):
        # strikes step 3 (85..115), spot 100 -> body 100, wings 97 and 103
        out = oc.butterfly(_synthetic_chain(underlying=100.0, expiry_days=(30,)))
        r = out.iloc[0]
        assert r["low_strike"] == 97.0
        assert r["mid_strike"] == 100.0
        assert r["high_strike"] == 103.0

    def test_long_call_is_debit_with_capped_payoff(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        out = oc.butterfly(chain, kind="call", side="long")
        r = out.iloc[0]
        assert r["net_debit"] > 0  # pay to open
        # peak at the body strike, loss capped at the debit
        assert r["max_profit"] == pytest.approx(3.0 - r["net_debit"])
        assert r["max_loss"] == pytest.approx(-r["net_debit"])
        assert r["breakeven_low"] == pytest.approx(97.0 + r["net_debit"])
        assert r["breakeven_high"] == pytest.approx(103.0 - r["net_debit"])

    def test_short_is_credit_mirror_of_long(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        lng = oc.butterfly(chain, side="long").iloc[0]
        sht = oc.butterfly(chain, side="short").iloc[0]
        assert sht["net_debit"] == pytest.approx(-lng["net_debit"])
        assert sht["max_profit"] == pytest.approx(-lng["max_loss"])
        assert sht["max_loss"] == pytest.approx(-lng["max_profit"])

    def test_call_and_put_butterfly_cost_match(self):
        # symmetric strikes -> parity makes the two debits equal
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        call = oc.butterfly(chain, kind="call").iloc[0]
        put = oc.butterfly(chain, kind="put").iloc[0]
        assert put["net_debit"] == pytest.approx(call["net_debit"])
        assert put["max_profit"] == pytest.approx(call["max_profit"])

    def test_width_widens_the_wings(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        w1 = oc.butterfly(chain, width=1).iloc[0]
        w2 = oc.butterfly(chain, width=2).iloc[0]
        assert w2["high_strike"] - w2["low_strike"] == pytest.approx(12.0)
        # wider body pays more but tops out higher
        assert w2["net_debit"] > w1["net_debit"]
        assert w2["max_profit"] > w1["max_profit"]

    def test_one_row_per_expiry_sorted(self):
        out = oc.butterfly(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["tte"]) == sorted(out["tte"])

    def test_returns_expected_columns(self):
        out = oc.butterfly(_synthetic_chain())
        for col in (
            "expiry",
            "tte",
            "kind",
            "side",
            "low_strike",
            "mid_strike",
            "high_strike",
            "underlying_price",
            "net_debit",
            "max_profit",
            "max_loss",
            "breakeven_low",
            "breakeven_high",
        ):
            assert col in out.columns

    def test_bad_args_raise(self):
        with pytest.raises(ValueError):
            oc.butterfly(_synthetic_chain(), width=0)
        with pytest.raises(ValueError):
            oc.butterfly(_synthetic_chain(), kind="iron")
        with pytest.raises(ValueError):
            oc.butterfly(_synthetic_chain(), side="flat")

    def test_empty_chain_returns_empty_frame(self):
        empty = pd.DataFrame(columns=["expiry", "strike", "kind", "underlying_price", "mid"])
        out = oc.butterfly(empty)
        assert out.empty
        assert "net_debit" in out.columns


class TestIronCondor:
    def test_legs_bracket_spot(self):
        # strikes step 3 (85..115), spot 100, gap 1, width 1
        out = oc.iron_condor(_synthetic_chain(underlying=100.0, expiry_days=(30,)))
        r = out.iloc[0]
        assert r["put_long_strike"] == 94.0
        assert r["put_short_strike"] == 97.0
        assert r["call_short_strike"] == 103.0
        assert r["call_long_strike"] == 106.0

    def test_short_is_credit_with_capped_loss(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        r = oc.iron_condor(chain).iloc[0]
        credit = -r["net_debit"]
        assert r["net_debit"] < 0  # collect to open
        assert r["max_profit"] == pytest.approx(credit)
        # loss is capped at the wing width (3) less the credit kept
        assert r["max_loss"] == pytest.approx(credit - 3.0)

    def test_breakevens_sit_credit_off_the_shorts(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        r = oc.iron_condor(chain).iloc[0]
        credit = -r["net_debit"]
        assert r["breakeven_low"] == pytest.approx(97.0 - credit)
        assert r["breakeven_high"] == pytest.approx(103.0 + credit)

    def test_long_is_debit_mirror_of_short(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        sht = oc.iron_condor(chain, side="short").iloc[0]
        lng = oc.iron_condor(chain, side="long").iloc[0]
        assert lng["net_debit"] == pytest.approx(-sht["net_debit"])
        assert lng["max_profit"] == pytest.approx(-sht["max_loss"])
        assert lng["max_loss"] == pytest.approx(-sht["max_profit"])

    def test_wider_wings_risk_more(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        w1 = oc.iron_condor(chain, width=1).iloc[0]
        w2 = oc.iron_condor(chain, width=2).iloc[0]
        assert w2["put_long_strike"] == 91.0
        assert w2["call_long_strike"] == 109.0
        # same shorts, wings further out -> more credit but a deeper max loss
        assert w2["max_loss"] < w1["max_loss"]

    def test_gap_pushes_shorts_out(self):
        chain = _synthetic_chain(underlying=100.0, expiry_days=(30,))
        r = oc.iron_condor(chain, gap=2).iloc[0]
        assert r["put_short_strike"] == 94.0
        assert r["call_short_strike"] == 106.0

    def test_one_row_per_expiry_sorted(self):
        out = oc.iron_condor(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["tte"]) == sorted(out["tte"])

    def test_returns_expected_columns(self):
        out = oc.iron_condor(_synthetic_chain())
        for col in (
            "expiry",
            "tte",
            "side",
            "put_long_strike",
            "put_short_strike",
            "call_short_strike",
            "call_long_strike",
            "underlying_price",
            "net_debit",
            "max_profit",
            "max_loss",
            "breakeven_low",
            "breakeven_high",
        ):
            assert col in out.columns

    def test_bad_args_raise(self):
        with pytest.raises(ValueError):
            oc.iron_condor(_synthetic_chain(), gap=0)
        with pytest.raises(ValueError):
            oc.iron_condor(_synthetic_chain(), width=0)
        with pytest.raises(ValueError):
            oc.iron_condor(_synthetic_chain(), side="flat")

    def test_empty_chain_returns_empty_frame(self):
        empty = pd.DataFrame(columns=["expiry", "strike", "kind", "underlying_price", "mid"])
        out = oc.iron_condor(empty)
        assert out.empty
        assert "net_debit" in out.columns


class TestCollar:
    def test_legs_bracket_spot(self):
        # strikes step 3 (85..115), spot 100, gap 1
        r = oc.collar(_synthetic_chain(underlying=100.0, expiry_days=(30,))).iloc[0]
        assert r["put_strike"] == 97.0
        assert r["call_strike"] == 103.0
        assert r["underlying_price"] == 100.0

    def test_floor_and_cap_span_the_strikes(self):
        # cap less floor is exactly the strike band, premium drops out
        r = oc.collar(_synthetic_chain(underlying=100.0, expiry_days=(30,))).iloc[0]
        assert r["max_profit"] - r["max_loss"] == pytest.approx(r["call_strike"] - r["put_strike"])
        assert r["max_profit"] == pytest.approx(r["call_strike"] - r["breakeven"])
        assert r["max_loss"] == pytest.approx(r["put_strike"] - r["breakeven"])

    def test_breakeven_is_spot_plus_net(self):
        r = oc.collar(_synthetic_chain(underlying=100.0, expiry_days=(30,))).iloc[0]
        assert r["breakeven"] == pytest.approx(r["underlying_price"] + r["net_debit"])

    def test_positive_carry_is_a_small_credit(self):
        # forward sits above spot under positive rates, so the cap call is dearer
        # than the equidistant floor put -> a credit and a breakeven below spot
        r = oc.collar(_synthetic_chain(underlying=100.0, rate=0.05, expiry_days=(30,))).iloc[0]
        assert r["net_debit"] < 0
        assert r["breakeven"] < 100.0

    def test_gap_pushes_legs_out(self):
        r = oc.collar(_synthetic_chain(underlying=100.0, expiry_days=(30,)), gap=2).iloc[0]
        assert r["put_strike"] == 94.0
        assert r["call_strike"] == 106.0

    def test_one_row_per_expiry_sorted(self):
        out = oc.collar(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["tte"]) == sorted(out["tte"])

    def test_returns_expected_columns(self):
        out = oc.collar(_synthetic_chain())
        for col in (
            "expiry",
            "tte",
            "put_strike",
            "call_strike",
            "underlying_price",
            "net_debit",
            "max_profit",
            "max_loss",
            "breakeven",
        ):
            assert col in out.columns

    def test_bad_gap_raises(self):
        with pytest.raises(ValueError):
            oc.collar(_synthetic_chain(), gap=0)

    def test_empty_chain_returns_empty_frame(self):
        empty = pd.DataFrame(columns=["expiry", "strike", "kind", "underlying_price", "mid"])
        out = oc.collar(empty)
        assert out.empty
        assert "net_debit" in out.columns


class TestMaxPain:
    def test_symmetric_uniform_oi_pins_center(self):
        # 85..115 strikes, equal OI everywhere, spot 100 -> center strike wins
        out = oc.max_pain(_synthetic_chain(underlying=100.0, expiry_days=(30,)))
        assert out["max_pain_strike"].iloc[0] == pytest.approx(100.0)

    def test_heavy_itm_call_oi_pulls_strike_down(self):
        # call OI piled deep ITM at 90 outweighs a put cluster at 110, so the
        # writers' least-pain settlement drops to 90. Hand-checked: pain(90)=2000,
        # pain(100)=4000, pain(110)=6000.
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 90.0,
                    "kind": "call",
                    "open_interest": 300,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 110.0,
                    "kind": "put",
                    "open_interest": 100,
                    "underlying_price": 100.0,
                },
            ]
        )
        out = oc.max_pain(chain)
        r = out.iloc[0]
        assert r["max_pain_strike"] == pytest.approx(90.0)
        assert r["total_oi"] == pytest.approx(400.0)
        assert r["pain_at_max_pain"] == pytest.approx(2000.0)

    def test_one_row_per_expiry_sorted(self):
        out = oc.max_pain(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_returns_expected_columns(self):
        out = oc.max_pain(_synthetic_chain(expiry_days=(30,)))
        for col in (
            "expiry",
            "underlying_price",
            "max_pain_strike",
            "total_oi",
            "pain_at_max_pain",
        ):
            assert col in out.columns

    def test_no_open_interest_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["open_interest"])
        out = oc.max_pain(chain)
        assert out.empty
        assert "max_pain_strike" in out.columns

    def test_zero_oi_expiry_skipped(self):
        chain = _synthetic_chain(expiry_days=(30,))
        chain["open_interest"] = 0
        out = oc.max_pain(chain)
        assert out.empty


class TestMaxPainCurve:
    def _two_strike_chain(self):
        # same hand-checked chain as TestMaxPain: pain(90)=2000, pain(110)=6000
        return pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 90.0,
                    "kind": "call",
                    "open_interest": 300,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 110.0,
                    "kind": "put",
                    "open_interest": 100,
                    "underlying_price": 100.0,
                },
            ]
        )

    def test_curve_matches_max_pain_minimum(self):
        chain = self._two_strike_chain()
        curve = oc.max_pain_curve(chain)
        ref = oc.max_pain(chain).iloc[0]
        pinned = curve[curve["is_max_pain"]]
        assert len(pinned) == 1
        assert pinned["strike"].iloc[0] == pytest.approx(ref["max_pain_strike"])
        assert pinned["total_pain"].iloc[0] == pytest.approx(ref["pain_at_max_pain"])
        assert curve["total_pain"].min() == pytest.approx(2000.0)
        assert curve["total_pain"].max() == pytest.approx(6000.0)

    def test_call_put_split_sums_to_total(self):
        curve = oc.max_pain_curve(self._two_strike_chain())
        assert (curve["call_pain"] + curve["put_pain"] == curve["total_pain"]).all()

    def test_one_row_per_strike_sorted(self):
        out = oc.max_pain_curve(_synthetic_chain(expiry_days=(60, 30)))
        assert list(out["expiry"]) == sorted(out["expiry"])
        for _, grp in out.groupby("expiry"):
            assert list(grp["strike"]) == sorted(grp["strike"])
            assert grp["is_max_pain"].sum() == 1

    def test_returns_expected_columns(self):
        out = oc.max_pain_curve(_synthetic_chain(expiry_days=(30,)))
        for col in (
            "expiry",
            "underlying_price",
            "strike",
            "call_pain",
            "put_pain",
            "total_pain",
            "is_max_pain",
        ):
            assert col in out.columns

    def test_no_open_interest_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["open_interest"])
        out = oc.max_pain_curve(chain)
        assert out.empty
        assert "total_pain" in out.columns


class TestPCR:
    def test_uniform_chain_is_one(self):
        # synthetic chain has equal call/put OI and volume everywhere
        out = oc.pcr(_synthetic_chain(expiry_days=(30,)))
        r = out.iloc[0]
        assert r["oi_pcr"] == pytest.approx(1.0)
        assert r["volume_pcr"] == pytest.approx(1.0)

    def test_more_puts_than_calls(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "open_interest": 100,
                    "volume": 10,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "put",
                    "open_interest": 300,
                    "volume": 40,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.pcr(chain).iloc[0]
        assert r["put_oi"] == pytest.approx(300.0)
        assert r["call_oi"] == pytest.approx(100.0)
        assert r["oi_pcr"] == pytest.approx(3.0)
        assert r["volume_pcr"] == pytest.approx(4.0)

    def test_zero_call_oi_gives_nan_ratio(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "put",
                    "open_interest": 200,
                    "volume": 5,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.pcr(chain).iloc[0]
        assert r["put_oi"] == pytest.approx(200.0)
        assert np.isnan(r["oi_pcr"])

    def test_missing_volume_column_keeps_oi_ratio(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["volume"])
        r = oc.pcr(chain).iloc[0]
        assert r["oi_pcr"] == pytest.approx(1.0)
        assert np.isnan(r["volume_pcr"])

    def test_one_row_per_expiry_sorted(self):
        out = oc.pcr(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_no_open_interest_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["open_interest"])
        out = oc.pcr(chain)
        assert out.empty
        assert "oi_pcr" in out.columns


class TestPCRByStrike:
    def test_collapses_expiries_onto_each_strike(self):
        # two expiries, same two strikes; per-strike OI sums across both
        chain = _synthetic_chain(expiry_days=(30, 60))
        out = oc.pcr_by_strike(chain)
        per_strike = _synthetic_chain(expiry_days=(30,))
        assert len(out) == per_strike["strike"].nunique()
        assert list(out["strike"]) == sorted(out["strike"])

    def test_more_puts_than_calls_at_a_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 90.0,
                    "kind": "call",
                    "open_interest": 50,
                    "volume": 5,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-08-01",
                    "strike": 90.0,
                    "kind": "put",
                    "open_interest": 100,
                    "volume": 30,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-08-01",
                    "strike": 90.0,
                    "kind": "put",
                    "open_interest": 50,
                    "volume": 10,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.pcr_by_strike(chain).iloc[0]
        assert r["strike"] == pytest.approx(90.0)
        assert r["put_oi"] == pytest.approx(150.0)
        assert r["call_oi"] == pytest.approx(50.0)
        assert r["oi_pcr"] == pytest.approx(3.0)
        assert r["volume_pcr"] == pytest.approx(8.0)

    def test_zero_call_oi_gives_nan_ratio(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 80.0,
                    "kind": "put",
                    "open_interest": 200,
                    "volume": 5,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.pcr_by_strike(chain).iloc[0]
        assert r["put_oi"] == pytest.approx(200.0)
        assert np.isnan(r["oi_pcr"])

    def test_missing_volume_column_keeps_oi_ratio(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["volume"])
        out = oc.pcr_by_strike(chain)
        assert list(out["oi_pcr"]) == pytest.approx([1.0] * len(out))
        assert out["volume_pcr"].isna().all()

    def test_no_open_interest_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["open_interest"])
        out = oc.pcr_by_strike(chain)
        assert out.empty
        assert "oi_pcr" in out.columns


class TestTurnover:
    def test_volume_over_oi_each_side(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "open_interest": 200,
                    "volume": 100,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "put",
                    "open_interest": 50,
                    "volume": 75,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.turnover(chain).iloc[0]
        assert r["call_turnover"] == pytest.approx(0.5)
        assert r["put_turnover"] == pytest.approx(1.5)

    def test_zero_oi_gives_nan_turnover(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "open_interest": 0,
                    "volume": 30,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.turnover(chain).iloc[0]
        assert r["call_volume"] == pytest.approx(30.0)
        assert np.isnan(r["call_turnover"])

    def test_missing_volume_column_gives_nan(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["volume"])
        r = oc.turnover(chain).iloc[0]
        assert r["call_oi"] > 0
        assert np.isnan(r["call_turnover"])

    def test_one_row_per_expiry_sorted(self):
        out = oc.turnover(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_no_open_interest_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["open_interest"])
        out = oc.turnover(chain)
        assert out.empty
        assert "call_turnover" in out.columns


class TestTurnoverByStrike:
    def test_collapses_expiries_onto_each_strike(self):
        chain = _synthetic_chain(expiry_days=(30, 60))
        out = oc.turnover_by_strike(chain)
        per_strike = _synthetic_chain(expiry_days=(30,))
        assert len(out) == per_strike["strike"].nunique()
        assert list(out["strike"]) == sorted(out["strike"])

    def test_volume_over_oi_each_side_at_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "open_interest": 200,
                    "volume": 100,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-08-01",
                    "strike": 100.0,
                    "kind": "put",
                    "open_interest": 30,
                    "volume": 30,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-08-01",
                    "strike": 100.0,
                    "kind": "put",
                    "open_interest": 20,
                    "volume": 45,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.turnover_by_strike(chain).iloc[0]
        assert r["strike"] == pytest.approx(100.0)
        assert r["call_turnover"] == pytest.approx(0.5)
        assert r["put_oi"] == pytest.approx(50.0)
        assert r["put_turnover"] == pytest.approx(1.5)

    def test_zero_oi_gives_nan_turnover(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 80.0,
                    "kind": "call",
                    "open_interest": 0,
                    "volume": 30,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.turnover_by_strike(chain).iloc[0]
        assert r["call_volume"] == pytest.approx(30.0)
        assert np.isnan(r["call_turnover"])

    def test_missing_volume_column_gives_nan(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["volume"])
        out = oc.turnover_by_strike(chain)
        assert (out["call_oi"] > 0).any()
        assert out["call_turnover"].isna().all()

    def test_no_open_interest_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["open_interest"])
        out = oc.turnover_by_strike(chain)
        assert out.empty
        assert "call_turnover" in out.columns


class TestDollarVolume:
    def test_premium_weighted_each_side(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "mid": 2.0,
                    "open_interest": 200,
                    "volume": 100,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "put",
                    "mid": 4.0,
                    "open_interest": 50,
                    "volume": 75,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.dollar_volume(chain).iloc[0]
        # call: 2 * 100 * 100 = 20000, put: 4 * 75 * 100 = 30000
        assert r["call_dollar_volume"] == pytest.approx(20000.0)
        assert r["put_dollar_volume"] == pytest.approx(30000.0)
        assert r["dollar_volume_pcr"] == pytest.approx(1.5)
        # oi: call 2*200*100=40000, put 4*50*100=20000
        assert r["call_dollar_oi"] == pytest.approx(40000.0)
        assert r["put_dollar_oi"] == pytest.approx(20000.0)
        assert r["dollar_oi_pcr"] == pytest.approx(0.5)

    def test_dollar_pcr_differs_from_count_pcr(self):
        # one expensive put outweighs many cheap calls in dollars but not in count
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "mid": 0.5,
                    "open_interest": 10,
                    "volume": 1000,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "put",
                    "mid": 20.0,
                    "open_interest": 10,
                    "volume": 100,
                    "underlying_price": 100.0,
                },
            ]
        )
        count = oc.pcr(chain).iloc[0]
        dollars = oc.dollar_volume(chain).iloc[0]
        assert count["volume_pcr"] == pytest.approx(0.1)
        assert dollars["dollar_volume_pcr"] == pytest.approx(4.0)

    def test_contract_size_scales(self):
        chain = _synthetic_chain(expiry_days=(30,))
        base = oc.dollar_volume(chain).iloc[0]["call_dollar_volume"]
        scaled = oc.dollar_volume(chain, contract_size=50.0).iloc[0]["call_dollar_volume"]
        assert scaled == pytest.approx(base / 2.0)

    def test_missing_volume_column_gives_nan(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["volume"])
        r = oc.dollar_volume(chain).iloc[0]
        assert r["call_dollar_oi"] > 0
        assert np.isnan(r["call_dollar_volume"])

    def test_zero_call_side_gives_nan_pcr(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "put",
                    "mid": 3.0,
                    "open_interest": 50,
                    "volume": 40,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.dollar_volume(chain).iloc[0]
        assert r["put_dollar_volume"] == pytest.approx(12000.0)
        assert np.isnan(r["dollar_volume_pcr"])

    def test_one_row_per_expiry_sorted(self):
        out = oc.dollar_volume(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_missing_price_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["mid"])
        out = oc.dollar_volume(chain)
        assert out.empty
        assert "dollar_volume_pcr" in out.columns


class TestDollarVolumeByStrike:
    def test_premium_weighted_per_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "mid": 2.0,
                    "open_interest": 200,
                    "volume": 100,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "put",
                    "mid": 4.0,
                    "open_interest": 50,
                    "volume": 75,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.dollar_volume_by_strike(chain)
        assert list(r["strike"]) == [100.0]
        row = r.iloc[0]
        # call: 2 * 100 * 100 = 20000, put: 4 * 75 * 100 = 30000
        assert row["call_dollar_volume"] == pytest.approx(20000.0)
        assert row["put_dollar_volume"] == pytest.approx(30000.0)
        assert row["dollar_volume_pcr"] == pytest.approx(1.5)
        assert row["call_dollar_oi"] == pytest.approx(40000.0)
        assert row["put_dollar_oi"] == pytest.approx(20000.0)
        assert row["dollar_oi_pcr"] == pytest.approx(0.5)

    def test_collapses_expiries_into_one_strike_row(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "mid": 2.0,
                    "open_interest": 100,
                    "volume": 10,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-08-01",
                    "strike": 100.0,
                    "kind": "call",
                    "mid": 3.0,
                    "open_interest": 100,
                    "volume": 10,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.dollar_volume_by_strike(chain)
        assert len(r) == 1
        # both expiries fold in: (2+3) * 100 * 100
        assert r.iloc[0]["call_dollar_oi"] == pytest.approx(50000.0)

    def test_contract_size_scales(self):
        chain = _synthetic_chain(expiry_days=(30,))
        base = oc.dollar_volume_by_strike(chain).iloc[0]["call_dollar_oi"]
        scaled = oc.dollar_volume_by_strike(chain, contract_size=50.0).iloc[0]["call_dollar_oi"]
        assert scaled == pytest.approx(base / 2.0)

    def test_missing_volume_column_gives_nan(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["volume"])
        r = oc.dollar_volume_by_strike(chain).iloc[0]
        assert r["call_dollar_oi"] > 0
        assert np.isnan(r["call_dollar_volume"])

    def test_zero_call_side_gives_nan_pcr(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 95.0,
                    "kind": "put",
                    "mid": 3.0,
                    "open_interest": 50,
                    "volume": 40,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.dollar_volume_by_strike(chain).iloc[0]
        assert r["put_dollar_oi"] == pytest.approx(15000.0)
        assert np.isnan(r["dollar_oi_pcr"])

    def test_one_row_per_strike_sorted(self):
        out = oc.dollar_volume_by_strike(_synthetic_chain(expiry_days=(30,)))
        assert list(out["strike"]) == sorted(out["strike"])
        assert out["strike"].is_unique

    def test_missing_price_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["mid"])
        out = oc.dollar_volume_by_strike(chain)
        assert out.empty
        assert "dollar_oi_pcr" in out.columns


class TestLiquidity:
    def test_relative_spread_from_explicit_quotes(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "bid": 1.90,
                    "ask": 2.10,
                    "mid": 2.0,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "bid": 0.95,
                    "ask": 1.05,
                    "mid": 1.0,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.liquidity(chain).iloc[0]
        assert r["n_quotes"] == 2
        # spreads 0.20 and 0.10 -> median 0.15; rel 0.10 and 0.10 -> median 0.10
        assert r["median_spread"] == pytest.approx(0.15)
        assert r["median_rel_spread"] == pytest.approx(0.10)
        assert r["max_rel_spread"] == pytest.approx(0.10)

    def test_widest_strike_shows_in_max_not_median(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "bid": 1.98,
                    "ask": 2.02,
                    "mid": 2.0,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "bid": 1.98,
                    "ask": 2.02,
                    "mid": 2.0,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 110.0,
                    "kind": "call",
                    "bid": 0.10,
                    "ask": 0.50,
                    "mid": 0.30,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.liquidity(chain).iloc[0]
        assert r["median_rel_spread"] == pytest.approx(0.02)
        # the wide strike: 0.40 / 0.30
        assert r["max_rel_spread"] == pytest.approx(0.40 / 0.30)

    def test_falls_back_to_midpoint_when_no_mid_column(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "bid": 1.0,
                    "ask": 3.0,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.liquidity(chain).iloc[0]
        # midpoint 2.0, spread 2.0 -> rel 1.0
        assert r["median_rel_spread"] == pytest.approx(1.0)

    def test_crossed_and_nonpositive_quotes_dropped(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "bid": 1.9,
                    "ask": 2.1,
                    "mid": 2.0,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "bid": 2.0,
                    "ask": 1.0,
                    "mid": 1.5,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 110.0,
                    "kind": "call",
                    "bid": 0.0,
                    "ask": 0.0,
                    "mid": 0.0,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.liquidity(chain).iloc[0]
        assert r["n_quotes"] == 1
        assert r["median_spread"] == pytest.approx(0.2)

    def test_one_row_per_expiry_sorted(self):
        out = oc.liquidity(_synthetic_chain(expiry_days=(120, 30, 60), spread_bps=100.0))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])
        assert (out["median_rel_spread"] > 0).all()

    def test_missing_bid_ask_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["bid", "ask"])
        out = oc.liquidity(chain)
        assert out.empty
        assert "median_rel_spread" in out.columns


class TestLiquidityByStrike:
    def test_one_row_per_quote_with_spreads(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "bid": 1.90,
                    "ask": 2.10,
                    "mid": 2.0,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "put",
                    "bid": 0.95,
                    "ask": 1.05,
                    "mid": 1.0,
                    "underlying_price": 100.0,
                },
            ]
        )
        out = oc.liquidity_by_strike(chain)
        assert len(out) == 2
        assert list(out.columns) == [
            "expiry",
            "strike",
            "kind",
            "bid",
            "ask",
            "mid",
            "spread",
            "rel_spread",
        ]
        first = out.iloc[0]
        assert first["strike"] == 100.0
        assert first["spread"] == pytest.approx(0.20)
        assert first["rel_spread"] == pytest.approx(0.10)
        second = out.iloc[1]
        assert second["kind"] == "put"
        assert second["spread"] == pytest.approx(0.10)
        assert second["rel_spread"] == pytest.approx(0.10)

    def test_sorted_by_expiry_then_strike(self):
        out = oc.liquidity_by_strike(
            _synthetic_chain(expiry_days=(60, 30), n_strikes=5, spread_bps=100.0)
        )
        keys = list(zip(out["expiry"], out["strike"]))
        assert keys == sorted(keys)

    def test_falls_back_to_midpoint_when_no_mid_column(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "bid": 1.0,
                    "ask": 3.0,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.liquidity_by_strike(chain).iloc[0]
        assert r["mid"] == pytest.approx(2.0)
        assert r["rel_spread"] == pytest.approx(1.0)

    def test_crossed_and_nonpositive_quotes_dropped(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "bid": 1.9,
                    "ask": 2.1,
                    "mid": 2.0,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "bid": 2.0,
                    "ask": 1.0,
                    "mid": 1.5,
                    "underlying_price": 100.0,
                },
            ]
        )
        out = oc.liquidity_by_strike(chain)
        assert len(out) == 1
        assert out.iloc[0]["strike"] == 100.0

    def test_missing_columns_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["bid", "ask"])
        out = oc.liquidity_by_strike(chain)
        assert out.empty
        assert "rel_spread" in out.columns


class TestOIWalls:
    def test_picks_highest_oi_strike_each_side(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "open_interest": 800,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 110.0,
                    "kind": "call",
                    "open_interest": 200,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 95.0,
                    "kind": "put",
                    "open_interest": 100,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 90.0,
                    "kind": "put",
                    "open_interest": 600,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.oi_walls(chain).iloc[0]
        assert r["call_wall"] == pytest.approx(105.0)
        assert r["call_wall_oi"] == pytest.approx(800.0)
        assert r["put_wall"] == pytest.approx(90.0)
        assert r["put_wall_oi"] == pytest.approx(600.0)

    def test_sums_split_rows_at_same_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "open_interest": 300,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "open_interest": 300,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "open_interest": 500,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.oi_walls(chain).iloc[0]
        assert r["call_wall"] == pytest.approx(100.0)
        assert r["call_wall_oi"] == pytest.approx(600.0)

    def test_tie_breaks_to_lower_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 110.0,
                    "kind": "call",
                    "open_interest": 400,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "open_interest": 400,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.oi_walls(chain).iloc[0]
        assert r["call_wall"] == pytest.approx(105.0)

    def test_missing_side_gives_nan_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "open_interest": 250,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.oi_walls(chain).iloc[0]
        assert r["call_wall"] == pytest.approx(100.0)
        assert np.isnan(r["put_wall"])
        assert r["put_wall_oi"] == pytest.approx(0.0)

    def test_one_row_per_expiry_sorted(self):
        out = oc.oi_walls(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_no_open_interest_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["open_interest"])
        out = oc.oi_walls(chain)
        assert out.empty
        assert "call_wall" in out.columns


class TestWallDistance:
    def _chain(self):
        return pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "open_interest": 800,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 90.0,
                    "kind": "put",
                    "open_interest": 600,
                    "underlying_price": 100.0,
                },
            ]
        )

    def test_signed_distance_to_each_wall(self):
        r = oc.wall_distance(self._chain()).iloc[0]
        assert r["call_wall"] == pytest.approx(105.0)
        assert r["call_wall_dist_pct"] == pytest.approx(5.0)
        assert r["put_wall"] == pytest.approx(90.0)
        assert r["put_wall_dist_pct"] == pytest.approx(-10.0)

    def test_missing_side_gives_nan_distance(self):
        chain = self._chain()
        chain = chain[chain["kind"] == "call"]
        r = oc.wall_distance(chain).iloc[0]
        assert r["call_wall_dist_pct"] == pytest.approx(5.0)
        assert np.isnan(r["put_wall"])
        assert np.isnan(r["put_wall_dist_pct"])

    def test_one_row_per_expiry_sorted(self):
        out = oc.wall_distance(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_no_open_interest_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["open_interest"])
        out = oc.wall_distance(chain)
        assert out.empty
        assert "call_wall_dist_pct" in out.columns


class TestOIProfile:
    def test_call_put_split_and_net_per_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "open_interest": 700,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "put",
                    "open_interest": 200,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 95.0,
                    "kind": "put",
                    "open_interest": 400,
                    "underlying_price": 100.0,
                },
            ]
        )
        out = oc.oi_profile(chain)
        assert list(out["strike"]) == [95.0, 100.0]
        at100 = out[out["strike"] == 100.0].iloc[0]
        assert at100["call_oi"] == pytest.approx(700.0)
        assert at100["put_oi"] == pytest.approx(200.0)
        assert at100["total_oi"] == pytest.approx(900.0)
        assert at100["net_oi"] == pytest.approx(500.0)
        at95 = out[out["strike"] == 95.0].iloc[0]
        assert at95["call_oi"] == pytest.approx(0.0)
        assert at95["net_oi"] == pytest.approx(-400.0)

    def test_sums_split_rows_at_same_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "open_interest": 300,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "open_interest": 250,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.oi_profile(chain).iloc[0]
        assert r["call_oi"] == pytest.approx(550.0)

    def test_walls_match_profile_peak(self):
        chain = _synthetic_chain(expiry_days=(30,))
        prof = oc.oi_profile(chain)
        walls = oc.oi_walls(chain).iloc[0]
        exp = walls["expiry"]
        sub = prof[prof["expiry"] == exp]
        top_call = sub.loc[sub["call_oi"].idxmax()]
        assert top_call["strike"] == pytest.approx(walls["call_wall"])
        assert top_call["call_oi"] == pytest.approx(walls["call_wall_oi"])

    def test_rows_sorted_by_expiry_then_strike(self):
        out = oc.oi_profile(_synthetic_chain(expiry_days=(60, 30)))
        keys = list(zip(out["expiry"], out["strike"]))
        assert keys == sorted(keys)

    def test_no_open_interest_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["open_interest"])
        out = oc.oi_profile(chain)
        assert out.empty
        assert "net_oi" in out.columns


class TestVolumeProfile:
    def test_call_put_split_and_net_per_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "volume": 700,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "put",
                    "volume": 200,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 95.0,
                    "kind": "put",
                    "volume": 400,
                    "underlying_price": 100.0,
                },
            ]
        )
        out = oc.volume_profile(chain)
        assert list(out["strike"]) == [95.0, 100.0]
        at100 = out[out["strike"] == 100.0].iloc[0]
        assert at100["call_volume"] == pytest.approx(700.0)
        assert at100["put_volume"] == pytest.approx(200.0)
        assert at100["total_volume"] == pytest.approx(900.0)
        assert at100["net_volume"] == pytest.approx(500.0)
        at95 = out[out["strike"] == 95.0].iloc[0]
        assert at95["call_volume"] == pytest.approx(0.0)
        assert at95["net_volume"] == pytest.approx(-400.0)

    def test_sums_split_rows_at_same_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "volume": 300,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "volume": 250,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.volume_profile(chain).iloc[0]
        assert r["call_volume"] == pytest.approx(550.0)

    def test_rows_sorted_by_expiry_then_strike(self):
        out = oc.volume_profile(_synthetic_chain(expiry_days=(60, 30)))
        keys = list(zip(out["expiry"], out["strike"]))
        assert keys == sorted(keys)

    def test_no_volume_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["volume"])
        out = oc.volume_profile(chain)
        assert out.empty
        assert "net_volume" in out.columns


class TestVolumeWalls:
    def test_picks_highest_volume_strike_each_side(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "volume": 800,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 110.0,
                    "kind": "call",
                    "volume": 200,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 95.0,
                    "kind": "put",
                    "volume": 100,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 90.0,
                    "kind": "put",
                    "volume": 600,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.volume_walls(chain).iloc[0]
        assert r["call_wall"] == pytest.approx(105.0)
        assert r["call_wall_volume"] == pytest.approx(800.0)
        assert r["put_wall"] == pytest.approx(90.0)
        assert r["put_wall_volume"] == pytest.approx(600.0)

    def test_sums_split_rows_at_same_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "volume": 300,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "volume": 300,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "volume": 500,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.volume_walls(chain).iloc[0]
        assert r["call_wall"] == pytest.approx(100.0)
        assert r["call_wall_volume"] == pytest.approx(600.0)

    def test_tie_breaks_to_lower_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 110.0,
                    "kind": "call",
                    "volume": 400,
                    "underlying_price": 100.0,
                },
                {
                    "expiry": "2026-07-01",
                    "strike": 105.0,
                    "kind": "call",
                    "volume": 400,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.volume_walls(chain).iloc[0]
        assert r["call_wall"] == pytest.approx(105.0)

    def test_missing_side_gives_nan_strike(self):
        chain = pd.DataFrame(
            [
                {
                    "expiry": "2026-07-01",
                    "strike": 100.0,
                    "kind": "call",
                    "volume": 250,
                    "underlying_price": 100.0,
                },
            ]
        )
        r = oc.volume_walls(chain).iloc[0]
        assert r["call_wall"] == pytest.approx(100.0)
        assert np.isnan(r["put_wall"])
        assert r["put_wall_volume"] == pytest.approx(0.0)

    def test_one_row_per_expiry_sorted(self):
        out = oc.volume_walls(_synthetic_chain(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_no_volume_column_returns_empty(self):
        chain = _synthetic_chain(expiry_days=(30,)).drop(columns=["volume"])
        out = oc.volume_walls(chain)
        assert out.empty
        assert "call_wall" in out.columns


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


def test_iv_skew_in_public_api():
    assert hasattr(oc, "iv_skew")
    assert callable(oc.iv_skew)


def test_rr_bf_in_public_api():
    assert hasattr(oc, "rr_bf")
    assert callable(oc.rr_bf)


def test_straddle_in_public_api():
    assert hasattr(oc, "straddle")
    assert callable(oc.straddle)


def test_strangle_in_public_api():
    assert hasattr(oc, "strangle")
    assert callable(oc.strangle)


class TestDeltaExposure:
    def _enriched(self, **kw):
        return oc.enrich(_synthetic_chain(**kw), rate=0.05)

    def test_returns_expected_columns(self):
        out = oc.delta_exposure(self._enriched(expiry_days=(30,)))
        for col in (
            "expiry",
            "underlying_price",
            "call_dex",
            "put_dex",
            "net_dex",
            "delta_wall_strike",
        ):
            assert col in out.columns

    def test_one_row_per_expiry_sorted(self):
        out = oc.delta_exposure(self._enriched(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_long_call_short_put_both_positive(self):
        # call delta is positive and put delta negative; under the long-call /
        # short-put convention both legs add positive dollar delta, so the book
        # reads net long.
        r = oc.delta_exposure(self._enriched(expiry_days=(30,))).iloc[0]
        assert r["call_dex"] > 0
        assert r["put_dex"] > 0
        assert r["net_dex"] == pytest.approx(r["call_dex"] + r["put_dex"])

    def test_call_heavy_book_lifts_net(self):
        chain = self._enriched(expiry_days=(30,))
        base = oc.delta_exposure(chain).iloc[0]["net_dex"]
        chain.loc[chain["kind"] == "call", "open_interest"] *= 3
        assert oc.delta_exposure(chain).iloc[0]["net_dex"] > base

    def test_delta_wall_within_grid(self):
        chain = self._enriched(underlying=100.0, expiry_days=(30,), n_strikes=11)
        r = oc.delta_exposure(chain).iloc[0]
        assert 85.0 <= r["delta_wall_strike"] <= 115.0

    def test_scales_with_contract_size(self):
        chain = self._enriched(expiry_days=(30,))
        base = oc.delta_exposure(chain, contract_size=100.0).iloc[0]["call_dex"]
        doubled = oc.delta_exposure(chain, contract_size=200.0).iloc[0]["call_dex"]
        assert doubled == pytest.approx(2.0 * base)

    def test_no_delta_column_returns_empty(self):
        out = oc.delta_exposure(_synthetic_chain(expiry_days=(30,)))
        assert out.empty
        assert "delta_wall_strike" in out.columns

    def test_zero_oi_expiry_skipped(self):
        chain = self._enriched(expiry_days=(30,))
        chain["open_interest"] = 0
        assert oc.delta_exposure(chain).empty


class TestDeltaExposureByStrike:
    def _enriched(self, **kw):
        return oc.enrich(_synthetic_chain(**kw), rate=0.05)

    def test_returns_expected_columns(self):
        out = oc.delta_exposure_by_strike(self._enriched(expiry_days=(30,)))
        for col in (
            "expiry",
            "underlying_price",
            "strike",
            "call_dex",
            "put_dex",
            "net_dex",
            "cumulative_net_dex",
            "is_delta_wall",
        ):
            assert col in out.columns

    def test_one_row_per_strike_sorted(self):
        out = oc.delta_exposure_by_strike(self._enriched(expiry_days=(30,), n_strikes=11))
        assert len(out) == 11
        assert list(out["strike"]) == sorted(out["strike"])

    def test_rows_sorted_by_expiry_then_strike(self):
        out = oc.delta_exposure_by_strike(self._enriched(expiry_days=(60, 30), n_strikes=5))
        assert len(out) == 10
        for _, grp in out.groupby("expiry"):
            assert list(grp["strike"]) == sorted(grp["strike"])
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_net_sums_to_aggregate(self):
        # collapsing the profile back over strikes must reproduce delta_exposure
        chain = self._enriched(expiry_days=(30,))
        chain.loc[chain["kind"] == "call", "open_interest"] *= 3
        agg = oc.delta_exposure(chain).iloc[0]
        prof = oc.delta_exposure_by_strike(chain)
        assert prof["net_dex"].sum() == pytest.approx(agg["net_dex"])
        assert prof["call_dex"].sum() == pytest.approx(agg["call_dex"])
        assert prof["put_dex"].sum() == pytest.approx(agg["put_dex"])

    def test_cumulative_is_running_sum(self):
        prof = oc.delta_exposure_by_strike(self._enriched(expiry_days=(30,), n_strikes=11))
        assert list(prof["cumulative_net_dex"]) == pytest.approx(list(prof["net_dex"].cumsum()))
        assert prof["cumulative_net_dex"].iloc[-1] == pytest.approx(prof["net_dex"].sum())

    def test_one_wall_per_expiry_matches_aggregate(self):
        chain = self._enriched(underlying=100.0, expiry_days=(30,), n_strikes=11)
        prof = oc.delta_exposure_by_strike(chain)
        walls = prof[prof["is_delta_wall"]]
        assert len(walls) == 1
        agg_wall = oc.delta_exposure(chain).iloc[0]["delta_wall_strike"]
        assert walls.iloc[0]["strike"] == pytest.approx(agg_wall)

    def test_no_delta_column_returns_empty(self):
        out = oc.delta_exposure_by_strike(_synthetic_chain(expiry_days=(30,)))
        assert out.empty
        assert "cumulative_net_dex" in out.columns

    def test_zero_oi_expiry_skipped(self):
        chain = self._enriched(expiry_days=(30,))
        chain["open_interest"] = 0
        assert oc.delta_exposure_by_strike(chain).empty

    def test_in_public_api(self):
        assert hasattr(oc, "delta_exposure_by_strike")
        assert callable(oc.delta_exposure_by_strike)


class TestGammaExposure:
    def _enriched(self, **kw):
        return oc.enrich(_synthetic_chain(**kw), rate=0.05)

    def test_returns_expected_columns(self):
        out = oc.gamma_exposure(self._enriched(expiry_days=(30,)))
        for col in (
            "expiry",
            "underlying_price",
            "call_gex",
            "put_gex",
            "net_gex",
            "gamma_wall_strike",
        ):
            assert col in out.columns

    def test_one_row_per_expiry_sorted(self):
        out = oc.gamma_exposure(self._enriched(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_call_gex_positive_put_gex_negative(self):
        # gamma is positive for both sides, so the call leg adds and the put leg
        # subtracts under the long-call / short-put convention.
        r = oc.gamma_exposure(self._enriched(expiry_days=(30,))).iloc[0]
        assert r["call_gex"] > 0
        assert r["put_gex"] < 0

    def test_symmetric_book_nets_near_zero(self):
        # equal OI on both sides + identical per-strike gamma -> net cancels
        r = oc.gamma_exposure(self._enriched(expiry_days=(30,))).iloc[0]
        assert r["net_gex"] == pytest.approx(0.0, abs=abs(r["call_gex"]) * 1e-6)

    def test_call_heavy_book_has_positive_net(self):
        chain = self._enriched(expiry_days=(30,))
        chain.loc[chain["kind"] == "call", "open_interest"] *= 3
        r = oc.gamma_exposure(chain).iloc[0]
        assert r["net_gex"] > 0

    def test_gamma_wall_near_atm(self):
        # gamma peaks at-the-money, so the gross-gamma wall sits at the strike
        # closest to spot (100 with the 85..115 grid)
        chain = self._enriched(underlying=100.0, expiry_days=(30,), n_strikes=11)
        r = oc.gamma_exposure(chain).iloc[0]
        assert r["gamma_wall_strike"] == pytest.approx(100.0)

    def test_scales_with_contract_size(self):
        chain = self._enriched(expiry_days=(30,))
        base = oc.gamma_exposure(chain, contract_size=100.0).iloc[0]["call_gex"]
        doubled = oc.gamma_exposure(chain, contract_size=200.0).iloc[0]["call_gex"]
        assert doubled == pytest.approx(2.0 * base)

    def test_no_gamma_column_returns_empty(self):
        out = oc.gamma_exposure(_synthetic_chain(expiry_days=(30,)))
        assert out.empty
        assert "gamma_wall_strike" in out.columns

    def test_zero_oi_expiry_skipped(self):
        chain = self._enriched(expiry_days=(30,))
        chain["open_interest"] = 0
        out = oc.gamma_exposure(chain)
        assert out.empty


class TestGammaExposureByStrike:
    def _enriched(self, **kw):
        return oc.enrich(_synthetic_chain(**kw), rate=0.05)

    def test_returns_expected_columns(self):
        out = oc.gamma_exposure_by_strike(self._enriched(expiry_days=(30,)))
        for col in (
            "expiry",
            "underlying_price",
            "strike",
            "call_gex",
            "put_gex",
            "net_gex",
            "cumulative_net_gex",
            "is_gamma_wall",
        ):
            assert col in out.columns

    def test_one_row_per_strike_sorted(self):
        out = oc.gamma_exposure_by_strike(self._enriched(expiry_days=(30,), n_strikes=11))
        assert len(out) == 11
        assert list(out["strike"]) == sorted(out["strike"])

    def test_rows_sorted_by_expiry_then_strike(self):
        out = oc.gamma_exposure_by_strike(self._enriched(expiry_days=(60, 30), n_strikes=5))
        assert len(out) == 10
        for _, grp in out.groupby("expiry"):
            assert list(grp["strike"]) == sorted(grp["strike"])
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_net_sums_to_aggregate(self):
        # collapsing the profile back over strikes must reproduce gamma_exposure
        chain = self._enriched(expiry_days=(30,))
        chain.loc[chain["kind"] == "call", "open_interest"] *= 3
        agg = oc.gamma_exposure(chain).iloc[0]
        prof = oc.gamma_exposure_by_strike(chain)
        assert prof["net_gex"].sum() == pytest.approx(agg["net_gex"])
        assert prof["call_gex"].sum() == pytest.approx(agg["call_gex"])
        assert prof["put_gex"].sum() == pytest.approx(agg["put_gex"])

    def test_cumulative_is_running_sum(self):
        prof = oc.gamma_exposure_by_strike(self._enriched(expiry_days=(30,), n_strikes=11))
        assert list(prof["cumulative_net_gex"]) == pytest.approx(list(prof["net_gex"].cumsum()))
        assert prof["cumulative_net_gex"].iloc[-1] == pytest.approx(prof["net_gex"].sum())

    def test_one_wall_per_expiry_matches_aggregate(self):
        chain = self._enriched(underlying=100.0, expiry_days=(30,), n_strikes=11)
        prof = oc.gamma_exposure_by_strike(chain)
        walls = prof[prof["is_gamma_wall"]]
        assert len(walls) == 1
        agg_wall = oc.gamma_exposure(chain).iloc[0]["gamma_wall_strike"]
        assert walls.iloc[0]["strike"] == pytest.approx(agg_wall)

    def test_no_gamma_column_returns_empty(self):
        out = oc.gamma_exposure_by_strike(_synthetic_chain(expiry_days=(30,)))
        assert out.empty
        assert "cumulative_net_gex" in out.columns

    def test_zero_oi_expiry_skipped(self):
        chain = self._enriched(expiry_days=(30,))
        chain["open_interest"] = 0
        assert oc.gamma_exposure_by_strike(chain).empty

    def test_in_public_api(self):
        assert hasattr(oc, "gamma_exposure_by_strike")
        assert callable(oc.gamma_exposure_by_strike)


class TestVegaExposureByStrike:
    def _enriched(self, **kw):
        return oc.enrich(_synthetic_chain(**kw), rate=0.05)

    def test_returns_expected_columns(self):
        out = oc.vega_exposure_by_strike(self._enriched(expiry_days=(30,)))
        for col in (
            "expiry",
            "underlying_price",
            "strike",
            "call_vex",
            "put_vex",
            "net_vex",
            "cumulative_net_vex",
            "is_vega_wall",
        ):
            assert col in out.columns

    def test_one_row_per_strike_sorted(self):
        out = oc.vega_exposure_by_strike(self._enriched(expiry_days=(30,), n_strikes=11))
        assert len(out) == 11
        assert list(out["strike"]) == sorted(out["strike"])

    def test_rows_sorted_by_expiry_then_strike(self):
        out = oc.vega_exposure_by_strike(self._enriched(expiry_days=(60, 30), n_strikes=5))
        assert len(out) == 10
        for _, grp in out.groupby("expiry"):
            assert list(grp["strike"]) == sorted(grp["strike"])
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_net_sums_to_aggregate(self):
        # collapsing the profile back over strikes must reproduce vega_exposure
        chain = self._enriched(expiry_days=(30,))
        chain.loc[chain["kind"] == "call", "open_interest"] *= 3
        agg = oc.vega_exposure(chain).iloc[0]
        prof = oc.vega_exposure_by_strike(chain)
        assert prof["net_vex"].sum() == pytest.approx(agg["net_vex"])
        assert prof["call_vex"].sum() == pytest.approx(agg["call_vex"])
        assert prof["put_vex"].sum() == pytest.approx(agg["put_vex"])

    def test_call_positive_put_negative_per_strike(self):
        prof = oc.vega_exposure_by_strike(self._enriched(expiry_days=(30,)))
        assert (prof["call_vex"] >= 0).all()
        assert (prof["put_vex"] <= 0).all()

    def test_cumulative_is_running_sum(self):
        prof = oc.vega_exposure_by_strike(self._enriched(expiry_days=(30,), n_strikes=11))
        assert list(prof["cumulative_net_vex"]) == pytest.approx(list(prof["net_vex"].cumsum()))
        assert prof["cumulative_net_vex"].iloc[-1] == pytest.approx(prof["net_vex"].sum())

    def test_one_wall_per_expiry_matches_aggregate(self):
        chain = self._enriched(underlying=100.0, expiry_days=(30,), n_strikes=11)
        prof = oc.vega_exposure_by_strike(chain)
        walls = prof[prof["is_vega_wall"]]
        assert len(walls) == 1
        agg_wall = oc.vega_exposure(chain).iloc[0]["vega_wall_strike"]
        assert walls.iloc[0]["strike"] == pytest.approx(agg_wall)

    def test_no_vega_column_returns_empty(self):
        out = oc.vega_exposure_by_strike(_synthetic_chain(expiry_days=(30,)))
        assert out.empty
        assert "cumulative_net_vex" in out.columns

    def test_zero_oi_expiry_skipped(self):
        chain = self._enriched(expiry_days=(30,))
        chain["open_interest"] = 0
        assert oc.vega_exposure_by_strike(chain).empty

    def test_in_public_api(self):
        assert hasattr(oc, "vega_exposure_by_strike")
        assert callable(oc.vega_exposure_by_strike)


class TestVegaExposure:
    def _enriched(self, **kw):
        return oc.enrich(_synthetic_chain(**kw), rate=0.05)

    def test_returns_expected_columns(self):
        out = oc.vega_exposure(self._enriched(expiry_days=(30,)))
        for col in (
            "expiry",
            "underlying_price",
            "call_vex",
            "put_vex",
            "net_vex",
            "vega_wall_strike",
        ):
            assert col in out.columns

    def test_one_row_per_expiry_sorted(self):
        out = oc.vega_exposure(self._enriched(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_call_vex_positive_put_vex_negative(self):
        # vega is positive for both sides, so the call leg adds and the put leg
        # subtracts under the long-call / short-put convention.
        r = oc.vega_exposure(self._enriched(expiry_days=(30,))).iloc[0]
        assert r["call_vex"] > 0
        assert r["put_vex"] < 0

    def test_symmetric_book_nets_near_zero(self):
        r = oc.vega_exposure(self._enriched(expiry_days=(30,))).iloc[0]
        assert r["net_vex"] == pytest.approx(0.0, abs=abs(r["call_vex"]) * 1e-6)

    def test_call_heavy_book_has_positive_net(self):
        chain = self._enriched(expiry_days=(30,))
        chain.loc[chain["kind"] == "call", "open_interest"] *= 3
        r = oc.vega_exposure(chain).iloc[0]
        assert r["net_vex"] > 0

    def test_vega_wall_near_atm(self):
        # vega peaks at-the-money, so the gross-vega wall sits at the strike
        # closest to spot (100 with the 85..115 grid)
        chain = self._enriched(underlying=100.0, expiry_days=(30,), n_strikes=11)
        r = oc.vega_exposure(chain).iloc[0]
        assert r["vega_wall_strike"] == pytest.approx(100.0)

    def test_scales_with_contract_size(self):
        chain = self._enriched(expiry_days=(30,))
        base = oc.vega_exposure(chain, contract_size=100.0).iloc[0]["call_vex"]
        doubled = oc.vega_exposure(chain, contract_size=200.0).iloc[0]["call_vex"]
        assert doubled == pytest.approx(2.0 * base)

    def test_no_vega_column_returns_empty(self):
        out = oc.vega_exposure(_synthetic_chain(expiry_days=(30,)))
        assert out.empty
        assert "vega_wall_strike" in out.columns

    def test_zero_oi_expiry_skipped(self):
        chain = self._enriched(expiry_days=(30,))
        chain["open_interest"] = 0
        assert oc.vega_exposure(chain).empty

    def test_in_public_api(self):
        assert hasattr(oc, "vega_exposure")
        assert callable(oc.vega_exposure)


class TestThetaExposure:
    def _enriched(self, **kw):
        return oc.enrich(_synthetic_chain(**kw), rate=0.05)

    def test_returns_expected_columns(self):
        out = oc.theta_exposure(self._enriched(expiry_days=(30,)))
        for col in (
            "expiry",
            "underlying_price",
            "call_tex",
            "put_tex",
            "net_tex",
            "theta_wall_strike",
        ):
            assert col in out.columns

    def test_one_row_per_expiry_sorted(self):
        out = oc.theta_exposure(self._enriched(expiry_days=(120, 30, 60)))
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_call_tex_negative_put_tex_positive(self):
        # long options carry negative theta, so the long-call leg is negative and
        # the short-put leg flips it positive under the long-call / short-put rule.
        r = oc.theta_exposure(self._enriched(expiry_days=(30,))).iloc[0]
        assert r["call_tex"] < 0
        assert r["put_tex"] > 0

    def test_net_is_call_plus_put(self):
        r = oc.theta_exposure(self._enriched(expiry_days=(30,))).iloc[0]
        assert r["net_tex"] == pytest.approx(r["call_tex"] + r["put_tex"])

    def test_theta_wall_near_atm(self):
        chain = self._enriched(underlying=100.0, expiry_days=(30,), n_strikes=11)
        r = oc.theta_exposure(chain).iloc[0]
        assert r["theta_wall_strike"] == pytest.approx(100.0)

    def test_scales_with_contract_size(self):
        chain = self._enriched(expiry_days=(30,))
        base = oc.theta_exposure(chain, contract_size=100.0).iloc[0]["call_tex"]
        doubled = oc.theta_exposure(chain, contract_size=200.0).iloc[0]["call_tex"]
        assert doubled == pytest.approx(2.0 * base)

    def test_no_theta_column_returns_empty(self):
        out = oc.theta_exposure(_synthetic_chain(expiry_days=(30,)))
        assert out.empty
        assert "theta_wall_strike" in out.columns

    def test_zero_oi_expiry_skipped(self):
        chain = self._enriched(expiry_days=(30,))
        chain["open_interest"] = 0
        assert oc.theta_exposure(chain).empty

    def test_in_public_api(self):
        assert hasattr(oc, "theta_exposure")
        assert callable(oc.theta_exposure)


class TestThetaExposureByStrike:
    def _enriched(self, **kw):
        return oc.enrich(_synthetic_chain(**kw), rate=0.05)

    def test_returns_expected_columns(self):
        out = oc.theta_exposure_by_strike(self._enriched(expiry_days=(30,)))
        for col in (
            "expiry",
            "underlying_price",
            "strike",
            "call_tex",
            "put_tex",
            "net_tex",
            "cumulative_net_tex",
            "is_theta_wall",
        ):
            assert col in out.columns

    def test_one_row_per_strike_sorted(self):
        out = oc.theta_exposure_by_strike(self._enriched(expiry_days=(30,), n_strikes=11))
        assert len(out) == 11
        assert list(out["strike"]) == sorted(out["strike"])

    def test_rows_sorted_by_expiry_then_strike(self):
        out = oc.theta_exposure_by_strike(self._enriched(expiry_days=(60, 30), n_strikes=5))
        assert len(out) == 10
        for _, grp in out.groupby("expiry"):
            assert list(grp["strike"]) == sorted(grp["strike"])
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_net_sums_to_aggregate(self):
        # collapsing the profile back over strikes must reproduce theta_exposure
        chain = self._enriched(expiry_days=(30,))
        chain.loc[chain["kind"] == "call", "open_interest"] *= 3
        agg = oc.theta_exposure(chain).iloc[0]
        prof = oc.theta_exposure_by_strike(chain)
        assert prof["net_tex"].sum() == pytest.approx(agg["net_tex"])
        assert prof["call_tex"].sum() == pytest.approx(agg["call_tex"])
        assert prof["put_tex"].sum() == pytest.approx(agg["put_tex"])

    def test_call_negative_put_positive_in_aggregate(self):
        # per strike the sign can flip (deep ITM puts carry positive theta), but
        # summed across strikes the long-call leg stays negative and the short-put
        # leg positive, same as theta_exposure
        prof = oc.theta_exposure_by_strike(self._enriched(expiry_days=(30,)))
        assert prof["call_tex"].sum() < 0
        assert prof["put_tex"].sum() > 0

    def test_cumulative_is_running_sum(self):
        prof = oc.theta_exposure_by_strike(self._enriched(expiry_days=(30,), n_strikes=11))
        assert list(prof["cumulative_net_tex"]) == pytest.approx(list(prof["net_tex"].cumsum()))
        assert prof["cumulative_net_tex"].iloc[-1] == pytest.approx(prof["net_tex"].sum())

    def test_one_wall_per_expiry_matches_aggregate(self):
        chain = self._enriched(underlying=100.0, expiry_days=(30,), n_strikes=11)
        prof = oc.theta_exposure_by_strike(chain)
        walls = prof[prof["is_theta_wall"]]
        assert len(walls) == 1
        agg_wall = oc.theta_exposure(chain).iloc[0]["theta_wall_strike"]
        assert walls.iloc[0]["strike"] == pytest.approx(agg_wall)

    def test_no_theta_column_returns_empty(self):
        out = oc.theta_exposure_by_strike(_synthetic_chain(expiry_days=(30,)))
        assert out.empty
        assert "cumulative_net_tex" in out.columns

    def test_zero_oi_expiry_skipped(self):
        chain = self._enriched(expiry_days=(30,))
        chain["open_interest"] = 0
        assert oc.theta_exposure_by_strike(chain).empty

    def test_in_public_api(self):
        assert hasattr(oc, "theta_exposure_by_strike")
        assert callable(oc.theta_exposure_by_strike)


class TestGammaFlip:
    def _enriched(self, **kw):
        return oc.enrich(_synthetic_chain(**kw), rate=0.05)

    def _skewed(self, **kw):
        # puts piled low, calls piled high -> short gamma below, long gamma above,
        # so net GEX must cross zero somewhere in the middle.
        chain = self._enriched(underlying=100.0, n_strikes=11, **kw)
        calls = chain["kind"] == "call"
        puts = chain["kind"] == "put"
        chain.loc[calls, "open_interest"] = np.where(chain.loc[calls, "strike"] >= 100.0, 2000, 50)
        chain.loc[puts, "open_interest"] = np.where(chain.loc[puts, "strike"] <= 100.0, 2000, 50)
        return chain

    def test_returns_expected_columns(self):
        out = oc.gamma_flip(self._enriched(expiry_days=(30,)), rate=0.05)
        for col in (
            "expiry",
            "underlying_price",
            "net_gex",
            "flip_spot",
            "flip_distance_pct",
            "regime",
        ):
            assert col in out.columns

    def test_one_row_per_expiry_sorted(self):
        out = oc.gamma_flip(self._enriched(expiry_days=(120, 30, 60)), rate=0.05)
        assert len(out) == 3
        assert list(out["expiry"]) == sorted(out["expiry"])

    def test_symmetric_book_is_flat_no_flip(self):
        # equal call/put OI at every strike -> net cancels at all spots
        r = oc.gamma_flip(self._enriched(expiry_days=(30,)), rate=0.05).iloc[0]
        assert r["regime"] == "flat"
        assert np.isnan(r["flip_spot"])

    def test_skewed_book_has_flip_in_range(self):
        r = oc.gamma_flip(self._skewed(expiry_days=(30,)), rate=0.05).iloc[0]
        assert not np.isnan(r["flip_spot"])
        # crossing should land inside the +/-20% scan window
        assert 80.0 < r["flip_spot"] < 120.0

    def test_flip_distance_matches_spot(self):
        r = oc.gamma_flip(self._skewed(expiry_days=(30,)), rate=0.05).iloc[0]
        expected = (r["flip_spot"] / r["underlying_price"] - 1.0) * 100.0
        assert r["flip_distance_pct"] == pytest.approx(expected)

    def test_no_iv_column_returns_empty(self):
        out = oc.gamma_flip(_synthetic_chain(expiry_days=(30,)))
        assert out.empty
        assert "flip_spot" in out.columns

    def test_in_public_api(self):
        assert hasattr(oc, "gamma_flip")
        assert callable(oc.gamma_flip)


def test_max_pain_in_public_api():
    assert hasattr(oc, "max_pain")
    assert callable(oc.max_pain)


def test_pcr_in_public_api():
    assert hasattr(oc, "pcr")
    assert callable(oc.pcr)


def test_turnover_in_public_api():
    assert hasattr(oc, "turnover")
    assert callable(oc.turnover)


def test_liquidity_in_public_api():
    assert hasattr(oc, "liquidity")
    assert callable(oc.liquidity)


def test_liquidity_by_strike_in_public_api():
    assert hasattr(oc, "liquidity_by_strike")
    assert callable(oc.liquidity_by_strike)


def test_oi_walls_in_public_api():
    assert hasattr(oc, "oi_walls")
    assert callable(oc.oi_walls)


def test_oi_profile_in_public_api():
    assert hasattr(oc, "oi_profile")
    assert callable(oc.oi_profile)


def test_wall_distance_in_public_api():
    assert hasattr(oc, "wall_distance")
    assert callable(oc.wall_distance)


def test_volume_profile_in_public_api():
    assert hasattr(oc, "volume_profile")
    assert callable(oc.volume_profile)


def test_volume_walls_in_public_api():
    assert hasattr(oc, "volume_walls")
    assert callable(oc.volume_walls)


def test_delta_exposure_in_public_api():
    assert hasattr(oc, "delta_exposure")
    assert callable(oc.delta_exposure)


def test_gamma_exposure_in_public_api():
    assert hasattr(oc, "gamma_exposure")
    assert callable(oc.gamma_exposure)
