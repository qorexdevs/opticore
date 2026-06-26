"""Tests for opticore.payoff_profile."""

import opticore as oc
import pytest


def test_long_call_breakeven_and_cost():
    p = oc.payoff_profile([oc.Leg("call", strike=100, qty=1, premium=4.0)])
    assert p.net_cost == pytest.approx(4.0)
    assert len(p.breakevens) == 1
    assert p.breakevens[0] == pytest.approx(104.0)
    assert p.max_loss == pytest.approx(-4.0)
    # long call profit is unbounded above, capped by the sampled grid
    assert p.max_profit > 0


def test_long_strangle_two_breakevens():
    legs = [
        oc.Leg("call", strike=105, qty=1, premium=3.50),
        oc.Leg("put", strike=95, qty=1, premium=2.10),
    ]
    p = oc.payoff_profile(legs)
    assert p.net_cost == pytest.approx(5.60)
    assert [round(b, 2) for b in p.breakevens] == [89.40, 110.60]


def test_short_put_credit_is_negative_cost():
    p = oc.payoff_profile([oc.Leg("put", strike=100, qty=-1, premium=3.0)])
    assert p.net_cost == pytest.approx(-3.0)
    # max profit equals the premium kept when spot stays above the strike
    assert p.max_profit == pytest.approx(3.0)
    assert p.breakevens[0] == pytest.approx(97.0)


def test_explicit_range_and_shape():
    p = oc.payoff_profile(
        [oc.Leg("call", strike=100, qty=1, premium=2.0)],
        spot_range=(80.0, 120.0),
        num_points=41,
    )
    assert p.spots.shape == (41,)
    assert p.pnl.shape == (41,)
    assert p.spots[0] == pytest.approx(80.0)
    assert p.spots[-1] == pytest.approx(120.0)


def test_empty_legs_raises():
    with pytest.raises(ValueError):
        oc.payoff_profile([])


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        oc.payoff_profile([oc.Leg("call", strike=100)], num_points=1)
