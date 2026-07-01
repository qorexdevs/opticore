"""Runnable quickstart, no account or network needed.

    python examples/quickstart.py

Everything here uses the bundled synthetic SPY chain (provider="sample") and
prints numbers instead of plotting, so it works in plain CI too.
"""

import numpy as np
import opticore as oc


def scalars():
    price = oc.price(spot=100, strike=105, expiry=0.5, rate=0.05, vol=0.20, kind="call")
    iv = oc.iv(price=price, spot=100, strike=105, expiry=0.5, rate=0.05, kind="call")
    g = oc.greeks(spot=100, strike=105, expiry=0.5, rate=0.05, vol=0.20, kind="call")
    print(f"call price      {price:.4f}")
    print(f"implied vol     {iv:.4f}  (round-trips back to the 0.20 we priced with)")
    print(f"greeks          d={g.delta:.4f} g={g.gamma:.4f} t={g.theta:.4f} "
          f"v={g.vega:.4f} r={g.rho:.4f}")


def vectorized():
    strikes = np.arange(90, 111, dtype=float)
    prices = oc.price(spot=100, strike=strikes, expiry=0.5, rate=0.05, vol=0.20, kind="call")
    print(f"priced {len(prices)} strikes in one call, "
          f"90 -> {prices[0]:.2f}, 110 -> {prices[-1]:.2f}")


def sample_chain():
    chain = oc.fetch_chain(provider="sample", symbol="SPY")
    enriched = oc.enrich(chain, rate=0.045, div_yield=0.013)
    print(f"sample chain    {len(enriched)} rows, columns include "
          f"{[c for c in ('iv', 'delta', 'vega') if c in enriched.columns]}")


def strategy():
    legs = [
        oc.Leg("call", strike=105, qty=1, premium=3.50),
        oc.Leg("put", strike=95, qty=1, premium=2.10),
    ]
    p = oc.payoff_profile(legs)
    print(f"long strangle   net cost {p.net_cost:.2f}, breakevens "
          f"{[round(b, 1) for b in p.breakevens]}")


def positioning():
    chain = oc.fetch_chain(provider="sample", symbol="SPY")
    enriched = oc.enrich(chain, rate=0.045, div_yield=0.013)
    gex = oc.gamma_exposure(enriched)
    r = gex.iloc[0]
    lean = "dampening" if r["net_gex"] > 0 else "amplifying"
    print(f"gamma exposure  net {r['net_gex']:.3e} ({lean}), "
          f"wall at strike {r['gamma_wall_strike']:.0f}")

    flip = oc.gamma_flip(enriched, rate=0.045, div_yield=0.013).iloc[0]
    if flip["flip_spot"] == flip["flip_spot"]:  # not NaN
        print(f"gamma flip      {flip['regime']} at spot {flip['underlying_price']:.2f}, "
              f"flip near {flip['flip_spot']:.2f} ({flip['flip_distance_pct']:+.1f}%)")
    else:
        print(f"gamma flip      {flip['regime']}, no flip in scanned range")


def pinning():
    chain = oc.fetch_chain(provider="sample", symbol="SPY")
    enriched = oc.enrich(chain, rate=0.045, div_yield=0.013)

    mpd = oc.max_pain_distance(enriched).iloc[0]
    print(f"max pain        strike {mpd['max_pain_strike']:.0f}, "
          f"{mpd['dist_pct']:+.1f}% from spot {mpd['underlying_price']:.0f}")

    em = oc.expected_move(enriched).iloc[0]
    print(f"expected move   +/-{em['expected_move']:.1f} ({em['move_pct'] * 100:.1f}%), "
          f"1-sigma band {em['lower']:.0f} - {em['upper']:.0f}")

    gc = oc.gamma_concentration(enriched).iloc[0]
    print(f"gamma pinning   strike {gc['top_strike']:.0f} holds "
          f"{gc['top_share'] * 100:.0f}% of gross gamma, top 3 hold "
          f"{gc['top3_share'] * 100:.0f}%")


if __name__ == "__main__":
    for name, fn in [
        ("scalar pricing / iv / greeks", scalars),
        ("vectorized over a strike range", vectorized),
        ("sample chain enrichment", sample_chain),
        ("strategy payoff as numbers", strategy),
        ("dealer gamma exposure", positioning),
        ("where the chain pins and how far it can move", pinning),
    ]:
        print(f"\n# {name}")
        fn()
