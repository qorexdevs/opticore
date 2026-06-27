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


if __name__ == "__main__":
    for name, fn in [
        ("scalar pricing / iv / greeks", scalars),
        ("vectorized over a strike range", vectorized),
        ("sample chain enrichment", sample_chain),
        ("strategy payoff as numbers", strategy),
        ("dealer gamma exposure", positioning),
    ]:
        print(f"\n# {name}")
        fn()
