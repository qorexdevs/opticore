# ⚡ OptiCore

**High-performance options pricing, IV solver, and Greeks — C++20 core with a Pythonic API.**

[![Release](https://img.shields.io/github/v/release/qorexdevs/opticore?sort=semver)](https://github.com/qorexdevs/opticore/releases)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://github.com/qorexdevs/opticore/releases)
[![CI](https://github.com/qorexdevs/opticore/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/qorexdevs/opticore/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-pdoc-blue.svg)](https://qorexdevs.github.io/opticore/)
[![codecov](https://codecov.io/gh/qorexdevs/opticore/branch/main/graph/badge.svg)](https://codecov.io/gh/qorexdevs/opticore)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)]()

Read an option chain the way a dealer desk does: net gamma by strike, the wall
where hedging flow pins, and the spot that flips the regime.

![Dealer gamma exposure by strike](docs/assets/dealer-gamma-exposure.png)

```python
chain = oc.enrich(oc.fetch_chain(provider="sample", symbol="SPY"))
oc.plot.exposure_profile(chain, greek="gamma")
```

---

## Why OptiCore?

| | OptiCore | QuantLib | py_vollib | FinancePy |
|---|---------|----------|-----------|-----------|
| **Install** | Prebuilt wheels, no source build | Compile from source + SWIG | `pip install` | `pip install` |
| **Price 10k options** | 0.65 ms* | ~50 ms | 40 ms* | ~100 ms |
| **IV precision** | 64-bit machine ε | 1e-8 | 64-bit (with Numba) | 1e-6 |
| **API style** | `oc.price(spot=100, ...)` | 15 lines of boilerplate | Function-based | OOP |
| **Greeks in 1 call** | ✅ All 5 | Manual per-Greek | ✅ | ✅ |
| **IBKR integration** | ✅ Built-in | ❌ | ❌ | ❌ |
| **License** | Apache-2.0 | BSD | MIT | GPL-3.0 |

\* measured with `pytest-benchmark`, see [Benchmarks](#benchmarks). QuantLib/FinancePy numbers are rough published figures, not measured by us.

## Quickstart

Not on PyPI yet. Grab a prebuilt wheel (cp310-cp313, Linux/macOS/Windows) from the
[releases page](https://github.com/qorexdevs/opticore/releases), or build from source
(needs a C++20 compiler and CMake):

```bash
pip install git+https://github.com/qorexdevs/opticore
```

```python
import opticore as oc

# Price a European call
price = oc.price(spot=100, strike=105, expiry=0.5, rate=0.05, vol=0.20, kind="call")
# => 4.582

# Implied volatility (Jaeckel's "Let's Be Rational" — full machine precision)
iv = oc.iv(price=4.582, spot=100, strike=105, expiry=0.5, rate=0.05, kind="call")
# => 0.2000

# All Greeks in one pass
g = oc.greeks(spot=100, strike=105, expiry=0.5, rate=0.05, vol=0.20, kind="call")
print(f"Δ={g.delta:.4f}  Γ={g.gamma:.4f}  Θ={g.theta:.4f}  ν={g.vega:.4f}  ρ={g.rho:.4f}")
```

## Vectorized — Price Entire Chains

```python
import numpy as np

strikes = np.arange(90, 111, dtype=float)
prices = oc.price(spot=100, strike=strikes, expiry=0.5, rate=0.05, vol=0.20, kind="call")
# => array of 21 prices, computed in < 0.01 ms
```

## Quick start without IBKR

No account, no API keys, no network — a tiny synthetic SPY chain ships
inside the wheel. Perfect for trying things out:

```python
chain = oc.fetch_chain(provider="sample", symbol="SPY")
enriched = oc.enrich(chain, rate=0.045, div_yield=0.013)
oc.plot.smile(enriched)
```

[`examples/quickstart.py`](examples/quickstart.py) runs the whole offline path
(pricing, IV, Greeks, chain enrichment, a strategy payoff) end to end:

```bash
python examples/quickstart.py
```

For ~15-min delayed real data without an IBKR account:

```python
chain = oc.fetch_chain("AAPL", provider="yfinance")
```
```bash
pip install opticore[data-yfinance]
```

## Interactive Brokers Integration

```python
# Fetch a live chain (requires TWS/Gateway running)
chain = oc.fetch_chain("AAPL", provider="ibkr")

# Enrich with IV + Greeks in one call
enriched = oc.enrich(chain, rate=0.045)
# => DataFrame with iv, delta, gamma, theta, vega, rho columns

# Plot the volatility smile
oc.plot.smile(enriched)
```

```bash
pip install opticore[ibkr]  # adds ib_async dependency
```

## Visualization

![IV smile across expiries](docs/assets/vol-smile.png)

```python
# IV Smile
oc.plot.smile(enriched)

# Strategy payoff diagram
legs = [
    oc.Leg("call", strike=105, qty=1, premium=3.50),
    oc.Leg("put",  strike=95,  qty=1, premium=2.10),
]
oc.plot.payoff(legs)

# Same strategy as numbers (no matplotlib) - break-evens, max profit/loss, net cost
p = oc.payoff_profile(legs)
p.breakevens  # [89.4, 110.6]
p.net_cost    # 5.6

# Greeks profile
oc.plot.greek("delta", spot_range=(80, 120), strike=100,
              expiry=0.5, rate=0.05, vol=0.20, kind="both")

# ATM IV term structure (contango vs backwardation, with a fitted slope)
oc.plot.term_structure(enriched)

# Per-expiry bid-ask spread, so you can see which tenors are worth working
oc.plot.liquidity(enriched)

# Net dealer gamma profile with the flip level marked (nearest expiry by default)
oc.plot.gamma_profile(enriched)

# Per-strike dealer exposure (delta/gamma/vega/theta) with the wall and spot marked
oc.plot.exposure_profile(enriched, greek="gamma")
```

## Positioning & flow analytics

Beyond pricing, OptiCore reads the chain itself - where open interest piles up,
which strikes are churning, and how the put/call balance leans. These are pure
summations over `open_interest`/`volume`, no IV solve, so they run on a raw chain
straight from any provider:

```python
chain = oc.fetch_chain(provider="sample", symbol="SPY")

# Strike that minimizes total option-holder payout, per expiry
oc.max_pain(chain)        # => expiry, max_pain_strike, total_oi, pain_at_max_pain
oc.max_pain_curve(chain)  # => the full pain curve per strike, is_max_pain flags the pin
oc.max_pain_distance(chain)  # => signed % from spot to the pin (which way it leans)

# Put/call ratios by open interest, volume, and dollar terms
oc.pcr(chain)             # => oi_pcr, volume_pcr per expiry
oc.dollar_volume(chain)   # => premium turnover, dollar_volume_pcr per expiry

# Where the open interest concentrates - the strikes that act as walls
oc.oi_walls(chain)        # => call_wall, put_wall and their OI per expiry
oc.volume_walls(chain)    # => the same walls keyed on traded volume per expiry
oc.wall_distance(chain)   # => signed % from spot to each wall (room before the pin)
oc.oi_concentration(chain)# => top-strike share and HHI per expiry (how pinned the book is)

# Day's volume against standing OI - flags fresh positioning vs old carry
oc.turnover(chain)        # => call/put turnover per expiry
oc.volume_concentration(chain) # => top-strike share and HHI of today's traded volume
```

Each ratio has a `*_by_strike` companion (`pcr_by_strike`, `turnover_by_strike`,
`dollar_volume_by_strike`, `liquidity_by_strike`) that collapses the expiry axis
and keeps a row per strike when you want the strike map instead of the per-expiry
summary. `oi_profile` and `volume_profile` give the raw OI/volume shape across
strikes the same way.

One more reads the Greeks rather than raw counts, so it wants an enriched chain:

```python
enriched = oc.enrich(chain, rate=0.05)

# Dealer delta exposure: directional hedging lean from open interest
oc.delta_exposure(enriched)  # => call_dex, put_dex, net_dex, delta_wall_strike
oc.delta_exposure_by_strike(enriched)  # => the DEX profile per strike, cumulative_net_dex brackets the flip

# Dealer gamma exposure: where hedging flow dampens or amplifies moves
oc.gamma_exposure(enriched)  # => call_gex, put_gex, net_gex, gamma_wall_strike
oc.gamma_exposure_by_strike(enriched)  # => the GEX profile per strike, cumulative_net_gex brackets the flip
oc.gamma_concentration(enriched)  # => top-strike share and HHI of gross gamma (how pinned the hedging book is)

# Gamma flip: the spot where net dealer gamma crosses zero
oc.gamma_flip(enriched, rate=0.05)  # => net_gex, flip_spot, flip_distance_pct, regime

# Dealer vega exposure: long or short vol through the book, per vol point
oc.vega_exposure(enriched)  # => call_vex, put_vex, net_vex, vega_wall_strike
oc.vega_exposure_by_strike(enriched)  # => the VEX profile per strike, cumulative_net_vex brackets the flip

# Dealer theta exposure: daily decay the writing side collects or pays
oc.theta_exposure(enriched)  # => call_tex, put_tex, net_tex, theta_wall_strike
oc.theta_exposure_by_strike(enriched)  # => the TEX profile per strike, cumulative_net_tex brackets the flip

# Straddle-implied expected move to each expiry, straight off the ATM vol
oc.expected_move(chain)  # => atm_iv, expected_move, move_pct and the lower/upper 1-sigma band
oc.expected_move(chain, sigmas=2.0)  # widen to the ~95% band
```

## Installation Options

Until the PyPI release, install from git (swap the extras in the brackets):

```bash
pip install "git+https://github.com/qorexdevs/opticore"          # Core: pricing, IV, Greeks (requires: numpy, pandas)
pip install "opticore[ibkr] @ git+https://github.com/qorexdevs/opticore"  # + Interactive Brokers data
pip install "opticore[viz]  @ git+https://github.com/qorexdevs/opticore"  # + matplotlib plotting
pip install "opticore[all]  @ git+https://github.com/qorexdevs/opticore"  # Everything
```

## How It Works

```
Python API  ──→  nanobind  ──→  C++20 Core
(easy)          (zero-copy)     (fast)

oc.price()  ──→  _core.so  ──→  bsm.cpp      (Black-Scholes-Merton)
oc.iv()     ──→  _core.so  ──→  jaeckel.cpp   (Let's Be Rational)
oc.greeks() ──→  _core.so  ──→  greeks.cpp    (analytic, single pass)
```

- **C++20 core** — all numerical work: BSM pricing, Jaeckel IV solver, analytic Greeks
- **nanobind** — zero-copy NumPy ↔ C++ bridge (4× faster compile, 5× smaller binary than pybind11)
- **Python layer** — type handling, DataFrames, plotting, IBKR adapter

## Benchmarks

Measured with `pytest-benchmark` on an i5-11400F, Windows, Python 3.12. The 10k chain
is 10,000 options with random strikes (70-130), expiries (0.05-2y), and vols (10-50%).

| Benchmark | Mean | Per option |
|---|---|---|
| Scalar `oc.price()` | 7.0 µs | 7.0 µs |
| Batch price, 10k options | 0.65 ms | 65 ns |
| Batch IV solve, 10k options | 6.8 ms | 0.68 µs |
| Price + all 5 Greeks (`greeks_table`), 10k | 1.0 ms | 0.10 µs |
| py_vollib scalar loop, 10k options | 40 ms | 4.0 µs |

Two honest caveats:

- For a **single scalar call**, py_vollib is actually ~2x faster (4 µs vs 7 µs) — at
  that size both are dominated by Python call overhead, not math. OptiCore's win is
  vectorization: hand it arrays and the whole chain prices in one C++ pass.
- Numbers vary by machine. Reproduce yours with:

```bash
pip install pytest-benchmark py_vollib
pytest tests/python/test_benchmarks.py -m benchmark --benchmark-only
```

## Building from Source

```bash
git clone https://github.com/opticore/opticore.git
cd opticore
pip install -e ".[dev]"
```

### C++ development

```bash
cmake -B build -DOPTICORE_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

### Run Python tests

```bash
pytest tests/python/
```

## Roadmap

- [x] **Phase 1** — BSM pricing, IV, Greeks, IBKR adapter, plots
- [ ] **Phase 2** — Vol surface (SVI, SABR, SSVI), arbitrage detection, 3D visualizer
- [ ] **Phase 3** — Heston model, barriers, Asians, Monte Carlo
- [ ] **Phase 4** — More data providers (Yahoo, Polygon, Deribit), strategy builder

Full details, acceptance criteria, and non-goals: [**ROADMAP.md**](ROADMAP.md).

## Project context & decisions

- [**AGENT.md**](AGENT.md) — project state, architecture, gotchas (read this first if you're jumping in cold)
- [**ROADMAP.md**](ROADMAP.md) — phase-by-phase scope and acceptance criteria
- [**docs/decisions/**](docs/decisions/) — Architecture Decision Records (why nanobind, why NaN-not-exceptions, why Apache-2.0, etc.)

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0 — use freely in commercial and open-source projects.

## Star history

[![Star history](https://api.star-history.com/svg?repos=qorexdevs/opticore&type=Date)](https://star-history.com/#qorexdevs/opticore&Date)

---

**⭐ Star this repo if you find it useful — it helps others discover it!**
