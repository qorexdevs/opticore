# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `oi_walls()`: per-expiry call and put open-interest walls, the strikes carrying
  the most OI on each side that tend to act as resistance and support. Pure
  summation, no IV solve; OI is aggregated per strike and ties go to the lower
  strike. Returns expiry, underlying_price, call_wall, call_wall_oi, put_wall,
  put_wall_oi.
- `pcr()`: per-expiry put/call ratio from open interest and volume, a crude
  sentiment gauge. Pure summation, no IV solve; volume is optional. Returns
  expiry, underlying_price, put_oi, call_oi, oi_pcr, put_volume, call_volume,
  volume_pcr.
- `max_pain()`: per-expiry max-pain strike from open interest, the settlement
  price where total intrinsic payout to holders is smallest. Pure OI arithmetic,
  no IV solve. Returns expiry, underlying_price, max_pain_strike, total_oi,
  pain_at_max_pain.
- `straddle()`: per-expiry ATM straddle cost, breakevens and the implied move
  (straddle over spot), straight from call+put prices with no IV solve. Returns
  expiry, tte, atm_strike, underlying_price, straddle_price, breakeven_low,
  breakeven_high, implied_move.
- `rr_bf()`: per-expiry risk reversal (`call_wing_iv - put_wing_iv`) and butterfly
  (`(put_wing_iv + call_wing_iv)/2 - atm_iv`) computed from the `iv_skew()` wings.
  The standard two-number summary of a single expiry's smile. Returns expiry, tte,
  atm_iv, rr, bf, n_strikes.
- `iv_skew()`: per-expiry volatility skew, the slope of IV against
  log-moneyness `ln(K/S)` across strikes within each expiry. Returns expiry,
  tte, atm_iv, skew, put_wing_iv, call_wing_iv, n_strikes. Negative skew is the
  usual equity shape (OTM puts bid up over calls).
- `term_slope()`: fits the `atm_iv()` term structure to a line and labels it
  contango, backwardation, or flat. Returns the slope (IV points per year of
  tenor) plus the front/back IV and tte anchors.
- `atm_iv()`: ATM implied-vol term structure, one IV per expiry from the strike
  nearest spot (call and put averaged at that strike). Returns expiry, tte,
  atm_strike, atm_iv, underlying_price. Plot it to read contango vs
  backwardation or use it as the ATM anchor for a vol model.
- `enrich()` now adds an `itm` boolean column (in-the-money flag, `intrinsic > 0`)
  so you can filter ITM/OTM contracts without recomputing it.
- `enrich()` now adds an `extrinsic` column (time value, `price_col` minus
  `intrinsic`), kept raw so sub-intrinsic quotes show up as negative.
- `notebooks/05_strategies.ipynb`: spreads, straddle vs expected move, iron
  condor, position Greeks, covered call. Fully offline on the sample chain.
- API reference generated with pdoc and published to GitHub Pages on every
  push to main: https://qorexdevs.github.io/opticore/

### Fixed
- `enrich()` now raises a clear `KeyError` ("Chain has no 'x' column.") when
  `price_col` names a missing column, matching `parity_check`. Previously it
  surfaced a bare pandas `KeyError: 'x'`.

## [0.3.0] - 2026-06-07

### Changed
- **`oc.price()` vectorized path is ~8x faster for varying-vol inputs.** The
  general broadcast case (array `vol` and/or `spot`) used to fall back to a
  Python loop of per-element scalar nanobind calls; it now goes through a
  single `_bsm_price_batch_full` C++ call (10k options: 5.4 ms -> 0.65 ms).
  The fast path for scalar spot+vol is unchanged. README gained a measured
  Benchmarks section to back the perf claims.
- **Breaking - `expiry` column is now `pd.Timestamp` (UTC midnight)** (#24).
  Both `fetch_chain` providers (`ibkr`, `yfinance`) now emit `expiry` as a
  timezone-aware `pd.Timestamp` instead of a `"YYYYMMDD"` string. This makes
  date arithmetic and filtering natural (`df[df.expiry >= "2026-06-01"]`)
  and matches what `enrich()` was already producing internally. `enrich()`
  still accepts legacy string expiries via `pd.to_datetime`, so user-built
  chains aren't broken. `oc.plot.smile(expiry=...)` accepts strings or
  Timestamps and normalizes both sides for comparison.
- **Breaking - `oc.plot.*` now returns `(fig, ax)`** (#27). All three plot
  helpers (`smile`, `payoff`, `greek`) now return a `(Figure, Axes)` tuple
  per matplotlib convention, instead of just `Figure`. This unlocks
  composition (annotations, shared axes, subplots) without reaching into
  `fig.axes`. Migrate `fig = oc.plot.smile(...)` -> `fig, ax = oc.plot.smile(...)`.
  Bare calls like `oc.plot.smile(df)` are unaffected (return value discarded).
- **Breaking - `fetch_chain()` signature** (#22). Provider-specific kwargs
  (`host`, `port`, `client_id`, `market_data_type`) are no longer top-level
  parameters; they now flow through `**provider_kwargs`. Old call sites
  using kwargs (e.g. `oc.fetch_chain("AAPL", port=4001, client_id=42)`)
  continue to work unchanged because the kwargs are forwarded to the IBKR
  adapter. Positional calls passing those args are no longer supported,
  but no documentation ever advertised that pattern. The yfinance provider
  now raises `TypeError` if any provider_kwargs are passed (they would be
  silently ignored before).
- **Library code no longer prints to stdout** (#23). All status/progress
  messages from `enrich()`, `fetch_ibkr_chain()`, and `fetch_yfinance_chain()`
  now route through the standard `logging` module under the `opticore.*`
  namespace. To see them, opt in once: `logging.basicConfig(level=logging.INFO)`.
  Notebooks/scripts that relied on the old prints will go silent - this is
  intentional; libraries shouldn't pollute stdout.
- **Breaking (keyword arg):** `oc.iv(price_val=...)` -> `oc.iv(price=...)` to
  match the docstring and notebooks. Positional calls are unaffected.
- `plot.payoff` param `spot_range` is now `Optional[tuple[float, float]]`
  (was implicitly Optional - PEP 484 no longer allows that).
- `enrich()`'s internal `greek_cols` dict now has an explicit type annotation.

### Performance
- **`enrich()` is now ~50x faster** on real-sized chains (#21). Replaced the
  per-row Python loop with two batched calls into the C++ core
  (`_implied_vol_batch`, `_greeks_batch`). A 1000-row chain enriches in
  ~2 ms (was ~100 ms). NaN propagation handles unsolvable rows naturally,
  removing the bare `except Exception` that was hiding errors (#25).

### Fixed
- `notebooks/03_yfinance_tutorial.ipynb` still referenced the old `model_price`
  column (renamed to `theo_price`, see #10); the consistency-check cell raised
  `KeyError` when run against the current code.

### Added
- **`notebooks/04_iv_analysis.ipynb`** - IV-analysis walkthrough on the bundled
  sample chain: per-expiry smiles, 25-delta risk reversal, ATM term structure,
  put-call parity as a no-arb check, implied forwards / dividend yield, and the
  mispricing screen. Runs fully offline - no account, no network.
- **Type stubs for `chain` and `plot` modules** (#30) - adds
  `python/opticore/chain.pyi` and `python/opticore/plot.pyi` so mypy /
  IDE users see real types (not `Any`) for `oc.fetch_chain`, `oc.enrich`,
  `oc.parity_check`, `oc.implied_forward`, and the three plot helpers.
  `check_connection` now returns a `ConnectionStatus` TypedDict with
  autocomplete on the four keys. New `tests/python/typing/check_strict_api.py`
  is a `mypy --strict` smoke test exercising all 9 public APIs; CI gates
  on it (ubuntu/3.12 only) so the stubs can't drift.
- **Test coverage gate** (#8) - CI runs `pytest --cov=opticore` on the
  ubuntu/3.12 job and fails if coverage drops below 85% (currently ~94%).
  `pytest-cov` added to the dev extra; configuration in `[tool.coverage.*]`.
  Codecov badge on README.
- **Hypothesis property-based tests** (#6) - `tests/python/test_properties.py`
  fuzzes 100 examples per property: put-call parity, monotonicity in spot,
  vectorized == scalar, IV round-trip. Catches edge cases hand-written
  tests miss. `hypothesis>=6.0` added to the dev extra.
- **Benchmark suite** (#7) - `tests/python/test_benchmarks.py` measures
  scalar/batch pricing, IV solving, and full Greeks at 10k options, with
  optional head-to-head against `py_vollib`. Tagged with `@pytest.mark.benchmark`
  and skipped from the default run; opt-in via `pytest -m benchmark`.
- **Extended IV round-trip coverage** (#5) - `tests/python/test_iv_roundtrip.py`
  adds 60 assertions across moneyness sweeps, vol sweeps, a 2D grid, and
  NaN-propagation cases (zero expiry, negative price, arbitrage bounds,
  mixed-validity arrays).
- **Bundled sample chain** for zero-config quickstart (#9). New
  `provider="sample"` in `oc.fetch_chain(...)` loads a tiny synthetic
  SPY chain (~15 KiB parquet) shipped inside the wheel. No IBKR account,
  no yfinance install, no network - ideal for tutorials and CI. The data
  is BSM-priced with a realistic smile/skew and rebased to the current
  date on load so `enrich()` always produces sensible TTEs.
- **`enrich()` now adds `theo_price` and `mispricing` columns** by default (#10).
  `theo_price` is the BSM price at the recovered IV; `mispricing = price_col - theo_price`
  highlights stale quotes. Gate via `enrich(chain, include_theo=False)` to skip.
  The previous `model_price` column has been renamed to `theo_price` -
  small breaking change in column naming, motivated by the more standard
  finance term.
- **`oc.parity_check(chain, rate, div_yield)`** (#28) - per-(expiry, strike)
  put-call parity diagnostic. Returns a DataFrame with `parity_residual` and
  `residual_pct` columns. First-line tool for spotting stale quotes, wrong
  rate/div assumptions, or mid-pricing mistakes in fetched chains.
- **`oc.implied_forward(chain, rate)`** (#29) - recovers the implied forward
  price F(T) and dividend yield q per expiry from put-call parity, averaged
  across the N strikes nearest spot for stability. Round-trips a known q
  within ~1bp on synthetic chains.
- **yfinance provider** for `oc.fetch_chain()` - no account, no subscription,
  ~15-min delayed Yahoo data. Use via `provider="yfinance"`. Install with
  `pip install opticore[data-yfinance]`. IBKR remains the primary provider.
- Type stubs (`__init__.pyi`, `_core.pyi`) and `py.typed` marker for PEP 561
  compliance. IDEs and `mypy` now see real types for all public functions
  instead of `Any`, including `pd.DataFrame` returns and NumPy array overloads.
- `[tool.mypy]` config in `pyproject.toml`; `mypy` runs clean on
  `python/opticore` + `tests/python`.

## [0.2.0] - 2026-04-XX

### Added
- Packaging: `cibuildwheel` config + release workflow builds wheels for CPython
  3.10-3.13 on linux (manylinux2014) / macos / windows on tag push
- TestPyPI dry-run via `workflow_dispatch` input `publish_to_testpypi`
- CI: upload CTest and pytest logs as artifacts on failure
- CI status badge in README
- Project context docs: `AGENT.md`, `ROADMAP.md`, 5 ADRs in `docs/decisions/`

### Changed
- `norm_pdf` and `is_valid` in `include/opticore/math.hpp` are no longer
  `constexpr` - standard C++20 doesn't allow `std::exp` or `std::isnan` in
  constant expressions, and MSVC rejects them. GCC/Clang accepted the old
  code as an extension. No runtime impact - `inline` still permits full
  inlining by the optimizer.
- Dropped unused `strike` parameter from `jaeckel.cpp::initial_guess`
  (Brenner-Subrahmanyam uses the forward price, strike is redundant).
- Removed dead `prev_sigma` local in `implied_vol`.
- Two `TEST_CASE` names: the approx symbol -> `~=` (Windows CTest cannot handle Unicode in
  test-name filter args).

## [0.1.0] - unreleased (superseded by 0.2.0)

### Added
- Black-Scholes-Merton pricing for European calls and puts
- Jaeckel "Let's Be Rational" implied volatility solver (full 64-bit precision)
- Analytic Greeks: delta, gamma, theta (per day), vega (per 1%), rho (per 1%)
- Vectorized batch pricing and IV solving via NumPy arrays
- `greeks_table()` returning pandas DataFrame
- Interactive Brokers data adapter via `ib_async`
- Chain enrichment: `enrich()` adds IV + Greeks to any chain DataFrame
- Visualization: IV smile plots, payoff diagrams, Greeks profiles
- 5 Jupyter notebook examples
