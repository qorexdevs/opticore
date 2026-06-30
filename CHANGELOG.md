# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `theta_exposure` returns per-expiry dealer theta exposure (TEX) -
  `call_tex`/`put_tex`/`net_tex` and the gross-theta wall, the time-decay sibling
  of `vega_exposure`. Long options carry negative theta, so the long-call leg is
  negative and the short-put leg positive; net TEX is the dollars per day the
  writing side collects or pays while spot sits still.
- `plot.exposure_profile` charts per-strike dealer exposure for one expiry -
  call and put bars at each strike, net exposure overlaid as a line, and the
  exposure wall and current spot marked. `greek="delta"|"gamma"|"vega"` reads
  the matching `*_exposure_by_strike` table, so the by-strike profiles now have
  a plot the way `gamma_profile` plots the spot sweep. Defaults to the nearest
  expiry like `plot.smile`.
- `vega_exposure_by_strike` returns the per-strike VEX profile behind
  `vega_exposure` - `call_vex`/`put_vex`/`net_vex` per strike plus a
  `cumulative_net_vex` running up the board and an `is_vega_wall` flag on the
  gross-vega peak. The volatility sibling of `delta_exposure_by_strike` and
  `gamma_exposure_by_strike`, same `contract_size`-only scaling as the aggregate
  so the columns sum back to it; the sign change in the cumulative column
  brackets where net dealer vega flips.
- `vega_exposure` returns per-expiry dealer vega exposure (VEX) - the volatility
  sibling of `delta_exposure` and `gamma_exposure`. `call_vex`/`put_vex`/`net_vex`
  from `sign * vega * open_interest * contract_size` under the long-call,
  short-put convention (vega is already per 1% vol move), plus a
  `vega_wall_strike` on the gross-vega peak. Positive net VEX means the writing
  side carries long vega and gains when implied vol rises.
- `delta_exposure_by_strike` returns the per-strike DEX profile behind
  `delta_exposure` - `call_dex`/`put_dex`/`net_dex` per strike plus a
  `cumulative_net_dex` running up the board and an `is_delta_wall` flag on the
  gross-delta peak. The directional sibling of `gamma_exposure_by_strike`, same
  scaling as the aggregate so the columns sum back to it; the sign change in the
  cumulative column brackets where net dealer delta flips.

## [0.5.0] - 2026-06-29

### Added
- `plot.gamma_profile` draws the net dealer gamma curve for one expiry across
  the same spot grid `gamma_flip` scans, marking the current spot and the flip
  level and shading the price-dampening vs amplifying regions. Defaults to the
  nearest expiry. The scan is factored into a shared helper so the plot and the
  flip table always agree.
- `delta_exposure` returns per-expiry dealer delta exposure (DEX) - the
  directional sibling of `gamma_exposure`. `call_dex`/`put_dex`/`net_dex` from
  `sign * delta * open_interest * contract_size * S` under the long-call,
  short-put convention, plus a `delta_wall_strike` on the gross-delta peak.
- `gamma_exposure_by_strike` returns the per-strike GEX profile behind
  `gamma_exposure` - `call_gex`/`put_gex`/`net_gex` per strike plus a
  `cumulative_net_gex` running up the board and an `is_gamma_wall` flag on the
  gross-gamma peak. Same scaling as the aggregate, so the columns sum back to
  it; the sign change in the cumulative column brackets where net dealer gamma
  flips.
- `max_pain_curve` returns the per-strike pain curve behind `max_pain` - the
  call/put writer payout at every candidate settlement and an `is_max_pain`
  flag on the minimum, so you can chart how sharp the pin is instead of seeing
  only the strike. Pure open-interest arithmetic like the rest of the family.

## [0.4.1] - 2026-06-28

### Fixed
- Time-to-expiry across the whole chain module no longer routes through pandas
  datetime constructors. `pd.to_datetime` / `pd.DatetimeIndex` go through
  `_construct_from_dt64_naive`, which segfaults in manylinux2014 (pandas 2.3,
  cp312) - so `implied_forward`, `put_call_parity`, the strategy builders
  (`vertical`, `straddle`, `strangle`, `iron_condor`, `butterfly`, `collar`)
  and `enrich` would all crash for cp312 users, not only in CI. They now share
  a `_tte_years` helper that parses expiry with numpy `datetime64` (`asi8` for
  datetime columns, regex-normalized strings otherwise). `YYYYMMDD`, ISO 8601
  and `pd.Timestamp` expiry columns all continue to work.
- The cibuildwheel wheel smoke test no longer runs `test_chain.py` /
  `test_yfinance_adapter.py`; their fixtures call `pd.to_datetime` directly,
  which trips the same manylinux2014 segfault. Both are pure data-pipeline
  tests already covered by the full CI matrix, so the wheel test now just
  verifies the compiled core loads and prices correctly.
- `_pivot_call_put` now strips timezone from `DatetimeTZDtype` expiry columns
  via `asi8` before `pivot_table` / `groupby`, the same approach used in
  `_tte_years`. Fixes the remaining manylinux2014 cp312 segfault in
  `parity_check` and `implied_forward` when callers pass UTC-aware timestamps.
- `_tte_years` now correctly detects the datetime64 resolution unit on
  pandas 2.x. `DatetimeArray` (backing tz-stripped naive columns) does not
  expose `.unit` on its numpy `dtype`; the old `getattr(ea.dtype, "unit", "ns")`
  silently fell back to `"ns"` while actual values were in `"us"`, making
  `parity_check` residuals wrong and `implied_forward` return empty on Python
  3.11+. The lookup now tries `ea.dtype.unit` (DatetimeTZDtype), then `ea.unit`
  (DatetimeArray), then `np.datetime_data(ea.dtype)[0]` (any numpy datetime64).

## [0.4.0] - 2026-06-28

### Added
- `gamma_flip()`: per-expiry gamma flip level - the spot where net dealer GEX crosses zero.
  Recomputes gamma on a grid of hypothetical spots (default +/-20%, 81 points) from each
  option's recovered `iv` and time to expiry, finds the sign change of net GEX nearest the
  current spot and linearly interpolates the crossing. Reports `net_gex`, `flip_spot`,
  `flip_distance_pct` and a `regime` flag (`positive` dampening, `negative` amplifying,
  `flat` when a symmetric book cancels everywhere). Complements the single-spot
  `gamma_exposure`.
- `gamma_exposure()`: per-expiry dealer gamma exposure (GEX) from open interest and the
  enriched `gamma` column. Treats dealers as long call gamma and short put gamma, scales
  each leg by `gamma * open_interest * contract_size * spot**2 * 0.01` (dollar gamma per 1%
  move) and reports `call_gex`, `put_gex`, `net_gex` and the `gamma_wall_strike` carrying the
  most absolute exposure. Positive net GEX dampens moves, negative amplifies them. Needs an
  enriched chain (so it sees `gamma`), unlike the pure-OI `max_pain`/`oi_walls`.
- `collar()`: per-expiry collar on a held long position - buys a protective put `gap` strikes
  below spot and sells a covered call `gap` above it, reporting `net_debit` (put paid less call
  collected, near zero for a zero-cost collar), the floored `max_loss`, capped `max_profit` and
  the single breakeven. Read straight off the expiry payoff, no IV solve, the strategy-family
  companion to `iron_condor` that carries the underlying instead of being all options.
- `examples/quickstart.py`: a runnable script that walks the whole offline path - scalar
  and vectorized pricing, IV round-trip, Greeks, sample-chain enrichment and a strategy
  payoff - printing numbers instead of plotting, so `python examples/quickstart.py` works
  with no account, network or matplotlib.
- `notebooks/06_positioning_flow.ipynb`: positioning and flow walkthrough on the bundled
  sample chain - OI walls, max pain, OI vs volume put/call ratio, turnover and dollar
  volume, and when the `*_by_strike` view beats the per-expiry summary. Offline,
  `provider="sample"`, reads each output rather than just printing it.
- `turnover_by_strike()`: per-strike volume-to-open-interest turnover, the strike-level
  companion to `turnover()`. Collapses the expiry axis and keeps a row per strike, so a
  strike where today's volume rivals its standing book stands out as where fresh money is
  working. Same NaN rules as `turnover()`: turnover is NaN when a side has no open interest.
- `dollar_volume_by_strike()`: per-strike premium in dollars, the strike-level companion to
  `dollar_volume()`. Collapses the expiry axis and keeps a row per strike, so a few deep ITM
  strikes that dominate the dollar book but barely register in the contract count show up.
  Same NaN rules as `dollar_volume()`: dollar PCR is NaN when the call side is zero, row kept.
- `iron_condor()`: per-expiry iron condor cost, max profit/loss and the two breakevens,
  built from two out-of-the-money credit spreads (puts below spot, calls above). The short
  legs sit `gap` strikes either side of spot and the long wings `width` further out; `side`
  is `short` (credit, profits inside the shorts) or `long` (debit, profits past a wing).
  Metrics read straight off the expiry payoff, no IV solve.
- `pcr_by_strike()`: per-strike put/call ratio, the strike-level companion to `pcr()`.
  Collapses the expiry axis instead of the strike axis, summing open interest and volume
  per strike so you can see which strikes are put-heavy (downside hedging) versus call-heavy.
  Same NaN rules as `pcr()`: the ratio is NaN when the call side is zero but the row is kept.
- `butterfly()`: per-expiry butterfly spread cost, max profit/loss and the two breakevens,
  the range-bound, defined-risk sibling of `vertical()`. The body sits at the strike nearest
  spot and the wings `width` strikes either side; `side` is `long` (debit, profits if price
  pins the body) or `short` (credit, profits past a wing) and `kind` picks call/put. Metrics
  read straight off the expiry payoff, no IV solve.
- `vertical()`: per-expiry vertical spread cost, max profit/loss and breakeven, the
  defined-risk companion to `straddle()`/`strangle()`. `kind` and `side` pick the four
  standard combinations (bull/bear call/put); the lower leg is the nearest strike at/below
  spot and the upper the `width`-th above. `net_debit` is positive for debit spreads,
  negative for credits; metrics come straight off the expiry payoff, no IV solve.
- `payoff_profile()`: the numeric companion to `plot.payoff()`. Same intrinsic-value model
  but returns a `PayoffProfile` (sampled `spots`/`pnl`, interpolated `breakevens`, plus
  `max_profit`, `max_loss` and `net_cost`) instead of a figure, so multi-leg strategies can
  be screened without matplotlib. The grid auto-sizes around the strikes when `spot_range`
  is omitted; `max_profit`/`max_loss` are taken over the sampled grid.
- `strangle()`: per-expiry OTM strangle cost, breakevens and implied move, the companion to
  `straddle()`. The call leg is the `width`-th strike above spot and the put leg the `width`-th
  below (`width=1` is the nearest OTM pair); pure price arithmetic, no IV solve. An expiry is
  dropped when neither side has a strike that far out.
- `plot.term_structure()`: ATM IV against time to expiry from `atm_iv()`, one point
  per expiry so contango and backwardation read off the slope. With `fit=True` (default)
  the `term_slope()` line is overlaid and its shape labelled. Returns `(fig, ax)` like
  the other plotters.
- `plot.liquidity()`: bar chart of the per-expiry bid-ask spread from `liquidity()`,
  with a marker for each expiry's widest relative spread so a single untradeable strike
  still shows. `relative=True` (default) plots percent of mid, `relative=False` the
  absolute spread. Returns `(fig, ax)` like the other plotters.
- `liquidity_by_strike()`: the per-strike view behind `liquidity()`, one row per
  (expiry, strike, kind) with `bid`, `ask`, `mid`, absolute `spread` and `rel_spread`,
  no aggregation - the raw distribution the median collapses, for picking the actual
  contract to trade. Same drop rules as `liquidity()`; sorted by expiry then strike
  then kind.
- `liquidity()`: per-expiry bid-ask spread as a tradeability gauge, both absolute
  (`ask - bid`) and relative to mid, reported as the median across the expiry's quotes
  plus the widest relative spread so one untradeable strike can't hide behind a decent
  median. Pure arithmetic, no IV solve; `mid` used when present, else the midpoint,
  crossed or non-positive markets dropped. Returns expiry, underlying_price, n_quotes,
  median_spread, median_rel_spread, max_rel_spread.
- `dollar_volume()`: per-expiry premium in dollars traded and standing, call and put
  side. Where `pcr()` and `turnover()` count contracts, this weights each strike by
  its price, so a few expensive contracts can outweigh a swarm of cheap wings and the
  dollar put/call ratio reads where the money actually sits, not just the tally. Pure
  price arithmetic, no IV solve; `price_col` (default mid) and `contract_size` (default
  100) configurable, volume optional and NaN when missing. Returns expiry,
  underlying_price, call/put_dollar_volume, dollar_volume_pcr, call/put_dollar_oi,
  dollar_oi_pcr.
- `turnover()`: per-expiry volume-to-open-interest turnover, call and put side. A
  ratio near or above 1 means about as many contracts traded today as were already
  open, flagging fresh positioning over the carry of an existing book. Pure
  summation, no IV solve; volume optional and NaN when missing, turnover NaN when a
  side has no open interest. Returns expiry, underlying_price, call_volume, call_oi,
  call_turnover, put_volume, put_oi, put_turnover.
- `volume_walls()`: per-expiry call and put traded-volume walls, the strikes that
  traded the most contracts today, the day's-flow companion to `oi_walls()`. A
  volume wall that isn't an OI wall flags where fresh flow is concentrating before
  it settles into open interest. Pure summation, no IV solve; volume is aggregated
  per strike and ties go to the lower strike. Returns expiry, underlying_price,
  call_wall, call_wall_volume, put_wall, put_wall_volume.
- `volume_profile()`: per-strike call/put traded-volume profile, one row per
  expiry and strike, the day's-flow companion to `oi_profile()` - which side and
  strike actually changed hands today versus the standing book. Pure summation,
  no IV solve; volume is aggregated per strike and empty strikes dropped. Returns
  expiry, underlying_price, strike, call_volume, put_volume, total_volume,
  net_volume.
- `oi_profile()`: per-strike call/put open-interest profile, one row per expiry
  and strike, the raw distribution `oi_walls()` and `max_pain()` collapse. Pure
  summation, no IV solve; OI is aggregated per strike and empty strikes dropped.
  Returns expiry, underlying_price, strike, call_oi, put_oi, total_oi, net_oi.
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
