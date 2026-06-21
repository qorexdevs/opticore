"""Option chain fetching and enrichment."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_connection(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 99,
    timeout: float = 5.0,
) -> dict:
    """Test connectivity to TWS or IB Gateway.

    Quick way to verify your IBKR setup before fetching data.

    Returns
    -------
    dict with keys: connected (bool), account, server_version, message

    Examples
    --------
    >>> import opticore as oc
    >>> status = oc.check_connection()
    >>> print(status["message"])
    """
    from opticore.data.ibkr import check_connection as _check

    return _check(host=host, port=port, client_id=client_id, timeout=timeout)


def fetch_chain(
    symbol: str,
    provider: str = "ibkr",
    max_expiries: int = 6,
    strike_count: int = 20,
    timeout: float = 30.0,
    **provider_kwargs,
) -> pd.DataFrame:
    """Fetch an option chain for a given symbol.

    Parameters
    ----------
    symbol : str
        Underlying ticker symbol (e.g. "AAPL", "SPY").
    provider : str
        Data provider. Supported:
          - ``"ibkr"`` (default): Interactive Brokers via TWS/IB Gateway.
            Requires an account + market-data subscription.
          - ``"yfinance"`` (aliases: ``"yahoo"``, ``"yf"``): Yahoo Finance,
            ~15-min delayed, no account needed. Install via
            ``pip install opticore[data-yfinance]``.
          - ``"sample"``: a tiny synthetic SPY chain bundled with the wheel.
            Zero dependencies, zero config — ideal for tutorials and CI.
            Data is BSM-priced with a realistic smile, *not* real quotes.
    max_expiries : int
        Number of nearest expiries to fetch (default: 6). Shared contract
        across all providers.
    strike_count : int
        Number of strikes around ATM on each side (default: 20). Shared
        contract across all providers.
    timeout : float
        Maximum seconds to wait for data (default: 30). Shared contract
        across all providers.
    **provider_kwargs
        Provider-specific options. Unknown kwargs are forwarded as-is
        to the underlying provider adapter.

        ``ibkr`` accepts:
            - ``host`` (str, default ``"127.0.0.1"``)
            - ``port`` (int, default ``7497`` — TWS live; use ``4001`` for Gateway)
            - ``client_id`` (int, default ``1``)
            - ``market_data_type`` (int, default ``3`` — 1=live, 3=delayed, 4=frozen)

        ``yfinance`` accepts no extra kwargs.

    Returns
    -------
    pd.DataFrame
        Option chain with columns: symbol, strike, expiry, kind,
        bid, ask, last, volume, open_interest, underlying_price, mid.

    Examples
    --------
    >>> import opticore as oc
    >>> # Default IBKR (uses defaults for host/port/client_id/market_data_type)
    >>> chain = oc.fetch_chain("AAPL")  # doctest: +SKIP
    >>> # IBKR with explicit Gateway port
    >>> chain = oc.fetch_chain("AAPL", port=4001, client_id=42)  # doctest: +SKIP
    >>> # yfinance (no account)
    >>> chain = oc.fetch_chain("AAPL", provider="yfinance")  # doctest: +SKIP
    """
    p = provider.lower()
    shared = dict(
        symbol=symbol,
        max_expiries=max_expiries,
        strike_count=strike_count,
        timeout=timeout,
    )
    if p == "ibkr":
        from opticore.data.ibkr import fetch_ibkr_chain

        return fetch_ibkr_chain(**shared, **provider_kwargs)
    elif p in ("yfinance", "yahoo", "yf"):
        if provider_kwargs:
            raise TypeError(
                f"yfinance provider takes no provider_kwargs, got: {sorted(provider_kwargs)}"
            )
        from opticore.data.yfinance_adapter import fetch_yfinance_chain

        return fetch_yfinance_chain(**shared)
    elif p == "sample":
        if provider_kwargs:
            raise TypeError(
                f"sample provider takes no provider_kwargs, got: {sorted(provider_kwargs)}"
            )
        from opticore.data.sample import fetch_sample_chain

        return fetch_sample_chain(**shared)
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Supported: 'ibkr', 'yfinance', 'sample'."
        )


def enrich(
    chain: pd.DataFrame,
    rate: float = 0.045,
    div_yield: float = 0.0,
    price_col: str = "mid",
    include_theo: bool = True,
) -> pd.DataFrame:
    """Enrich an option chain DataFrame with IV and Greeks.

    Adds columns: ``mid, tte, iv, delta, gamma, theta, vega, rho,
    moneyness, intrinsic, extrinsic, itm``. ``itm`` is a boolean
    in-the-money flag (``intrinsic > 0``). ``extrinsic`` is the time value
    (``price_col`` minus ``intrinsic``), kept raw so sub-intrinsic quotes
    show up as negative. When ``include_theo=True`` (default), also
    adds ``theo_price`` (BSM price at the recovered IV) and ``mispricing``
    (``price_col`` minus ``theo_price``) — useful for spotting stale quotes.

    Parameters
    ----------
    chain : pd.DataFrame
        Must have columns: strike, expiry, kind, underlying_price,
        and either 'mid' or 'bid'+'ask' (or the column named by price_col).
    rate : float
        Risk-free interest rate (default: 0.045).
    div_yield : float
        Continuous dividend yield (default: 0.0).
    price_col : str
        Column to use for option price: 'mid', 'bid', 'ask', 'last'.
    include_theo : bool
        If True (default), add ``theo_price`` and ``mispricing`` columns.
        Set False to skip them — useful when you only need IV/Greeks
        and want to keep the output narrow.

    Returns
    -------
    pd.DataFrame
        Original chain with added columns.
    """
    from opticore._core import _greeks_batch, _implied_vol_batch

    df = chain.copy()

    # ── Compute mid if not present ───────────────────────────────────────
    if "mid" not in df.columns and "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2.0

    if price_col not in df.columns:
        raise KeyError(f"Chain has no {price_col!r} column.")

    # ── Time to expiry in years ──────────────────────────────────────────
    # Accept either pd.Timestamp (current schema) or legacy "YYYYMMDD" /
    # "YYYY-MM-DD" strings (pandas can parse both via to_datetime).
    now = datetime.now(timezone.utc)
    expiry_dt = pd.to_datetime(df["expiry"], utc=True)
    df["tte"] = (expiry_dt - now).dt.total_seconds() / (365.25 * 24 * 3600)
    df["tte"] = df["tte"].clip(lower=1e-6)  # avoid zero/negative

    # ── Moneyness ────────────────────────────────────────────────────────
    df["moneyness"] = df["strike"] / df["underlying_price"]

    # ── Intrinsic value ──────────────────────────────────────────────────
    is_call = df["kind"].str.lower().isin(["call", "c"])
    df["intrinsic"] = np.where(
        is_call,
        np.maximum(df["underlying_price"] - df["strike"], 0),
        np.maximum(df["strike"] - df["underlying_price"], 0),
    )

    # ── Extrinsic (time) value ───────────────────────────────────────────
    # Kept raw, not clipped at zero: a value below zero means the quote sits
    # under intrinsic (stale or arb), which is exactly the signal you want.
    df["extrinsic"] = df[price_col] - df["intrinsic"]

    # ── In-the-money flag ────────────────────────────────────────────────
    # intrinsic > 0 is exactly ITM; at-the-money (spot == strike) is False.
    df["itm"] = df["intrinsic"] > 0

    # ── Vectorized IV + Greeks ───────────────────────────────────────────
    # Single trip into C++ for IV solve, then a second for Greeks. NaN
    # propagation handles unsolvable rows: _implied_vol_batch returns NaN,
    # _greeks_batch then produces NaN price/greeks for those rows naturally.
    #
    # The C++ binding declares `nb::ndarray<double, nb::c_contig>` which is
    # strict on dtype, alignment, C-contiguous layout, AND writeability.
    # Pandas 3.0's `Series.to_numpy()` returns a read-only view by default
    # (writeable=False), which nanobind rejects even though we only read
    # from the array. `np.require` with "W" forces a writeable copy when
    # needed; "C"/"A" handle layout + alignment in the same call.
    def _coerce(series, dtype):
        return np.require(series.to_numpy(), dtype=dtype, requirements=["C", "A", "W"])

    prices = _coerce(df[price_col], np.float64)
    spots = _coerce(df["underlying_price"], np.float64)
    strikes = _coerce(df["strike"], np.float64)
    ttes = _coerce(df["tte"], np.float64)
    is_call_arr = _coerce(is_call, bool)

    iv_values = np.asarray(
        _implied_vol_batch(
            prices,
            spots,
            strikes,
            ttes,
            float(rate),
            float(div_yield),
            is_call_arr,
        )
    )
    df["iv"] = iv_values

    theo_price, delta, gamma, theta, vega, rho = _greeks_batch(
        spots,
        strikes,
        ttes,
        float(rate),
        iv_values,
        float(div_yield),
        is_call_arr,
    )

    if include_theo:
        df["theo_price"] = np.asarray(theo_price)
        df["mispricing"] = df[price_col] - df["theo_price"]
    df["delta"] = np.asarray(delta)
    df["gamma"] = np.asarray(gamma)
    df["theta"] = np.asarray(theta)
    df["vega"] = np.asarray(vega)
    df["rho"] = np.asarray(rho)

    # ── Summary ──────────────────────────────────────────────────────────
    n_total = len(df)
    n_failed = df["iv"].isna().sum()
    n_success = n_total - n_failed
    pct_failed = (n_failed / n_total * 100) if n_total > 0 else 0

    logger.info("Enriched %d options, %d IV failures (%.1f%%)", n_success, n_failed, pct_failed)

    return df


# ════════════════════════════════════════════════════════════════════════════
# Parity diagnostics
# ════════════════════════════════════════════════════════════════════════════


def _pivot_call_put(chain: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Inner helper: align call/put rows side-by-side per (expiry, strike).

    Returns a frame with columns: expiry, strike, underlying_price,
    call_mid, put_mid (only rows where BOTH call and put exist).
    """
    df = chain.copy()
    if price_col == "mid" and "mid" not in df.columns and {"bid", "ask"}.issubset(df.columns):
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
    if price_col not in df.columns:
        raise KeyError(f"Chain has no {price_col!r} column.")

    if df.empty or "kind" not in df.columns:
        return pd.DataFrame(
            columns=[
                "expiry",
                "strike",
                "underlying_price",
                "call_mid",
                "put_mid",
            ]
        )

    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    keep = ["expiry", "strike", "underlying_price", "_kind", price_col]
    sub = df[keep].dropna(subset=[price_col])

    if sub.empty or sub["_kind"].nunique() < 2:
        return pd.DataFrame(
            columns=[
                "expiry",
                "strike",
                "underlying_price",
                "call_mid",
                "put_mid",
            ]
        )

    # Pivot kinds into columns; keep underlying_price (assumed constant per expiry)
    pivot = sub.pivot_table(
        index=["expiry", "strike"],
        columns="_kind",
        values=price_col,
        aggfunc="first",
    ).reset_index()

    # Bring underlying_price back (first observed per expiry)
    spot_per_expiry = sub.groupby("expiry")["underlying_price"].first().rename("underlying_price")
    pivot = pivot.merge(spot_per_expiry, on="expiry", how="left")

    pivot = pivot.dropna(subset=["call", "put"])
    return pivot.rename(columns={"call": "call_mid", "put": "put_mid"})


def parity_check(
    chain: pd.DataFrame,
    rate: float = 0.045,
    div_yield: float = 0.0,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Compute per-(expiry, strike) put-call parity residuals.

    Parity (Black-Scholes-Merton with continuous dividend yield)::

        C - P = S * exp(-q*T) - K * exp(-r*T)

    Large residuals indicate stale quotes, wide spreads, mid-pricing error,
    or a wrong assumption about ``rate`` / ``div_yield``. This is the first
    diagnostic to run when an enriched chain looks weird.

    Parameters
    ----------
    chain : pd.DataFrame
        Must have columns: ``expiry``, ``strike``, ``kind``,
        ``underlying_price``, and the column named by ``price_col``
        (or ``bid`` + ``ask`` to compute ``mid``).
    rate : float
        Risk-free rate (default: 0.045).
    div_yield : float
        Continuous dividend yield (default: 0.0).
    price_col : str
        Which price to use: 'mid' (default), 'last', 'bid', 'ask'.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, strike, call_mid, put_mid,
        parity_residual, residual_pct.
        ``parity_residual = (C - P) - (S*exp(-q*T) - K*exp(-r*T))``.
        ``residual_pct = residual / underlying_price * 100``.

    Examples
    --------
    >>> import opticore as oc
    >>> diag = oc.parity_check(chain, rate=0.05)  # doctest: +SKIP
    >>> diag.nlargest(5, "residual_pct")          # doctest: +SKIP
    """
    p = _pivot_call_put(chain, price_col)
    if p.empty:
        return pd.DataFrame(
            columns=[
                "expiry",
                "strike",
                "call_mid",
                "put_mid",
                "parity_residual",
                "residual_pct",
            ]
        )

    # Time to expiry in years (accept Timestamp or legacy string)
    now = datetime.now(timezone.utc)
    expiry_dt = pd.to_datetime(p["expiry"], utc=True)
    tte = (expiry_dt - now).dt.total_seconds() / (365.25 * 24 * 3600)
    tte = tte.clip(lower=1e-6).to_numpy(dtype=np.float64)

    S = p["underlying_price"].to_numpy(dtype=np.float64)
    K = p["strike"].to_numpy(dtype=np.float64)
    expected = S * np.exp(-div_yield * tte) - K * np.exp(-rate * tte)
    actual = p["call_mid"].to_numpy(dtype=np.float64) - p["put_mid"].to_numpy(dtype=np.float64)
    residual = actual - expected

    out = p[["expiry", "strike", "call_mid", "put_mid"]].copy()
    out["parity_residual"] = residual
    out["residual_pct"] = residual / S * 100.0
    return out.reset_index(drop=True)


def implied_forward(
    chain: pd.DataFrame,
    rate: float = 0.045,
    n_atm_strikes: int = 3,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Recover the implied forward price F(T) and dividend yield q per expiry.

    From put-call parity::

        C - P = exp(-r*T) * (F - K)
        ⇒ F = K + exp(r*T) * (C - P)

    Then::

        F = S * exp((r - q) * T)
        ⇒ q = r - ln(F / S) / T

    For numerical stability, F is averaged across the ``n_atm_strikes``
    strikes nearest the spot — these have the tightest (C - P) and least
    bid/ask noise leverage.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``parity_check`` / ``enrich``.
    rate : float
        Risk-free rate (default: 0.045). The implied yield is computed
        relative to this — pass the same rate you'd use for ``enrich``.
    n_atm_strikes : int
        Number of strikes nearest spot used to average F per expiry (default: 3).
    price_col : str
        Which price to use (default: 'mid').

    Returns
    -------
    pd.DataFrame
        Columns: expiry, tte, forward, implied_div_yield, n_strikes_used.

    Examples
    --------
    >>> import opticore as oc
    >>> oc.implied_forward(chain, rate=0.05)  # doctest: +SKIP
    """
    p = _pivot_call_put(chain, price_col)
    if p.empty:
        return pd.DataFrame(
            columns=["expiry", "tte", "forward", "implied_div_yield", "n_strikes_used"]
        )

    now = datetime.now(timezone.utc)
    expiry_dt = pd.to_datetime(p["expiry"], utc=True)
    p = p.assign(_tte=(expiry_dt - now).dt.total_seconds() / (365.25 * 24 * 3600))
    # F per row from parity; average the k nearest spot per expiry.
    p["_F_row"] = p["strike"] + np.exp(rate * p["_tte"]) * (p["call_mid"] - p["put_mid"])
    p["_dist"] = (p["strike"] - p["underlying_price"]).abs()

    rows = []
    for exp, grp in p.groupby("expiry", sort=True):
        atm = grp.nsmallest(n_atm_strikes, "_dist")
        if atm.empty:
            continue
        tte = float(atm["_tte"].iloc[0])
        if tte <= 0:
            continue
        F = float(atm["_F_row"].mean())
        S = float(atm["underlying_price"].iloc[0])
        if F <= 0 or S <= 0:
            q = float("nan")
        else:
            q = rate - np.log(F / S) / tte
        rows.append(
            {
                "expiry": exp,
                "tte": tte,
                "forward": F,
                "implied_div_yield": q,
                "n_strikes_used": int(len(atm)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["expiry", "tte", "forward", "implied_div_yield", "n_strikes_used"],
    )


def atm_iv(
    chain: pd.DataFrame,
    rate: float = 0.045,
    div_yield: float = 0.0,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Recover the ATM implied-vol term structure, one IV per expiry.

    For each expiry the strike nearest spot is taken as ATM, and its implied
    vol is read from the enriched chain. When both a call and a put sit at that
    strike, their IVs are averaged — the two should agree under parity, and the
    mean smooths bid/ask noise. Rows whose IV could not be solved are skipped.

    This is the curve you plot to see term structure (contango vs backwardation)
    or feed into a vol model as the ATM anchor.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``enrich`` / ``parity_check``.
    rate : float
        Risk-free rate (default: 0.045).
    div_yield : float
        Continuous dividend yield (default: 0.0).
    price_col : str
        Which price to use (default: 'mid').

    Returns
    -------
    pd.DataFrame
        Columns: expiry, tte, atm_strike, atm_iv, underlying_price.
        One row per expiry, sorted by expiry.

    Examples
    --------
    >>> import opticore as oc
    >>> oc.atm_iv(chain)  # doctest: +SKIP
    """
    cols = ["expiry", "tte", "atm_strike", "atm_iv", "underlying_price"]
    if chain.empty:
        return pd.DataFrame(columns=cols)

    enriched = enrich(
        chain, rate=rate, div_yield=div_yield, price_col=price_col, include_theo=False
    )

    rows = []
    for exp, grp in enriched.groupby("expiry", sort=True):
        valid = grp.dropna(subset=["iv"])
        if valid.empty:
            continue
        spot = float(valid["underlying_price"].iloc[0])
        dist = (valid["strike"] - spot).abs()
        nearest = valid.loc[dist == dist.min()]
        rows.append(
            {
                "expiry": exp,
                "tte": float(nearest["tte"].iloc[0]),
                "atm_strike": float(nearest["strike"].iloc[0]),
                "atm_iv": float(nearest["iv"].mean()),
                "underlying_price": spot,
            }
        )

    return pd.DataFrame(rows, columns=cols)


class TermSlope(NamedTuple):
    """Least-squares slope of the ATM IV term structure.

    ``slope`` is in IV points per year of tenor: positive means longer-dated
    options carry more vol (contango), negative means the front is bid up
    (backwardation).
    """

    slope: float
    shape: str  # "contango", "backwardation", or "flat"
    front_iv: float
    back_iv: float
    front_tte: float
    back_tte: float


def term_slope(atm: pd.DataFrame, flat_tol: float = 1e-3) -> TermSlope:
    """Fit the ATM IV term structure to a line and label its shape.

    Takes the output of :func:`atm_iv` and regresses ``atm_iv`` on ``tte``.
    The sign of the slope tells you whether the curve is in contango or
    backwardation; ``flat_tol`` is the dead band around zero treated as flat.

    Parameters
    ----------
    atm : pd.DataFrame
        Output of ``atm_iv`` (needs ``tte`` and ``atm_iv`` columns).
    flat_tol : float
        Slopes with magnitude below this are reported as 'flat' (default 1e-3).

    Returns
    -------
    TermSlope
        Named tuple with slope, shape, and the front/back IV and tte anchors.

    Examples
    --------
    >>> import opticore as oc
    >>> oc.term_slope(oc.atm_iv(chain))  # doctest: +SKIP
    """
    if "tte" not in atm.columns or "atm_iv" not in atm.columns:
        raise KeyError("term_slope needs the atm_iv() output (columns 'tte', 'atm_iv')")

    df = atm.dropna(subset=["tte", "atm_iv"]).sort_values("tte")
    if len(df) < 2:
        raise ValueError("term_slope needs at least two expiries with a solved IV")

    tte = df["tte"].to_numpy(dtype=float)
    ivs = df["atm_iv"].to_numpy(dtype=float)
    slope = float(np.polyfit(tte, ivs, 1)[0])

    if slope > flat_tol:
        shape = "contango"
    elif slope < -flat_tol:
        shape = "backwardation"
    else:
        shape = "flat"

    return TermSlope(
        slope=slope,
        shape=shape,
        front_iv=float(ivs[0]),
        back_iv=float(ivs[-1]),
        front_tte=float(tte[0]),
        back_tte=float(tte[-1]),
    )


def iv_skew(
    chain: pd.DataFrame,
    rate: float = 0.045,
    div_yield: float = 0.0,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Per-expiry volatility skew: slope of IV against log-moneyness.

    Where :func:`atm_iv` / :func:`term_slope` look across expiries, this looks
    across strikes within each expiry. For every expiry the IVs are regressed on
    ``ln(K / S)`` and the slope is reported: equities usually print a negative
    skew (out-of-the-money puts bid up over calls). Call and put IVs at the same
    strike are averaged first, and rows with an unsolved IV are skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``enrich`` / ``atm_iv``.
    rate : float
        Risk-free rate (default: 0.045).
    div_yield : float
        Continuous dividend yield (default: 0.0).
    price_col : str
        Which price to use (default: 'mid').

    Returns
    -------
    pd.DataFrame
        Columns: expiry, tte, atm_iv, skew, put_wing_iv, call_wing_iv,
        n_strikes. ``skew`` is d(IV)/d(ln(K/S)); ``put_wing_iv`` and
        ``call_wing_iv`` are the IVs at the lowest and highest strike.
        One row per expiry, sorted by expiry.

    Examples
    --------
    >>> import opticore as oc
    >>> oc.iv_skew(chain)  # doctest: +SKIP
    """
    cols = ["expiry", "tte", "atm_iv", "skew", "put_wing_iv", "call_wing_iv", "n_strikes"]
    if chain.empty:
        return pd.DataFrame(columns=cols)

    enriched = enrich(
        chain, rate=rate, div_yield=div_yield, price_col=price_col, include_theo=False
    )

    rows = []
    for exp, grp in enriched.groupby("expiry", sort=True):
        valid = grp.dropna(subset=["iv"])
        if valid.empty:
            continue
        spot = float(valid["underlying_price"].iloc[0])
        # one IV per strike: average the call and put leg if both are present
        per_strike = (
            valid.groupby("strike").agg(iv=("iv", "mean"), tte=("tte", "first")).reset_index()
        )
        if len(per_strike) < 2:
            continue
        lm = np.log(per_strike["strike"].to_numpy(dtype=float) / spot)
        ivs = per_strike["iv"].to_numpy(dtype=float)
        slope = float(np.polyfit(lm, ivs, 1)[0])
        order = np.argsort(lm)
        atm_idx = int(np.argmin(np.abs(lm)))
        rows.append(
            {
                "expiry": exp,
                "tte": float(per_strike["tte"].iloc[0]),
                "atm_iv": float(ivs[atm_idx]),
                "skew": slope,
                "put_wing_iv": float(ivs[order[0]]),
                "call_wing_iv": float(ivs[order[-1]]),
                "n_strikes": int(len(per_strike)),
            }
        )

    return pd.DataFrame(rows, columns=cols)


def rr_bf(
    chain: pd.DataFrame,
    rate: float = 0.045,
    div_yield: float = 0.0,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Per-expiry risk reversal and butterfly from the smile wings.

    Built on :func:`iv_skew`. ``rr`` (risk reversal) is
    ``call_wing_iv - put_wing_iv`` — positive when calls are bid over puts,
    negative for the usual equity put skew. ``bf`` (butterfly) is
    ``(put_wing_iv + call_wing_iv) / 2 - atm_iv`` — how far the wings sit above
    the money, i.e. the smile's convexity. Together they're the standard
    two-number summary of a single expiry's skew.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``enrich`` / ``iv_skew``.
    rate : float
        Risk-free rate (default: 0.045).
    div_yield : float
        Continuous dividend yield (default: 0.0).
    price_col : str
        Which price to use (default: 'mid').

    Returns
    -------
    pd.DataFrame
        Columns: expiry, tte, atm_iv, rr, bf, n_strikes. One row per expiry,
        sorted by expiry. Empty if no expiry has at least two solved strikes.

    Examples
    --------
    >>> import opticore as oc
    >>> oc.rr_bf(chain)  # doctest: +SKIP
    """
    cols = ["expiry", "tte", "atm_iv", "rr", "bf", "n_strikes"]
    skew = iv_skew(chain, rate=rate, div_yield=div_yield, price_col=price_col)
    if skew.empty:
        return pd.DataFrame(columns=cols)

    out = skew.copy()
    out["rr"] = out["call_wing_iv"] - out["put_wing_iv"]
    out["bf"] = (out["put_wing_iv"] + out["call_wing_iv"]) / 2.0 - out["atm_iv"]
    return out[cols]


def straddle(
    chain: pd.DataFrame,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Per-expiry ATM straddle cost, breakevens and the implied move.

    For each expiry the strike nearest spot is taken as ATM and its call and put
    prices are summed: that's what a long straddle costs and what the market is
    pricing in as the expected move into that expiry. Breakevens sit one straddle
    width either side of the strike; ``implied_move`` is the straddle as a
    fraction of spot, the quick "the options imply a +/- X% move" read.

    Unlike :func:`atm_iv` this is pure price arithmetic - no IV solve - so it
    works even on rows where the IV doesn't converge. Only strikes that quote
    both a call and a put are used.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``parity_check`` / ``enrich``.
    price_col : str
        Which price to use (default: 'mid').

    Returns
    -------
    pd.DataFrame
        Columns: expiry, tte, atm_strike, underlying_price, straddle_price,
        breakeven_low, breakeven_high, implied_move. One row per expiry,
        sorted by expiry.

    Examples
    --------
    >>> import opticore as oc
    >>> oc.straddle(chain)  # doctest: +SKIP
    """
    cols = [
        "expiry",
        "tte",
        "atm_strike",
        "underlying_price",
        "straddle_price",
        "breakeven_low",
        "breakeven_high",
        "implied_move",
    ]
    p = _pivot_call_put(chain, price_col)
    if p.empty:
        return pd.DataFrame(columns=cols)

    now = datetime.now(timezone.utc)
    expiry_dt = pd.to_datetime(p["expiry"], utc=True)
    p = p.assign(_tte=(expiry_dt - now).dt.total_seconds() / (365.25 * 24 * 3600))
    p["_dist"] = (p["strike"] - p["underlying_price"]).abs()

    rows = []
    for exp, grp in p.groupby("expiry", sort=True):
        atm = grp.loc[grp["_dist"] == grp["_dist"].min()].iloc[0]
        spot = float(atm["underlying_price"])
        strike = float(atm["strike"])
        cost = float(atm["call_mid"] + atm["put_mid"])
        rows.append(
            {
                "expiry": exp,
                "tte": float(atm["_tte"]),
                "atm_strike": strike,
                "underlying_price": spot,
                "straddle_price": cost,
                "breakeven_low": strike - cost,
                "breakeven_high": strike + cost,
                "implied_move": cost / spot if spot > 0 else float("nan"),
            }
        )

    return pd.DataFrame(rows, columns=cols)


def max_pain(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry max-pain strike from open interest.

    Max pain is the settlement price at which the total intrinsic payout owed to
    option holders is smallest - i.e. where writers, as a group, lose the least.
    For each candidate price S (every listed strike) the payout is
    ``sum(call_oi * max(S - K, 0)) + sum(put_oi * max(K - S, 0))`` over all
    strikes K in that expiry. The strike that minimises it is the max-pain point
    often quoted as a magnet near expiry.

    Pure open-interest arithmetic, no IV solve, so it works on any chain that
    carries ``open_interest``. Strikes missing OI are treated as zero; an expiry
    with no open interest at all is skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``parity_check`` / ``enrich``; needs ``open_interest``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, max_pain_strike, total_oi,
        pain_at_max_pain. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "max_pain_strike",
        "total_oi",
        "pain_at_max_pain",
    ]
    if chain.empty or "kind" not in chain.columns or "open_interest" not in chain.columns:
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "strike", "open_interest"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        strikes = np.sort(grp["strike"].unique())
        calls = grp[grp["_kind"] == "call"]
        puts = grp[grp["_kind"] == "put"]
        call_oi = calls.groupby("strike")["open_interest"].sum()
        put_oi = puts.groupby("strike")["open_interest"].sum()
        total_oi = float(call_oi.sum() + put_oi.sum())
        if total_oi <= 0:
            continue

        best_strike = float("nan")
        best_pain = float("inf")
        for s in strikes:
            pain = 0.0
            for k, oi in call_oi.items():
                if s > k:
                    pain += float(oi) * (s - k)
            for k, oi in put_oi.items():
                if k > s:
                    pain += float(oi) * (k - s)
            if pain < best_pain:
                best_pain = pain
                best_strike = float(s)

        rows.append(
            {
                "expiry": exp,
                "underlying_price": float(grp["underlying_price"].iloc[0]),
                "max_pain_strike": best_strike,
                "total_oi": total_oi,
                "pain_at_max_pain": best_pain,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def pcr(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry put/call ratio from open interest and volume.

    The put/call ratio is a crude sentiment gauge: more puts than calls (ratio
    above 1) reads as defensive or bearish positioning, below 1 as bullish. Both
    the open-interest and the volume cut are reported - OI is the standing book,
    volume is the day's flow, and they often disagree.

    Pure summation, no IV solve. ``open_interest`` is required; ``volume`` is
    optional and its ratio is NaN when the column is missing or an expiry has no
    call volume. A ratio is NaN when the call side is zero (division by zero) but
    the row is still emitted so the put total stays visible.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``max_pain`` / ``enrich``; needs ``open_interest``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, put_oi, call_oi, oi_pcr,
        put_volume, call_volume, volume_pcr. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "put_oi",
        "call_oi",
        "oi_pcr",
        "put_volume",
        "call_volume",
        "volume_pcr",
    ]
    if chain.empty or "kind" not in chain.columns or "open_interest" not in chain.columns:
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "open_interest"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    has_volume = "volume" in df.columns

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        calls = grp[grp["_kind"] == "call"]
        puts = grp[grp["_kind"] == "put"]
        call_oi = float(calls["open_interest"].sum())
        put_oi = float(puts["open_interest"].sum())
        oi_pcr = put_oi / call_oi if call_oi > 0 else float("nan")

        if has_volume:
            call_vol = float(calls["volume"].sum(skipna=True))
            put_vol = float(puts["volume"].sum(skipna=True))
            volume_pcr = put_vol / call_vol if call_vol > 0 else float("nan")
        else:
            call_vol = put_vol = volume_pcr = float("nan")

        rows.append(
            {
                "expiry": exp,
                "underlying_price": float(grp["underlying_price"].iloc[0]),
                "put_oi": put_oi,
                "call_oi": call_oi,
                "oi_pcr": oi_pcr,
                "put_volume": put_vol,
                "call_volume": call_vol,
                "volume_pcr": volume_pcr,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def oi_walls(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry call and put open-interest walls.

    The "walls" are the strikes carrying the most open interest on each side: the
    call wall tends to act as overhead resistance and the put wall as support,
    since the dealers short those options hedge against price moving through them.
    Reported per expiry alongside the OI sitting at each wall.

    Pure summation over ``open_interest``, no IV solve. OI is aggregated per
    strike, so split call/put rows at the same strike are summed. Ties go to the
    lower strike. A side with no open interest gives a NaN strike and zero OI; an
    expiry with no OI on either side is skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``max_pain`` / ``pcr``; needs ``open_interest``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, call_wall, call_wall_oi, put_wall,
        put_wall_oi. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "call_wall",
        "call_wall_oi",
        "put_wall",
        "put_wall_oi",
    ]
    if chain.empty or "kind" not in chain.columns or "open_interest" not in chain.columns:
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "strike", "open_interest"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    def _wall(grp):
        # sum OI per strike, sort by strike so idxmax breaks ties on the lower one
        by_strike = grp.groupby("strike")["open_interest"].sum().sort_index()
        by_strike = by_strike[by_strike > 0]
        if by_strike.empty:
            return float("nan"), 0.0
        k = by_strike.idxmax()
        return float(k), float(by_strike.loc[k])

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        call_wall, call_wall_oi = _wall(grp[grp["_kind"] == "call"])
        put_wall, put_wall_oi = _wall(grp[grp["_kind"] == "put"])
        if call_wall_oi == 0.0 and put_wall_oi == 0.0:
            continue
        rows.append(
            {
                "expiry": exp,
                "underlying_price": float(grp["underlying_price"].iloc[0]),
                "call_wall": call_wall,
                "call_wall_oi": call_wall_oi,
                "put_wall": put_wall,
                "put_wall_oi": put_wall_oi,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def oi_profile(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-strike call/put open-interest profile, one row per expiry and strike.

    This is the raw distribution ``oi_walls`` and ``max_pain`` collapse: the call
    and put open interest standing at every strike, side by side, plus the total
    and the net (call minus put) at each. Charted as a histogram it's the open
    interest profile traders read to spot where positioning clusters, not just the
    single heaviest strike.

    Pure summation over ``open_interest``, no IV solve. OI is aggregated per
    strike, so split rows at the same strike are summed; a side with no contracts
    at a strike contributes zero. Strikes with no OI on either side are dropped.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``oi_walls`` / ``max_pain``; needs ``open_interest``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, strike, call_oi, put_oi, total_oi,
        net_oi. One row per (expiry, strike), sorted by expiry then strike.
    """
    cols = [
        "expiry",
        "underlying_price",
        "strike",
        "call_oi",
        "put_oi",
        "total_oi",
        "net_oi",
    ]
    if chain.empty or "kind" not in chain.columns or "open_interest" not in chain.columns:
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "strike", "open_interest"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        price = float(grp["underlying_price"].iloc[0])
        call_oi = grp[grp["_kind"] == "call"].groupby("strike")["open_interest"].sum()
        put_oi = grp[grp["_kind"] == "put"].groupby("strike")["open_interest"].sum()
        for k in sorted(set(call_oi.index) | set(put_oi.index)):
            c = float(call_oi.get(k, 0.0))
            p = float(put_oi.get(k, 0.0))
            if c == 0.0 and p == 0.0:
                continue
            rows.append(
                {
                    "expiry": exp,
                    "underlying_price": price,
                    "strike": float(k),
                    "call_oi": c,
                    "put_oi": p,
                    "total_oi": c + p,
                    "net_oi": c - p,
                }
            )

    return pd.DataFrame(rows, columns=cols)


def volume_profile(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-strike call/put traded-volume profile, one row per expiry and strike.

    The day's-flow companion to ``oi_profile``: where ``oi_profile`` shows the
    standing book, this shows what actually changed hands today. Same per-strike
    layout - call and put volume side by side at every strike, plus the total and
    the net (call minus put) - so a spike here that isn't in the OI profile flags
    fresh positioning before it settles into open interest.

    Pure summation over ``volume``, no IV solve. Volume is aggregated per strike,
    so split rows at the same strike are summed; a side with no contracts at a
    strike contributes zero. Strikes with no volume on either side are dropped.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``oi_profile`` / ``max_pain``; needs ``volume``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, strike, call_volume, put_volume,
        total_volume, net_volume. One row per (expiry, strike), sorted by expiry
        then strike.
    """
    cols = [
        "expiry",
        "underlying_price",
        "strike",
        "call_volume",
        "put_volume",
        "total_volume",
        "net_volume",
    ]
    if chain.empty or "kind" not in chain.columns or "volume" not in chain.columns:
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "strike", "volume"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        price = float(grp["underlying_price"].iloc[0])
        call_vol = grp[grp["_kind"] == "call"].groupby("strike")["volume"].sum()
        put_vol = grp[grp["_kind"] == "put"].groupby("strike")["volume"].sum()
        for k in sorted(set(call_vol.index) | set(put_vol.index)):
            c = float(call_vol.get(k, 0.0))
            p = float(put_vol.get(k, 0.0))
            if c == 0.0 and p == 0.0:
                continue
            rows.append(
                {
                    "expiry": exp,
                    "underlying_price": price,
                    "strike": float(k),
                    "call_volume": c,
                    "put_volume": p,
                    "total_volume": c + p,
                    "net_volume": c - p,
                }
            )

    return pd.DataFrame(rows, columns=cols)
