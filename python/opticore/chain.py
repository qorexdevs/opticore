"""Option chain fetching and enrichment."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _tte_years(expiry: pd.Series, now: datetime) -> pd.Series:
    """Years from ``now`` to each expiry, without pandas datetime constructors.

    pd.to_datetime / pd.DatetimeIndex both route through tslibs
    ._construct_from_dt64_naive, which segfaults in manylinux2014 (pandas 2.3,
    cp312). numpy's datetime64 parser is independent of pandas tslibs.
    """
    if pd.api.types.is_datetime64_any_dtype(expiry):
        ea = expiry.array
        # pandas DatetimeTZDtype and DatetimeArray expose .unit directly;
        # numpy dtype (for stripped naive columns) does not, so fall back to
        # np.datetime_data which works for all datetime64 numpy dtypes.
        unit = (
            getattr(ea.dtype, "unit", None)
            or getattr(ea, "unit", None)
            or np.datetime_data(ea.dtype)[0]
        )
        expiry64 = ea.asi8.view(f"datetime64[{unit}]").astype("datetime64[s]")
    else:
        s = expiry.astype(str).str.strip()
        iso = s.str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True)
        try:
            expiry64 = np.asarray(iso.to_numpy(), dtype="datetime64[s]")
        except (ValueError, TypeError) as e:
            raise ValueError("expiry must be a Timestamp, 'YYYYMMDD', or ISO date string") from e
    now64 = np.datetime64(int(now.timestamp()), "s")
    tte = (expiry64 - now64) / np.timedelta64(1, "s") / (365.25 * 24 * 3600)
    return pd.Series(tte, index=expiry.index)


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
    # Accept either pd.Timestamp column or "YYYYMMDD" / ISO date strings.
    now = datetime.now(timezone.utc)
    df["tte"] = _tte_years(df["expiry"], now).clip(lower=1e-6)  # avoid zero/negative

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
    # Normalize tz-aware timestamps to UTC-naive: pivot_table/groupby on
    # DatetimeTZDtype triggers a tslibs segfault in manylinux2014 (pandas 2.3, cp312).
    if "expiry" in df.columns and pd.api.types.is_datetime64_any_dtype(df["expiry"]):
        ea = df["expiry"].array
        if hasattr(ea, "tz") and ea.tz is not None:
            unit = getattr(ea.dtype, "unit", "ns")
            df["expiry"] = ea.asi8.view(f"datetime64[{unit}]")
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
    tte = _tte_years(p["expiry"], now).clip(lower=1e-6).to_numpy(dtype=np.float64)

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
    p = p.assign(_tte=_tte_years(p["expiry"], now))
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


def expected_move(
    chain: pd.DataFrame,
    sigmas: float = 1.0,
    rate: float = 0.045,
    div_yield: float = 0.0,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Straddle-implied expected move to each expiry, one row per expiry.

    The move the options market is pricing for the underlying between now and
    each expiry, read straight off the ATM vol:

        move = S * atm_iv * sqrt(tte) * sigmas

    That is the lognormal 1-sigma move scaled by ``sigmas``. ``lower`` and
    ``upper`` bracket spot by that amount, so a straddle buyer breaks even
    roughly at the ``sigmas=1`` edges and a range trader watches the band.
    ``move_pct`` is the same as a fraction of spot. Bump ``sigmas`` to 2 for the
    wider ~95% band.

    Built on ``atm_iv`` - expiries whose ATM vol could not be solved are
    skipped, same as there.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``enrich`` / ``atm_iv``.
    sigmas : float
        How many standard deviations to scale the move by (default: 1.0).
    rate : float
        Risk-free rate (default: 0.045).
    div_yield : float
        Continuous dividend yield (default: 0.0).
    price_col : str
        Which price to use (default: 'mid').

    Returns
    -------
    pd.DataFrame
        Columns: expiry, tte, underlying_price, atm_iv, expected_move,
        move_pct, lower, upper. One row per expiry, sorted by expiry.

    Examples
    --------
    >>> import opticore as oc
    >>> oc.expected_move(chain, sigmas=1.0)  # doctest: +SKIP
    """
    cols = [
        "expiry",
        "tte",
        "underlying_price",
        "atm_iv",
        "expected_move",
        "move_pct",
        "lower",
        "upper",
    ]
    atm = atm_iv(chain, rate=rate, div_yield=div_yield, price_col=price_col)
    if atm.empty:
        return pd.DataFrame(columns=cols)

    spot = atm["underlying_price"].to_numpy()
    move = spot * atm["atm_iv"].to_numpy() * np.sqrt(atm["tte"].to_numpy()) * sigmas
    out = pd.DataFrame(
        {
            "expiry": atm["expiry"].to_numpy(),
            "tte": atm["tte"].to_numpy(),
            "underlying_price": spot,
            "atm_iv": atm["atm_iv"].to_numpy(),
            "expected_move": move,
            "move_pct": np.where(spot != 0.0, move / spot, np.nan),
            "lower": spot - move,
            "upper": spot + move,
        },
        columns=cols,
    )
    return out


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
    p = p.assign(_tte=_tte_years(p["expiry"], now))
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


def strangle(
    chain: pd.DataFrame,
    price_col: str = "mid",
    width: int = 1,
) -> pd.DataFrame:
    """Per-expiry OTM strangle cost, breakevens and the implied move.

    A strangle buys an out-of-the-money call above spot and an out-of-the-money
    put below it - cheaper than the ATM straddle but needing a larger move to pay
    off. For each expiry the call leg is the ``width``-th strike above spot and
    the put leg the ``width``-th strike below; ``width=1`` is the nearest pair of
    OTM strikes, ``width=2`` skips one out, and so on. ``strangle_price`` is the
    two legs summed, breakevens sit one strangle width outside each strike, and
    ``implied_move`` is the cost as a fraction of spot.

    Like :func:`straddle` this is pure price arithmetic - no IV solve - and only
    strikes quoting both a call and a put are considered. An expiry is dropped
    when it has no strike ``width`` steps out on either side.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``straddle`` / ``enrich``.
    price_col : str
        Which price to use (default: 'mid').
    width : int
        How many strikes out of the money each leg sits, counting from spot
        (default: 1, the nearest OTM pair). Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, tte, put_strike, call_strike, underlying_price,
        strangle_price, breakeven_low, breakeven_high, implied_move. One row per
        expiry, sorted by expiry.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    cols = [
        "expiry",
        "tte",
        "put_strike",
        "call_strike",
        "underlying_price",
        "strangle_price",
        "breakeven_low",
        "breakeven_high",
        "implied_move",
    ]
    p = _pivot_call_put(chain, price_col)
    if p.empty:
        return pd.DataFrame(columns=cols)

    now = datetime.now(timezone.utc)
    p = p.assign(_tte=_tte_years(p["expiry"], now))

    rows = []
    for exp, grp in p.groupby("expiry", sort=True):
        spot = float(grp["underlying_price"].iloc[0])
        calls = grp[grp["strike"] > spot].sort_values("strike")
        puts = grp[grp["strike"] < spot].sort_values("strike", ascending=False)
        if len(calls) < width or len(puts) < width:
            continue
        call = calls.iloc[width - 1]
        put = puts.iloc[width - 1]
        call_strike = float(call["strike"])
        put_strike = float(put["strike"])
        cost = float(call["call_mid"] + put["put_mid"])
        rows.append(
            {
                "expiry": exp,
                "tte": float(call["_tte"]),
                "put_strike": put_strike,
                "call_strike": call_strike,
                "underlying_price": spot,
                "strangle_price": cost,
                "breakeven_low": put_strike - cost,
                "breakeven_high": call_strike + cost,
                "implied_move": cost / spot if spot > 0 else float("nan"),
            }
        )

    return pd.DataFrame(rows, columns=cols)


def vertical(
    chain: pd.DataFrame,
    kind: str = "call",
    side: str = "bull",
    width: int = 1,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Per-expiry vertical spread cost, max profit/loss and breakeven.

    A vertical spread buys one option and sells another of the same kind and
    expiry at a different strike - the defined-risk workhorse the straddle and
    strangle leave out. Both legs sit around spot: the lower leg is the nearest
    strike at or below spot and the upper leg the ``width``-th strike above it,
    so ``width=1`` is the tightest spread and larger values widen it.

    ``side`` picks the direction and ``kind`` the option type, the four standard
    combinations: a bull call and bear put are debit spreads (you pay to open,
    ``net_debit > 0``), a bear call and bull put are credit spreads (you collect,
    ``net_debit < 0``). Max profit, max loss and the breakeven are read straight
    off the expiry payoff, which is flat outside the strikes and linear between
    them, so no IV solve is needed. Only strikes quoting both a call and a put
    are used; an expiry is dropped when no strike sits at/below spot or there is
    none ``width`` steps above it.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``straddle`` / ``strangle``.
    kind : str
        ``'call'`` or ``'put'`` (default: 'call').
    side : str
        ``'bull'`` or ``'bear'`` (default: 'bull').
    width : int
        How many strikes apart the legs sit (default: 1). Must be >= 1.
    price_col : str
        Which price to use (default: 'mid').

    Returns
    -------
    pd.DataFrame
        Columns: expiry, tte, kind, side, long_strike, short_strike,
        underlying_price, net_debit, max_profit, max_loss, breakeven. One row
        per expiry, sorted by expiry. ``net_debit`` is positive for a debit
        spread, negative for a credit; ``max_loss`` is negative.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    kind = kind.lower()
    side = side.lower()
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if side not in ("bull", "bear"):
        raise ValueError("side must be 'bull' or 'bear'")
    cols = [
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
    ]
    p = _pivot_call_put(chain, price_col)
    if p.empty:
        return pd.DataFrame(columns=cols)

    now = datetime.now(timezone.utc)
    p = p.assign(_tte=_tte_years(p["expiry"], now))

    prem_col = "call_mid" if kind == "call" else "put_mid"
    # bull is long the low strike, bear is long the high strike (both kinds)
    qty_low = 1.0 if side == "bull" else -1.0
    qty_high = -qty_low

    rows = []
    for exp, grp in p.groupby("expiry", sort=True):
        grp = grp.sort_values("strike").reset_index(drop=True)
        spot = float(grp["underlying_price"].iloc[0])
        below = grp[grp["strike"] <= spot]
        if below.empty:
            continue
        lo_idx = int(below.index[-1])
        hi_idx = lo_idx + width
        if hi_idx >= len(grp):
            continue
        low = grp.iloc[lo_idx]
        high = grp.iloc[hi_idx]
        k_low = float(low["strike"])
        k_high = float(high["strike"])
        prem_low = float(low[prem_col])
        prem_high = float(high[prem_col])
        spread_w = k_high - k_low

        net_cost = qty_low * prem_low + qty_high * prem_high
        # payoff is flat outside [k_low, k_high]; evaluate the two kinks
        if kind == "call":
            pnl_low = -net_cost
            pnl_high = qty_low * spread_w - net_cost
        else:
            pnl_low = qty_high * spread_w - net_cost
            pnl_high = -net_cost

        max_profit = max(pnl_low, pnl_high)
        max_loss = min(pnl_low, pnl_high)
        if pnl_low == pnl_high:
            breakeven = float("nan")
        else:
            breakeven = k_low + (0.0 - pnl_low) * spread_w / (pnl_high - pnl_low)

        rows.append(
            {
                "expiry": exp,
                "tte": float(low["_tte"]),
                "kind": kind,
                "side": side,
                "long_strike": k_low if side == "bull" else k_high,
                "short_strike": k_high if side == "bull" else k_low,
                "underlying_price": spot,
                "net_debit": net_cost,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven": breakeven,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def butterfly(
    chain: pd.DataFrame,
    kind: str = "call",
    side: str = "long",
    width: int = 1,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Per-expiry butterfly spread cost, max profit/loss and breakevens.

    A butterfly is the range-bound, defined-risk cousin of the vertical: it
    buys one option, sells two at a middle strike and buys one further out, all
    of the same kind and expiry. The body sits at the strike nearest spot and
    the wings ``width`` strikes either side, so the payoff tents up to a peak at
    the middle strike and is flat outside the wings whatever the strike spacing.

    ``side='long'`` pays to open (``net_debit > 0``) and profits if price pins
    the middle strike; ``side='short'`` collects (``net_debit < 0``) and profits
    if price runs past either wing. ``kind`` picks call or put butterflies, which
    share the same payoff shape. Max profit, max loss and the two breakevens are
    read straight off the expiry payoff - flat outside the wings, linear between
    the kinks - so no IV solve is needed. Only strikes quoting both a call and a
    put are used; an expiry is dropped when the body has no ``width`` strikes on
    one of its sides.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``straddle`` / ``vertical``.
    kind : str
        ``'call'`` or ``'put'`` (default: 'call').
    side : str
        ``'long'`` or ``'short'`` (default: 'long').
    width : int
        How many strikes each wing sits from the body (default: 1). Must be >= 1.
    price_col : str
        Which price to use (default: 'mid').

    Returns
    -------
    pd.DataFrame
        Columns: expiry, tte, kind, side, low_strike, mid_strike, high_strike,
        underlying_price, net_debit, max_profit, max_loss, breakeven_low,
        breakeven_high. One row per expiry, sorted by expiry. ``net_debit`` is
        positive for a long (debit) butterfly, negative for a short (credit);
        ``max_loss`` is negative. Breakevens are ``nan`` when a segment is flat.
    """
    if width < 1:
        raise ValueError("width must be >= 1")
    kind = kind.lower()
    side = side.lower()
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    cols = [
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
    ]
    p = _pivot_call_put(chain, price_col)
    if p.empty:
        return pd.DataFrame(columns=cols)

    now = datetime.now(timezone.utc)
    p = p.assign(_tte=_tte_years(p["expiry"], now))

    prem_col = "call_mid" if kind == "call" else "put_mid"
    sign = 1.0 if side == "long" else -1.0

    rows = []
    for exp, grp in p.groupby("expiry", sort=True):
        grp = grp.sort_values("strike").reset_index(drop=True)
        spot = float(grp["underlying_price"].iloc[0])
        mid_idx = int((grp["strike"] - spot).abs().idxmin())
        lo_idx = mid_idx - width
        hi_idx = mid_idx + width
        if lo_idx < 0 or hi_idx >= len(grp):
            continue

        k_low = float(grp["strike"].iloc[lo_idx])
        k_mid = float(grp["strike"].iloc[mid_idx])
        k_high = float(grp["strike"].iloc[hi_idx])
        prem_low = float(grp[prem_col].iloc[lo_idx])
        prem_mid = float(grp[prem_col].iloc[mid_idx])
        prem_high = float(grp[prem_col].iloc[hi_idx])

        # qty +1/-2/+1 for a long butterfly, flipped for a short
        net_cost = sign * (prem_low - 2.0 * prem_mid + prem_high)

        def _pnl(s: float) -> float:
            if kind == "call":
                legs = max(s - k_low, 0.0) - 2.0 * max(s - k_mid, 0.0) + max(s - k_high, 0.0)
            else:
                legs = max(k_low - s, 0.0) - 2.0 * max(k_mid - s, 0.0) + max(k_high - s, 0.0)
            return sign * legs - net_cost

        pnl_low = _pnl(k_low)
        pnl_mid = _pnl(k_mid)
        pnl_high = _pnl(k_high)
        max_profit = max(pnl_low, pnl_mid, pnl_high)
        max_loss = min(pnl_low, pnl_mid, pnl_high)

        def _cross(ka: float, pa: float, kb: float, pb: float) -> float:
            if pa == pb or (pa > 0) == (pb > 0):
                return float("nan")
            return ka + (0.0 - pa) * (kb - ka) / (pb - pa)

        be_low = _cross(k_low, pnl_low, k_mid, pnl_mid)
        be_high = _cross(k_mid, pnl_mid, k_high, pnl_high)

        rows.append(
            {
                "expiry": exp,
                "tte": float(grp["_tte"].iloc[mid_idx]),
                "kind": kind,
                "side": side,
                "low_strike": k_low,
                "mid_strike": k_mid,
                "high_strike": k_high,
                "underlying_price": spot,
                "net_debit": net_cost,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven_low": be_low,
                "breakeven_high": be_high,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def iron_condor(
    chain: pd.DataFrame,
    side: str = "short",
    gap: int = 1,
    width: int = 1,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Per-expiry iron condor cost, max profit/loss and breakevens.

    An iron condor is two out-of-the-money credit spreads, one in puts below
    spot and one in calls above it, sharing the same expiry. The short strikes
    sit ``gap`` strikes either side of the strike nearest spot and the long
    wings ``width`` strikes further out, so the payoff is a flat plateau between
    the shorts that tapers to a capped loss past either wing.

    ``side='short'`` is the textbook condor: it collects a credit
    (``net_debit < 0``) and profits while price stays inside the short strikes.
    ``side='long'`` flips every leg, pays a debit (``net_debit > 0``) and profits
    on a move past either wing. Max profit, max loss and the two breakevens are
    read straight off the expiry payoff - flat between the shorts, linear across
    the wings - so no IV solve is needed. Only strikes quoting both a call and a
    put are used; an expiry is dropped when a wing runs past the listed strikes.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``straddle`` / ``butterfly``.
    side : str
        ``'short'`` (credit, default) or ``'long'`` (debit).
    gap : int
        How many strikes each short leg sits from the body (default: 1). >= 1.
    width : int
        How many strikes each long wing sits past its short (default: 1). >= 1.
    price_col : str
        Which price to use (default: 'mid').

    Returns
    -------
    pd.DataFrame
        Columns: expiry, tte, side, put_long_strike, put_short_strike,
        call_short_strike, call_long_strike, underlying_price, net_debit,
        max_profit, max_loss, breakeven_low, breakeven_high. One row per expiry,
        sorted by expiry. ``net_debit`` is negative for a short (credit) condor
        and positive for a long (debit); ``max_loss`` is negative.
    """
    if gap < 1:
        raise ValueError("gap must be >= 1")
    if width < 1:
        raise ValueError("width must be >= 1")
    side = side.lower()
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    cols = [
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
    ]
    p = _pivot_call_put(chain, price_col)
    if p.empty:
        return pd.DataFrame(columns=cols)

    now = datetime.now(timezone.utc)
    p = p.assign(_tte=_tte_years(p["expiry"], now))

    sign = 1.0 if side == "long" else -1.0

    rows = []
    for exp, grp in p.groupby("expiry", sort=True):
        grp = grp.sort_values("strike").reset_index(drop=True)
        spot = float(grp["underlying_price"].iloc[0])
        c_idx = int((grp["strike"] - spot).abs().idxmin())
        lp_idx = c_idx - gap - width
        sp_idx = c_idx - gap
        sc_idx = c_idx + gap
        lc_idx = c_idx + gap + width
        if lp_idx < 0 or lc_idx >= len(grp):
            continue

        k_lp = float(grp["strike"].iloc[lp_idx])
        k_sp = float(grp["strike"].iloc[sp_idx])
        k_sc = float(grp["strike"].iloc[sc_idx])
        k_lc = float(grp["strike"].iloc[lc_idx])
        prem_lp = float(grp["put_mid"].iloc[lp_idx])
        prem_sp = float(grp["put_mid"].iloc[sp_idx])
        prem_sc = float(grp["call_mid"].iloc[sc_idx])
        prem_lc = float(grp["call_mid"].iloc[lc_idx])

        # sell the inner strikes, buy the wings -> usually a net credit
        credit = (prem_sp - prem_lp) + (prem_sc - prem_lc)
        net_cost = sign * credit

        def _pnl(s: float) -> float:
            put_spread = max(k_sp - s, 0.0) - max(k_lp - s, 0.0)
            call_spread = max(s - k_sc, 0.0) - max(s - k_lc, 0.0)
            return sign * (put_spread + call_spread - credit)

        pnls = [_pnl(k) for k in (k_lp, k_sp, k_sc, k_lc)]
        max_profit = max(pnls)
        max_loss = min(pnls)

        def _cross(ka: float, pa: float, kb: float, pb: float) -> float:
            if pa == pb or (pa > 0) == (pb > 0):
                return float("nan")
            return ka + (0.0 - pa) * (kb - ka) / (pb - pa)

        be_low = _cross(k_lp, pnls[0], k_sp, pnls[1])
        be_high = _cross(k_sc, pnls[2], k_lc, pnls[3])

        rows.append(
            {
                "expiry": exp,
                "tte": float(grp["_tte"].iloc[c_idx]),
                "side": side,
                "put_long_strike": k_lp,
                "put_short_strike": k_sp,
                "call_short_strike": k_sc,
                "call_long_strike": k_lc,
                "underlying_price": spot,
                "net_debit": net_cost,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "breakeven_low": be_low,
                "breakeven_high": be_high,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def collar(
    chain: pd.DataFrame,
    gap: int = 1,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Per-expiry collar on a long underlying position: floor, cap and net cost.

    A collar protects a held long position by buying an out-of-the-money put
    ``gap`` strikes below spot (a floor) and selling an out-of-the-money call
    ``gap`` strikes above it (a cap), paying for the put with the call premium.
    Unlike the pure-option spreads here it carries the underlying, so its payoff
    is measured against entry at spot: below the put strike the loss is floored,
    above the call strike the gain is capped, and in between it tracks the stock
    one for one.

    ``net_debit`` is the put premium paid minus the call premium collected, so a
    zero-cost collar sits near 0 and a credit collar goes negative. Max profit
    (``call_strike - spot - net_debit``), max loss (``put_strike - spot -
    net_debit``) and the single breakeven (``spot + net_debit``) read straight
    off the expiry payoff - no IV solve. Only strikes quoting both a call and a
    put count, and an expiry is dropped when it has no strike ``gap`` steps out
    on either side.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``straddle`` / ``iron_condor``.
    gap : int
        How many strikes each leg sits from spot (default: 1). Must be >= 1.
    price_col : str
        Which price to use (default: 'mid').

    Returns
    -------
    pd.DataFrame
        Columns: expiry, tte, put_strike, call_strike, underlying_price,
        net_debit, max_profit, max_loss, breakeven. One row per expiry, sorted by
        expiry. ``max_loss`` is negative once the floor sits below entry.
    """
    if gap < 1:
        raise ValueError("gap must be >= 1")
    cols = [
        "expiry",
        "tte",
        "put_strike",
        "call_strike",
        "underlying_price",
        "net_debit",
        "max_profit",
        "max_loss",
        "breakeven",
    ]
    p = _pivot_call_put(chain, price_col)
    if p.empty:
        return pd.DataFrame(columns=cols)

    now = datetime.now(timezone.utc)
    p = p.assign(_tte=_tte_years(p["expiry"], now))

    rows = []
    for exp, grp in p.groupby("expiry", sort=True):
        grp = grp.sort_values("strike").reset_index(drop=True)
        spot = float(grp["underlying_price"].iloc[0])
        c_idx = int((grp["strike"] - spot).abs().idxmin())
        p_idx = c_idx - gap
        call_idx = c_idx + gap
        if p_idx < 0 or call_idx >= len(grp):
            continue

        k_put = float(grp["strike"].iloc[p_idx])
        k_call = float(grp["strike"].iloc[call_idx])
        prem_put = float(grp["put_mid"].iloc[p_idx])
        prem_call = float(grp["call_mid"].iloc[call_idx])
        net_cost = prem_put - prem_call

        rows.append(
            {
                "expiry": exp,
                "tte": float(grp["_tte"].iloc[c_idx]),
                "put_strike": k_put,
                "call_strike": k_call,
                "underlying_price": spot,
                "net_debit": net_cost,
                "max_profit": k_call - spot - net_cost,
                "max_loss": k_put - spot - net_cost,
                "breakeven": spot + net_cost,
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


def max_pain_curve(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-strike max-pain curve, the full payout shape behind ``max_pain``.

    ``max_pain`` reports only the strike that minimises writer payout; this is the
    curve it minimises over. For each expiry and each candidate settlement price S
    (every listed strike) it returns the call pain ``sum(call_oi * max(S - K, 0))``,
    the put pain ``sum(put_oi * max(K - S, 0))``, their total, and a flag marking
    the max-pain strike. Charted against strike it's the pain profile traders read
    to see how sharp the pin is, not just where it sits.

    Pure open-interest arithmetic, no IV solve. Strikes missing OI count as zero;
    an expiry with no open interest at all is skipped, matching ``max_pain``.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``max_pain``; needs ``open_interest``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, strike, call_pain, put_pain,
        total_pain, is_max_pain. One row per (expiry, strike), sorted by expiry
        then strike.
    """
    cols = [
        "expiry",
        "underlying_price",
        "strike",
        "call_pain",
        "put_pain",
        "total_pain",
        "is_max_pain",
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
        strikes = np.sort(grp["strike"].unique())
        call_oi = grp[grp["_kind"] == "call"].groupby("strike")["open_interest"].sum()
        put_oi = grp[grp["_kind"] == "put"].groupby("strike")["open_interest"].sum()
        if float(call_oi.sum() + put_oi.sum()) <= 0:
            continue

        curve = []
        for s in strikes:
            call_pain = sum(float(oi) * (s - k) for k, oi in call_oi.items() if s > k)
            put_pain = sum(float(oi) * (k - s) for k, oi in put_oi.items() if k > s)
            curve.append((float(s), call_pain, put_pain))

        min_total = min(c + p for _, c, p in curve)
        for s, call_pain, put_pain in curve:
            total = call_pain + put_pain
            rows.append(
                {
                    "expiry": exp,
                    "underlying_price": price,
                    "strike": s,
                    "call_pain": call_pain,
                    "put_pain": put_pain,
                    "total_pain": total,
                    "is_max_pain": total == min_total,
                }
            )

    return pd.DataFrame(rows, columns=cols)


def max_pain_distance(chain: pd.DataFrame) -> pd.DataFrame:
    """Signed distance from spot to the max-pain strike, per expiry.

    Builds on :func:`max_pain` and adds ``dist_pct``, the signed gap from the
    underlying to the max-pain strike as a percentage of spot. It is positive
    when max pain sits above spot (the pin pulls up into expiry) and negative
    when it sits below (the pin pulls down), so the desk can read which way and
    how hard the open-interest magnet leans without eyeballing the strike.

    Pure open-interest arithmetic, no IV solve. Expiries skipped by ``max_pain``
    (no open interest at all) are skipped here too.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``max_pain``; needs ``open_interest``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, max_pain_strike, dist_pct. One row per
        expiry, sorted by expiry.
    """
    cols = ["expiry", "underlying_price", "max_pain_strike", "dist_pct"]
    mp = max_pain(chain)
    if mp.empty:
        return pd.DataFrame(columns=cols)

    spot = mp["underlying_price"]
    out = pd.DataFrame(
        {
            "expiry": mp["expiry"],
            "underlying_price": spot,
            "max_pain_strike": mp["max_pain_strike"],
            "dist_pct": (mp["max_pain_strike"] - spot) / spot * 100.0,
        },
        columns=cols,
    )
    return out.reset_index(drop=True)


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


def pcr_by_strike(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-strike put/call ratio, the strike-level view behind ``pcr``.

    ``pcr`` ratios puts against calls per expiry, which reads sentiment over time
    but hides where on the board the positioning sits. This collapses the expiry
    axis instead and keeps a row per strike, so you can see which strikes are
    put-heavy (downside hedging or support) versus call-heavy (upside bets).

    Same arithmetic and NaN rules as ``pcr``: ``open_interest`` is required,
    ``volume`` optional. A ratio is NaN when the call side is zero (division by
    zero) but the row is still emitted so the put total stays visible.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``pcr``; needs ``open_interest``, ``strike`` and ``kind``.

    Returns
    -------
    pd.DataFrame
        Columns: strike, put_oi, call_oi, oi_pcr, put_volume, call_volume,
        volume_pcr. One row per strike, sorted by strike.
    """
    cols = [
        "strike",
        "put_oi",
        "call_oi",
        "oi_pcr",
        "put_volume",
        "call_volume",
        "volume_pcr",
    ]
    if (
        chain.empty
        or "kind" not in chain.columns
        or "open_interest" not in chain.columns
        or "strike" not in chain.columns
    ):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "strike", "open_interest"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    has_volume = "volume" in df.columns

    rows = []
    for strike, grp in df.groupby("strike", sort=True):
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
                "strike": float(strike),
                "put_oi": put_oi,
                "call_oi": call_oi,
                "oi_pcr": oi_pcr,
                "put_volume": put_vol,
                "call_volume": call_vol,
                "volume_pcr": volume_pcr,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def turnover(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry volume-to-open-interest turnover, call and put side.

    Turnover is the day's volume over the standing open interest. A ratio near or
    above 1 means roughly as many contracts changed hands today as were already
    open, which flags fresh positioning rather than the carry of an existing book.
    Reported per side since calls and puts often turn over at very different rates.

    Pure summation, no IV solve. ``open_interest`` is required; ``volume`` is
    optional and the turnover is NaN when the column is missing. A ratio is NaN
    when a side has no open interest (division by zero) but the row is still
    emitted so the volume stays visible.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``pcr`` / ``max_pain``; needs ``open_interest``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, call_volume, call_oi, call_turnover,
        put_volume, put_oi, put_turnover. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "call_volume",
        "call_oi",
        "call_turnover",
        "put_volume",
        "put_oi",
        "put_turnover",
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

        if has_volume:
            call_vol = float(calls["volume"].sum(skipna=True))
            put_vol = float(puts["volume"].sum(skipna=True))
        else:
            call_vol = put_vol = float("nan")

        call_turnover = call_vol / call_oi if call_oi > 0 else float("nan")
        put_turnover = put_vol / put_oi if put_oi > 0 else float("nan")

        rows.append(
            {
                "expiry": exp,
                "underlying_price": float(grp["underlying_price"].iloc[0]),
                "call_volume": call_vol,
                "call_oi": call_oi,
                "call_turnover": call_turnover,
                "put_volume": put_vol,
                "put_oi": put_oi,
                "put_turnover": put_turnover,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def turnover_by_strike(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-strike volume-to-open-interest turnover, the strike view behind ``turnover``.

    ``turnover`` ratios the day's volume against standing open interest per expiry,
    which reads fresh positioning over time but hides which strikes are churning.
    This collapses the expiry axis and keeps a row per strike, so a strike where
    today's volume rivals its open book stands out as where new money is working,
    versus strikes carrying an old, quiet position. Reported per side since calls
    and puts often turn over at very different rates.

    Same arithmetic and NaN rules as ``turnover``: ``open_interest`` is required,
    ``volume`` optional and the turnover is NaN when missing. A ratio is NaN when a
    side has no open interest at that strike, but the row is still emitted so the
    volume stays visible.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``pcr_by_strike``; needs ``open_interest``, ``strike`` and
        ``kind``.

    Returns
    -------
    pd.DataFrame
        Columns: strike, call_volume, call_oi, call_turnover, put_volume, put_oi,
        put_turnover. One row per strike, sorted by strike.
    """
    cols = [
        "strike",
        "call_volume",
        "call_oi",
        "call_turnover",
        "put_volume",
        "put_oi",
        "put_turnover",
    ]
    if (
        chain.empty
        or "kind" not in chain.columns
        or "open_interest" not in chain.columns
        or "strike" not in chain.columns
    ):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "strike", "open_interest"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    has_volume = "volume" in df.columns

    rows = []
    for strike, grp in df.groupby("strike", sort=True):
        calls = grp[grp["_kind"] == "call"]
        puts = grp[grp["_kind"] == "put"]
        call_oi = float(calls["open_interest"].sum())
        put_oi = float(puts["open_interest"].sum())

        if has_volume:
            call_vol = float(calls["volume"].sum(skipna=True))
            put_vol = float(puts["volume"].sum(skipna=True))
        else:
            call_vol = put_vol = float("nan")

        call_turnover = call_vol / call_oi if call_oi > 0 else float("nan")
        put_turnover = put_vol / put_oi if put_oi > 0 else float("nan")

        rows.append(
            {
                "strike": float(strike),
                "call_volume": call_vol,
                "call_oi": call_oi,
                "call_turnover": call_turnover,
                "put_volume": put_vol,
                "put_oi": put_oi,
                "put_turnover": put_turnover,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def liquidity(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry bid-ask spread, a liquidity gauge for picking tradeable strikes.

    A tight spread means you can work an order near mid; a wide one is a tax on
    every fill. Reported per expiry as the median across that expiry's quotes,
    both absolute (``ask - bid``) and relative to mid, since a 0.10 spread is
    cheap on a 5.00 option and dear on a 0.20 one. The widest relative spread is
    kept too, to flag an expiry with one untradeable strike hiding behind an
    otherwise decent median.

    Pure arithmetic, no IV solve. Needs ``bid`` and ``ask``; ``mid`` is used when
    present, otherwise the midpoint is taken. Quotes with a missing or crossed
    (``bid > ask``) market or a non-positive mid are dropped, and an expiry left
    with nothing is omitted.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``max_pain`` / ``enrich``; needs ``bid`` and ``ask``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, n_quotes, median_spread,
        median_rel_spread, max_rel_spread. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "n_quotes",
        "median_spread",
        "median_rel_spread",
        "max_rel_spread",
    ]
    if chain.empty or "bid" not in chain.columns or "ask" not in chain.columns:
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    mid = df["mid"] if "mid" in df.columns else (df["bid"] + df["ask"]) / 2.0
    df["_mid"] = mid
    df = df.dropna(subset=["bid", "ask", "_mid"])
    df = df[(df["ask"] >= df["bid"]) & (df["_mid"] > 0)]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["_spread"] = df["ask"] - df["bid"]
    df["_rel"] = df["_spread"] / df["_mid"]

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        rows.append(
            {
                "expiry": exp,
                "underlying_price": float(grp["underlying_price"].iloc[0]),
                "n_quotes": int(len(grp)),
                "median_spread": float(grp["_spread"].median()),
                "median_rel_spread": float(grp["_rel"].median()),
                "max_rel_spread": float(grp["_rel"].max()),
            }
        )

    return pd.DataFrame(rows, columns=cols)


def liquidity_by_strike(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-strike bid-ask spread, the contract-level view behind ``liquidity``.

    ``liquidity`` medians the spread per expiry, which ranks expiries but hides
    the strikes; when you're picking the actual contract to trade you want the
    spread on each one. This keeps a row per (expiry, strike, kind) with the
    ``bid``, ``ask``, ``mid``, absolute ``spread`` and relative ``rel_spread``,
    no aggregation - the raw distribution the median collapses.

    Same arithmetic and drop rules as ``liquidity``: ``mid`` is used when present,
    otherwise the midpoint of ``bid``/``ask``; quotes with a missing or crossed
    (``bid > ask``) market or a non-positive mid are dropped.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``liquidity``; needs ``bid``, ``ask`` and ``kind``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, strike, kind, bid, ask, mid, spread, rel_spread. One row
        per quote, sorted by expiry then strike then kind.
    """
    cols = ["expiry", "strike", "kind", "bid", "ask", "mid", "spread", "rel_spread"]
    if (
        chain.empty
        or "bid" not in chain.columns
        or "ask" not in chain.columns
        or "kind" not in chain.columns
    ):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    mid = df["mid"] if "mid" in df.columns else (df["bid"] + df["ask"]) / 2.0
    df["_mid"] = mid
    df = df.dropna(subset=["_kind", "strike", "bid", "ask", "_mid"])
    df = df[(df["ask"] >= df["bid"]) & (df["_mid"] > 0)]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["_spread"] = df["ask"] - df["bid"]
    df["_rel"] = df["_spread"] / df["_mid"]
    df = df.sort_values(["expiry", "strike", "_kind"], kind="stable")

    rows = [
        {
            "expiry": r["expiry"],
            "strike": float(r["strike"]),
            "kind": r["_kind"],
            "bid": float(r["bid"]),
            "ask": float(r["ask"]),
            "mid": float(r["_mid"]),
            "spread": float(r["_spread"]),
            "rel_spread": float(r["_rel"]),
        }
        for _, r in df.iterrows()
    ]

    return pd.DataFrame(rows, columns=cols)


def dollar_volume(
    chain: pd.DataFrame,
    price_col: str = "mid",
    contract_size: float = 100.0,
) -> pd.DataFrame:
    """Per-expiry premium in dollars traded and standing, call and put side.

    Where ``pcr`` and ``turnover`` count contracts, this weights each strike by
    its price: dollar volume is ``price * volume * contract_size`` summed per
    side, dollar open interest the same for the standing book. A handful of deep
    in-the-money contracts can carry more premium than a swarm of cheap wings, so
    the dollar put/call ratio often reads differently than the count one - it
    shows where the money actually sits, not just the contract tally.

    Pure price arithmetic, no IV solve. ``open_interest`` and the price column are
    required; ``volume`` is optional and its dollar figure is NaN when missing.
    Strikes with no quoted price are dropped. A dollar PCR is NaN when the call
    side is zero, but the row is still emitted so the put total stays visible.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``pcr`` / ``turnover``; needs ``open_interest`` and
        ``price_col``.
    price_col : str
        Which price to weight by (default: 'mid').
    contract_size : float
        Multiplier per contract (default: 100, one equity option).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, call_dollar_volume, put_dollar_volume,
        dollar_volume_pcr, call_dollar_oi, put_dollar_oi, dollar_oi_pcr. One row
        per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "call_dollar_volume",
        "put_dollar_volume",
        "dollar_volume_pcr",
        "call_dollar_oi",
        "put_dollar_oi",
        "dollar_oi_pcr",
    ]
    if (
        chain.empty
        or "kind" not in chain.columns
        or "open_interest" not in chain.columns
        or price_col not in chain.columns
    ):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "open_interest", price_col])
    if df.empty:
        return pd.DataFrame(columns=cols)

    has_volume = "volume" in df.columns

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        calls = grp[grp["_kind"] == "call"]
        puts = grp[grp["_kind"] == "put"]
        call_doi = float((calls[price_col] * calls["open_interest"]).sum()) * contract_size
        put_doi = float((puts[price_col] * puts["open_interest"]).sum()) * contract_size
        doi_pcr = put_doi / call_doi if call_doi > 0 else float("nan")

        if has_volume:
            call_dvol = float((calls[price_col] * calls["volume"]).sum(skipna=True)) * contract_size
            put_dvol = float((puts[price_col] * puts["volume"]).sum(skipna=True)) * contract_size
            dvol_pcr = put_dvol / call_dvol if call_dvol > 0 else float("nan")
        else:
            call_dvol = put_dvol = dvol_pcr = float("nan")

        rows.append(
            {
                "expiry": exp,
                "underlying_price": float(grp["underlying_price"].iloc[0]),
                "call_dollar_volume": call_dvol,
                "put_dollar_volume": put_dvol,
                "dollar_volume_pcr": dvol_pcr,
                "call_dollar_oi": call_doi,
                "put_dollar_oi": put_doi,
                "dollar_oi_pcr": doi_pcr,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def dollar_volume_by_strike(
    chain: pd.DataFrame,
    price_col: str = "mid",
    contract_size: float = 100.0,
) -> pd.DataFrame:
    """Per-strike premium in dollars, the strike-level view behind ``dollar_volume``.

    ``dollar_volume`` sums premium per expiry, which tracks where the money sits
    over time but hides the strikes it clusters at. This collapses the expiry axis
    and keeps a row per strike, so you can see which strikes hold the most premium
    rather than the most contracts - a few deep ITM strikes can dominate the dollar
    book while barely registering in the contract count.

    Same arithmetic and NaN rules as ``dollar_volume``: ``open_interest`` and the
    price column are required, ``volume`` optional and its dollar figure NaN when
    missing. Strikes with no quoted price are dropped. A dollar PCR is NaN when the
    call side is zero but the row is still emitted so the put total stays visible.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``dollar_volume``; needs ``open_interest``, ``strike``,
        ``kind`` and ``price_col``.
    price_col : str
        Which price to weight by (default: 'mid').
    contract_size : float
        Multiplier per contract (default: 100, one equity option).

    Returns
    -------
    pd.DataFrame
        Columns: strike, call_dollar_volume, put_dollar_volume, dollar_volume_pcr,
        call_dollar_oi, put_dollar_oi, dollar_oi_pcr. One row per strike, sorted by
        strike.
    """
    cols = [
        "strike",
        "call_dollar_volume",
        "put_dollar_volume",
        "dollar_volume_pcr",
        "call_dollar_oi",
        "put_dollar_oi",
        "dollar_oi_pcr",
    ]
    if (
        chain.empty
        or "kind" not in chain.columns
        or "open_interest" not in chain.columns
        or "strike" not in chain.columns
        or price_col not in chain.columns
    ):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "strike", "open_interest", price_col])
    if df.empty:
        return pd.DataFrame(columns=cols)

    has_volume = "volume" in df.columns

    rows = []
    for strike, grp in df.groupby("strike", sort=True):
        calls = grp[grp["_kind"] == "call"]
        puts = grp[grp["_kind"] == "put"]
        call_doi = float((calls[price_col] * calls["open_interest"]).sum()) * contract_size
        put_doi = float((puts[price_col] * puts["open_interest"]).sum()) * contract_size
        doi_pcr = put_doi / call_doi if call_doi > 0 else float("nan")

        if has_volume:
            call_dvol = float((calls[price_col] * calls["volume"]).sum(skipna=True)) * contract_size
            put_dvol = float((puts[price_col] * puts["volume"]).sum(skipna=True)) * contract_size
            dvol_pcr = put_dvol / call_dvol if call_dvol > 0 else float("nan")
        else:
            call_dvol = put_dvol = dvol_pcr = float("nan")

        rows.append(
            {
                "strike": float(strike),
                "call_dollar_volume": call_dvol,
                "put_dollar_volume": put_dvol,
                "dollar_volume_pcr": dvol_pcr,
                "call_dollar_oi": call_doi,
                "put_dollar_oi": put_doi,
                "dollar_oi_pcr": doi_pcr,
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


def wall_distance(chain: pd.DataFrame) -> pd.DataFrame:
    """How far spot sits from each open-interest wall, per expiry.

    Builds on :func:`oi_walls` and adds the signed distance from the underlying
    to each wall as a percentage of spot. ``call_wall_dist_pct`` is positive when
    the call wall sits overhead (resistance above spot) and ``put_wall_dist_pct``
    is negative when the put wall sits below (support under spot). The desk reads
    these to judge how much room price has before it runs into pinned hedging
    flow: a call wall 1% away is a much tighter ceiling than one 6% away.

    Pure arithmetic on the ``oi_walls`` output, no IV solve. A side with no open
    interest carries a NaN wall and a NaN distance. Expiries skipped by
    ``oi_walls`` (no OI on either side) are skipped here too.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``oi_walls``; needs ``open_interest``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, call_wall, call_wall_dist_pct,
        put_wall, put_wall_dist_pct. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "call_wall",
        "call_wall_dist_pct",
        "put_wall",
        "put_wall_dist_pct",
    ]
    walls = oi_walls(chain)
    if walls.empty:
        return pd.DataFrame(columns=cols)

    spot = walls["underlying_price"]
    out = pd.DataFrame(
        {
            "expiry": walls["expiry"],
            "underlying_price": spot,
            "call_wall": walls["call_wall"],
            "call_wall_dist_pct": (walls["call_wall"] - spot) / spot * 100.0,
            "put_wall": walls["put_wall"],
            "put_wall_dist_pct": (walls["put_wall"] - spot) / spot * 100.0,
        },
        columns=cols,
    )
    return out.reset_index(drop=True)


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


def oi_concentration(chain: pd.DataFrame) -> pd.DataFrame:
    """How tightly open interest clusters across strikes, per expiry.

    Collapses the ``oi_profile`` distribution into a few concentration numbers.
    ``top_share`` is the fraction of total OI sitting at the single heaviest
    strike and ``top3_share`` the fraction in the three heaviest; ``hhi`` is the
    Herfindahl index (sum of squared per-strike shares), running from ~0 when OI
    is spread thin over many strikes up to 1.0 when it all sits at one. The desk
    reads a high reading as pin risk: positioning bunched at a couple of strikes
    pulls harder into expiry than the same OI smeared across the ladder, even
    when the single wall looks the same.

    Total OI here is call plus put at each strike (``oi_profile``'s ``total_oi``),
    so it measures overall clustering, not one side. Pure arithmetic, no IV
    solve. Ties for the top strike go to the lower one. Expiries dropped by
    ``oi_profile`` (no OI anywhere) are skipped here too.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``oi_profile``; needs ``open_interest``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, n_strikes, total_oi, top_strike,
        top_share, top3_share, hhi. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "n_strikes",
        "total_oi",
        "top_strike",
        "top_share",
        "top3_share",
        "hhi",
    ]
    prof = oi_profile(chain)
    if prof.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in prof.groupby("expiry", sort=True):
        by_strike = grp.groupby("strike")["total_oi"].sum().sort_index()
        by_strike = by_strike[by_strike > 0]
        total = float(by_strike.sum())
        shares = by_strike / total
        top_strike = float(by_strike.idxmax())
        top3 = by_strike.sort_values(ascending=False).head(3)
        rows.append(
            {
                "expiry": exp,
                "underlying_price": float(grp["underlying_price"].iloc[0]),
                "n_strikes": int(len(by_strike)),
                "total_oi": total,
                "top_strike": top_strike,
                "top_share": float(by_strike.loc[top_strike] / total),
                "top3_share": float(top3.sum() / total),
                "hhi": float((shares**2).sum()),
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


def volume_walls(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry call and put traded-volume walls.

    The day's-flow companion to ``oi_walls``: where ``oi_walls`` finds the strike
    carrying the most standing open interest on each side, this finds the strike
    that traded the most contracts today. A volume wall that isn't an OI wall flags
    where fresh flow is concentrating before it settles into open interest.

    Pure summation over ``volume``, no IV solve. Volume is aggregated per strike,
    so split call/put rows at the same strike are summed. Ties go to the lower
    strike. A side with no volume gives a NaN strike and zero volume; an expiry
    with no volume on either side is skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``oi_walls`` / ``volume_profile``; needs ``volume``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, call_wall, call_wall_volume, put_wall,
        put_wall_volume. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "call_wall",
        "call_wall_volume",
        "put_wall",
        "put_wall_volume",
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

    def _wall(grp):
        # sum volume per strike, sort by strike so idxmax breaks ties on the lower one
        by_strike = grp.groupby("strike")["volume"].sum().sort_index()
        by_strike = by_strike[by_strike > 0]
        if by_strike.empty:
            return float("nan"), 0.0
        k = by_strike.idxmax()
        return float(k), float(by_strike.loc[k])

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        call_wall, call_wall_volume = _wall(grp[grp["_kind"] == "call"])
        put_wall, put_wall_volume = _wall(grp[grp["_kind"] == "put"])
        if call_wall_volume == 0.0 and put_wall_volume == 0.0:
            continue
        rows.append(
            {
                "expiry": exp,
                "underlying_price": float(grp["underlying_price"].iloc[0]),
                "call_wall": call_wall,
                "call_wall_volume": call_wall_volume,
                "put_wall": put_wall,
                "put_wall_volume": put_wall_volume,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def volume_wall_distance(chain: pd.DataFrame) -> pd.DataFrame:
    """How far spot sits from each traded-volume wall, per expiry.

    The day's-flow companion to :func:`wall_distance`: where that measures the
    gap to the standing open-interest walls, this measures it to the strikes that
    traded the most today. Builds on :func:`volume_walls` and adds the signed
    distance from the underlying to each wall as a percentage of spot.
    ``call_wall_dist_pct`` is positive when the busy call strike sits overhead and
    ``put_wall_dist_pct`` negative when the busy put strike sits below. A volume
    wall pinned right on top of spot is fresh flow being defended at the money;
    one several percent out is positioning for a move, not a pin.

    Pure arithmetic on the ``volume_walls`` output, no IV solve. A side with no
    volume carries a NaN wall and a NaN distance. Expiries skipped by
    ``volume_walls`` (no volume on either side) are skipped here too.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``volume_walls``; needs ``volume``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, call_wall, call_wall_dist_pct,
        put_wall, put_wall_dist_pct. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "call_wall",
        "call_wall_dist_pct",
        "put_wall",
        "put_wall_dist_pct",
    ]
    walls = volume_walls(chain)
    if walls.empty:
        return pd.DataFrame(columns=cols)

    spot = walls["underlying_price"]
    out = pd.DataFrame(
        {
            "expiry": walls["expiry"],
            "underlying_price": spot,
            "call_wall": walls["call_wall"],
            "call_wall_dist_pct": (walls["call_wall"] - spot) / spot * 100.0,
            "put_wall": walls["put_wall"],
            "put_wall_dist_pct": (walls["put_wall"] - spot) / spot * 100.0,
        },
        columns=cols,
    )
    return out.reset_index(drop=True)


def volume_concentration(chain: pd.DataFrame) -> pd.DataFrame:
    """How tightly traded volume clusters across strikes, per expiry.

    The day's-flow companion to ``oi_concentration``: where that reads clustering
    in the standing book, this reads it in what changed hands today. Collapses the
    ``volume_profile`` distribution into a few concentration numbers. ``top_share``
    is the fraction of total volume at the single busiest strike and ``top3_share``
    the fraction in the three busiest; ``hhi`` is the Herfindahl index (sum of
    squared per-strike shares), near 0 when flow is smeared across many strikes and
    1.0 when it all trades at one. Concentrated volume means the day's flow is
    aimed at a couple of strikes - fresh positioning bunching up before it settles
    into open interest - rather than routine two-sided churn spread down the ladder.

    Total volume is call plus put at each strike (``volume_profile``'s
    ``total_volume``), so it measures overall clustering, not one side. Pure
    arithmetic, no IV solve. Ties for the top strike go to the lower one. Expiries
    dropped by ``volume_profile`` (no volume anywhere) are skipped here too.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``volume_profile``; needs ``volume``.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, n_strikes, total_volume, top_strike,
        top_share, top3_share, hhi. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "n_strikes",
        "total_volume",
        "top_strike",
        "top_share",
        "top3_share",
        "hhi",
    ]
    prof = volume_profile(chain)
    if prof.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in prof.groupby("expiry", sort=True):
        by_strike = grp.groupby("strike")["total_volume"].sum().sort_index()
        by_strike = by_strike[by_strike > 0]
        if by_strike.empty:
            continue
        total = float(by_strike.sum())
        shares = by_strike / total
        top_strike = float(by_strike.idxmax())
        top3 = by_strike.sort_values(ascending=False).head(3)
        rows.append(
            {
                "expiry": exp,
                "underlying_price": float(grp["underlying_price"].iloc[0]),
                "n_strikes": int(len(by_strike)),
                "total_volume": total,
                "top_strike": top_strike,
                "top_share": float(by_strike.loc[top_strike] / total),
                "top3_share": float(top3.sum() / total),
                "hhi": float((shares**2).sum()),
            }
        )

    return pd.DataFrame(rows, columns=cols)


def delta_exposure(chain: pd.DataFrame, contract_size: float = 100.0) -> pd.DataFrame:
    """Per-expiry dealer delta exposure (DEX) from open interest and Greeks.

    The directional sibling of ``gamma_exposure``. Under the same dealer
    convention - long call, short put - per option

        dex_i = sign * delta_i * open_interest_i * contract_size * S

    with ``sign = +1`` for calls and ``-1`` for puts and ``S`` the underlying
    price. That is dollar delta: how many dollars of spot the writing side is net
    long or short through these strikes. Positive net DEX means dealers carry long
    deltas and lean to sell into strength; negative means they are short and buy
    it. ``delta_wall_strike`` is the strike holding the most gross dollar delta
    (both sides, sign ignored), where directional hedging concentrates.

    Needs the ``delta`` column from ``enrich`` plus ``open_interest``. Rows with
    NaN delta or no open interest contribute nothing; an expiry with no usable
    delta is skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        An enriched chain (see ``enrich``); needs ``delta`` and ``open_interest``.
    contract_size : float
        Shares per contract (default: 100).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, call_dex, put_dex, net_dex,
        delta_wall_strike. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "call_dex",
        "put_dex",
        "net_dex",
        "delta_wall_strike",
    ]
    needed = {"kind", "delta", "open_interest", "strike", "underlying_price"}
    if chain.empty or not needed.issubset(chain.columns):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "delta", "open_interest", "strike"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        spot = float(grp["underlying_price"].iloc[0])
        scale = contract_size * spot
        sign = np.where(grp["_kind"].to_numpy() == "call", 1.0, -1.0)
        dex = sign * grp["delta"].to_numpy() * grp["open_interest"].to_numpy() * scale
        gross = np.abs(grp["delta"].to_numpy()) * grp["open_interest"].to_numpy() * scale
        grp = grp.assign(_dex=dex, _gross=gross)
        call_dex = float(grp.loc[grp["_kind"] == "call", "_dex"].sum())
        put_dex = float(grp.loc[grp["_kind"] == "put", "_dex"].sum())

        by_strike = grp.groupby("strike")["_gross"].sum().sort_index()
        if (by_strike > 0).any():
            wall = float(by_strike.idxmax())
        else:
            continue

        rows.append(
            {
                "expiry": exp,
                "underlying_price": spot,
                "call_dex": call_dex,
                "put_dex": put_dex,
                "net_dex": call_dex + put_dex,
                "delta_wall_strike": wall,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def delta_exposure_by_strike(chain: pd.DataFrame, contract_size: float = 100.0) -> pd.DataFrame:
    """Per-strike dealer delta exposure, the profile behind ``delta_exposure``.

    ``delta_exposure`` nets DEX to one row per expiry, which gives the directional
    lean but hides where the dealer deltas sit. This keeps the strike axis: for
    each expiry and strike it returns ``call_dex``, ``put_dex``, ``net_dex`` and a
    ``cumulative_net_dex`` running up the strikes, plus ``is_delta_wall`` on the
    strike carrying the most gross dollar delta. Charted against strike it's the
    DEX profile that sits next to the GEX profile, and the sign change in
    ``cumulative_net_dex`` brackets the strike where net dealer delta flips.

    Same scaling and NaN rules as ``delta_exposure``: per option
    ``sign * delta * open_interest * contract_size * S`` with ``sign`` +1 for
    calls and -1 for puts and ``S`` the underlying price. Rows with NaN delta or
    no open interest contribute nothing; an expiry with no usable delta is
    skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        An enriched chain (see ``enrich``); needs ``delta`` and ``open_interest``.
    contract_size : float
        Shares per contract (default: 100).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, strike, call_dex, put_dex, net_dex,
        cumulative_net_dex, is_delta_wall. One row per (expiry, strike), sorted
        by expiry then strike.
    """
    cols = [
        "expiry",
        "underlying_price",
        "strike",
        "call_dex",
        "put_dex",
        "net_dex",
        "cumulative_net_dex",
        "is_delta_wall",
    ]
    needed = {"kind", "delta", "open_interest", "strike", "underlying_price"}
    if chain.empty or not needed.issubset(chain.columns):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "delta", "open_interest", "strike"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        spot = float(grp["underlying_price"].iloc[0])
        scale = contract_size * spot
        sign = np.where(grp["_kind"].to_numpy() == "call", 1.0, -1.0)
        grp = grp.assign(
            _dex=sign * grp["delta"].to_numpy() * grp["open_interest"].to_numpy() * scale,
            _gross=np.abs(grp["delta"].to_numpy()) * grp["open_interest"].to_numpy() * scale,
        )
        gross_by_strike = grp.groupby("strike")["_gross"].sum()
        if not (gross_by_strike > 0).any():
            continue
        wall = float(gross_by_strike.idxmax())

        call_dex = grp.loc[grp["_kind"] == "call"].groupby("strike")["_dex"].sum()
        put_dex = grp.loc[grp["_kind"] == "put"].groupby("strike")["_dex"].sum()

        cum = 0.0
        for s in np.sort(grp["strike"].unique()):
            c = float(call_dex.get(s, 0.0))
            p = float(put_dex.get(s, 0.0))
            net = c + p
            cum += net
            rows.append(
                {
                    "expiry": exp,
                    "underlying_price": spot,
                    "strike": float(s),
                    "call_dex": c,
                    "put_dex": p,
                    "net_dex": net,
                    "cumulative_net_dex": cum,
                    "is_delta_wall": float(s) == wall,
                }
            )

    return pd.DataFrame(rows, columns=cols)


def gamma_exposure(chain: pd.DataFrame, contract_size: float = 100.0) -> pd.DataFrame:
    """Per-expiry dealer gamma exposure (GEX) from open interest and Greeks.

    Dealer gamma exposure approximates how much the option-writing side has to
    hedge as spot moves. The common convention treats dealers as long call gamma
    and short put gamma, so per option

        gex_i = sign * gamma_i * open_interest_i * contract_size * S**2 * 0.01

    with ``sign = +1`` for calls and ``-1`` for puts and ``S`` the underlying
    price. That is dollar gamma per 1% move in spot. Positive net GEX means
    dealers buy dips and sell rips (price-dampening); negative net GEX means they
    chase the move (price-amplifying). ``gamma_wall_strike`` is the strike carrying
    the most gross dollar gamma (both sides, sign ignored) - where hedging flow
    concentrates regardless of which way it leans.

    Needs the ``gamma`` column from ``enrich`` plus ``open_interest``. Rows with
    NaN gamma or no open interest contribute nothing; an expiry with no usable
    gamma is skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        An enriched chain (see ``enrich``); needs ``gamma`` and ``open_interest``.
    contract_size : float
        Shares per contract (default: 100).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, call_gex, put_gex, net_gex,
        gamma_wall_strike. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "call_gex",
        "put_gex",
        "net_gex",
        "gamma_wall_strike",
    ]
    needed = {"kind", "gamma", "open_interest", "strike", "underlying_price"}
    if chain.empty or not needed.issubset(chain.columns):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "gamma", "open_interest", "strike"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        spot = float(grp["underlying_price"].iloc[0])
        scale = contract_size * spot * spot * 0.01
        sign = np.where(grp["_kind"].to_numpy() == "call", 1.0, -1.0)
        gex = sign * grp["gamma"].to_numpy() * grp["open_interest"].to_numpy() * scale
        gross = np.abs(grp["gamma"].to_numpy()) * grp["open_interest"].to_numpy() * scale
        grp = grp.assign(_gex=gex, _gross=gross)
        call_gex = float(grp.loc[grp["_kind"] == "call", "_gex"].sum())
        put_gex = float(grp.loc[grp["_kind"] == "put", "_gex"].sum())

        by_strike = grp.groupby("strike")["_gross"].sum().sort_index()
        if (by_strike > 0).any():
            wall = float(by_strike.idxmax())
        else:
            continue

        rows.append(
            {
                "expiry": exp,
                "underlying_price": spot,
                "call_gex": call_gex,
                "put_gex": put_gex,
                "net_gex": call_gex + put_gex,
                "gamma_wall_strike": wall,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def gamma_exposure_by_strike(chain: pd.DataFrame, contract_size: float = 100.0) -> pd.DataFrame:
    """Per-strike dealer gamma exposure, the profile behind ``gamma_exposure``.

    ``gamma_exposure`` nets GEX to one row per expiry, which gives the lean but
    hides where the hedging flow sits. This keeps the strike axis: for each
    expiry and strike it returns ``call_gex``, ``put_gex``, ``net_gex`` and a
    ``cumulative_net_gex`` running up the strikes, plus ``is_gamma_wall`` on the
    strike carrying the most gross dollar gamma. Charted against strike it's the
    GEX profile traders read, and the sign change in ``cumulative_net_gex``
    brackets the strike where net dealer gamma flips.

    Same scaling and NaN rules as ``gamma_exposure``: per option
    ``sign * gamma * open_interest * contract_size * S**2 * 0.01`` with ``sign``
    +1 for calls and -1 for puts and ``S`` the underlying price. Rows with NaN
    gamma or no open interest contribute nothing; an expiry with no usable gamma
    is skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        An enriched chain (see ``enrich``); needs ``gamma`` and ``open_interest``.
    contract_size : float
        Shares per contract (default: 100).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, strike, call_gex, put_gex, net_gex,
        cumulative_net_gex, is_gamma_wall. One row per (expiry, strike), sorted
        by expiry then strike.
    """
    cols = [
        "expiry",
        "underlying_price",
        "strike",
        "call_gex",
        "put_gex",
        "net_gex",
        "cumulative_net_gex",
        "is_gamma_wall",
    ]
    needed = {"kind", "gamma", "open_interest", "strike", "underlying_price"}
    if chain.empty or not needed.issubset(chain.columns):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "gamma", "open_interest", "strike"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        spot = float(grp["underlying_price"].iloc[0])
        scale = contract_size * spot * spot * 0.01
        sign = np.where(grp["_kind"].to_numpy() == "call", 1.0, -1.0)
        grp = grp.assign(
            _gex=sign * grp["gamma"].to_numpy() * grp["open_interest"].to_numpy() * scale,
            _gross=np.abs(grp["gamma"].to_numpy()) * grp["open_interest"].to_numpy() * scale,
        )
        gross_by_strike = grp.groupby("strike")["_gross"].sum()
        if not (gross_by_strike > 0).any():
            continue
        wall = float(gross_by_strike.idxmax())

        call_gex = grp.loc[grp["_kind"] == "call"].groupby("strike")["_gex"].sum()
        put_gex = grp.loc[grp["_kind"] == "put"].groupby("strike")["_gex"].sum()

        cum = 0.0
        for s in np.sort(grp["strike"].unique()):
            c = float(call_gex.get(s, 0.0))
            p = float(put_gex.get(s, 0.0))
            net = c + p
            cum += net
            rows.append(
                {
                    "expiry": exp,
                    "underlying_price": spot,
                    "strike": float(s),
                    "call_gex": c,
                    "put_gex": p,
                    "net_gex": net,
                    "cumulative_net_gex": cum,
                    "is_gamma_wall": float(s) == wall,
                }
            )

    return pd.DataFrame(rows, columns=cols)


def gamma_concentration(chain: pd.DataFrame, contract_size: float = 100.0) -> pd.DataFrame:
    """How tightly dealer gamma clusters across strikes, per expiry.

    The GEX analogue of ``oi_concentration``: it collapses the
    ``gamma_exposure_by_strike`` profile into a few numbers. Weighting is gross
    dollar gamma at each strike (``|call_gex| + |put_gex|``, sign ignored), so it
    measures where hedging flow bunches regardless of which way it leans.
    ``top_share`` is the fraction of gross gamma at the single heaviest strike,
    ``top3_share`` the fraction in the three heaviest, and ``hhi`` the Herfindahl
    index running from ~0 when gamma is smeared over many strikes up to 1.0 when
    it all sits at one. Read like ``oi_concentration`` but sharper into expiry:
    gamma peaks at-the-money, so a high reading usually means one or two near-spot
    strikes carry the whole hedging book and pin harder than the same gamma spread
    down the ladder.

    Built on ``gamma_exposure_by_strike`` and inherits its scaling and NaN rules:
    per option ``sign * gamma * open_interest * contract_size * S**2 * 0.01``.
    Expiries with no usable gamma are skipped there and so absent here too. Ties
    for the top strike go to the lower one.

    Parameters
    ----------
    chain : pd.DataFrame
        An enriched chain (see ``enrich``); needs ``gamma`` and ``open_interest``.
    contract_size : float
        Shares per contract (default: 100).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, n_strikes, gross_gex, top_strike,
        top_share, top3_share, hhi. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "n_strikes",
        "gross_gex",
        "top_strike",
        "top_share",
        "top3_share",
        "hhi",
    ]
    prof = gamma_exposure_by_strike(chain, contract_size=contract_size)
    if prof.empty:
        return pd.DataFrame(columns=cols)

    prof = prof.assign(_gross=prof["call_gex"].abs() + prof["put_gex"].abs())
    rows = []
    for exp, grp in prof.groupby("expiry", sort=True):
        by_strike = grp.groupby("strike")["_gross"].sum().sort_index()
        by_strike = by_strike[by_strike > 0]
        if by_strike.empty:
            continue
        total = float(by_strike.sum())
        shares = by_strike / total
        top_strike = float(by_strike.idxmax())
        top3 = by_strike.sort_values(ascending=False).head(3)
        rows.append(
            {
                "expiry": exp,
                "underlying_price": float(grp["underlying_price"].iloc[0]),
                "n_strikes": int(len(by_strike)),
                "gross_gex": total,
                "top_strike": top_strike,
                "top_share": float(by_strike.loc[top_strike] / total),
                "top3_share": float(top3.sum() / total),
                "hhi": float((shares**2).sum()),
            }
        )

    return pd.DataFrame(rows, columns=cols)


def _scan_gamma_net(grp, rate, div_yield, contract_size, spot_range, n_points):
    """Scan net and gross dealer GEX across a spot grid for one expiry group.

    Shared by ``gamma_flip`` and ``opticore.plot.gamma_profile`` so the curve a
    plot draws and the flip level a table reports always come from the same scan.
    ``grp`` must already carry the ``_kind`` column and be filtered to usable
    rows. Returns ``(grid, net, gross, net_spot, flip, regime)``: the spot grid,
    net and gross GEX along it, net GEX interpolated at the current spot, the
    flip level nearest spot (NaN when nothing crosses inside the window) and the
    regime at spot (``positive``/``negative``/``flat``).
    """
    from opticore._core import _greeks_batch

    def _rw(a, dt):
        return np.require(a, dtype=dt, requirements=["C", "A", "W"])

    spot = float(grp["underlying_price"].iloc[0])
    strikes = _rw(grp["strike"].to_numpy(), np.float64)
    ttes = _rw(grp["tte"].to_numpy(), np.float64)
    ivs = _rw(grp["iv"].to_numpy(), np.float64)
    oi = grp["open_interest"].to_numpy(dtype=np.float64)
    is_call = _rw(grp["_kind"].to_numpy() == "call", bool)
    sign = np.where(is_call, 1.0, -1.0)
    weight = sign * oi

    grid = np.linspace(spot * (1.0 - spot_range), spot * (1.0 + spot_range), n_points)
    n_opts = len(strikes)

    # single vectorized call instead of n_points separate calls - avoids
    # repeated alloc/free of nanobind-owned arrays in manylinux environments
    s_all = _rw(np.repeat(grid, n_opts), np.float64)
    k_all = _rw(np.tile(strikes, n_points), np.float64)
    t_all = _rw(np.tile(ttes, n_points), np.float64)
    v_all = _rw(np.tile(ivs, n_points), np.float64)
    ic_all = _rw(np.tile(is_call, n_points), bool)
    _, _, gamma_all, _, _, _ = _greeks_batch(
        s_all, k_all, t_all, float(rate), v_all, float(div_yield), ic_all
    )
    gamma_mat = np.nan_to_num(np.asarray(gamma_all), nan=0.0).reshape(n_points, n_opts)
    scales = contract_size * grid * grid * 0.01
    net = np.sum(weight * gamma_mat, axis=1) * scales
    gross = np.sum(np.abs(weight) * gamma_mat, axis=1) * scales

    # a near-symmetric book nets to floating-point noise, not a real
    # exposure, so judge sign relative to the gross gamma on the book
    tol = 1e-6 * float(np.max(gross)) if gross.size else 0.0

    net_spot = float(np.interp(spot, grid, net))
    if net_spot > tol:
        regime = "positive"
    elif net_spot < -tol:
        regime = "negative"
    else:
        regime = "flat"

    flip = float("nan")
    crossings = np.nonzero(np.diff(np.sign(net)) != 0)[0]
    if crossings.size and np.max(np.abs(net)) > tol:
        # pick the crossing whose interpolated spot is closest to current spot
        best = float("nan")
        for j in crossings:
            a, b = net[j], net[j + 1]
            if a == b:
                continue
            x = grid[j] + (grid[j + 1] - grid[j]) * (-a) / (b - a)
            if np.isnan(best) or abs(x - spot) < abs(best - spot):
                best = x
        flip = best

    return grid, net, gross, net_spot, flip, regime


def gamma_flip(
    chain: pd.DataFrame,
    rate: float = 0.045,
    div_yield: float = 0.0,
    contract_size: float = 100.0,
    spot_range: float = 0.2,
    n_points: int = 81,
) -> pd.DataFrame:
    """Per-expiry gamma flip level: the spot where net dealer GEX crosses zero.

    Net dealer gamma (long calls, short puts) depends on spot because each
    option's gamma peaks near its own strike. Far below the book the short-put
    leg tends to dominate (net short gamma, price-amplifying); far above it the
    long-call leg dominates (net long gamma, price-dampening). The ``flip_spot``
    is the level in between where net GEX changes sign - the "zero gamma" level
    dealers' hedging flips around.

    Gamma is recomputed at a grid of hypothetical spots spanning
    ``spot * (1 +/- spot_range)`` using each option's recovered ``iv`` and time
    to expiry, so the answer reflects the whole strike ladder, not just gamma at
    the current spot. The crossing nearest the current spot is reported, linearly
    interpolated between the two bracketing grid points. ``regime`` is the sign of
    net GEX at the current spot: ``positive`` (dampening), ``negative``
    (amplifying), or ``flat`` when the book nets to zero everywhere (a perfectly
    symmetric call/put book cancels at every spot and has no flip).

    Needs ``iv`` and ``tte`` from ``enrich`` plus ``open_interest``, ``strike``,
    ``kind`` and ``underlying_price``. Rows with NaN iv contribute nothing; an
    expiry with no usable gamma is skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        An enriched chain (see ``enrich``); needs ``iv``, ``tte`` and
        ``open_interest``.
    rate, div_yield : float
        BSM inputs for the gamma re-pricing (match what you passed to ``enrich``).
    contract_size : float
        Shares per contract (default: 100). Scales GEX but not ``flip_spot``.
    spot_range : float
        Half-width of the spot grid as a fraction of spot (default: 0.2 -> +/-20%).
    n_points : int
        Number of grid spots to scan (default: 81).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, net_gex, flip_spot,
        flip_distance_pct, regime. One row per expiry, sorted by expiry.
        ``flip_spot`` and ``flip_distance_pct`` are NaN when no crossing falls in
        the scanned range.
    """
    cols = [
        "expiry",
        "underlying_price",
        "net_gex",
        "flip_spot",
        "flip_distance_pct",
        "regime",
    ]
    needed = {"kind", "iv", "tte", "open_interest", "strike", "underlying_price"}
    if chain.empty or not needed.issubset(chain.columns) or n_points < 2:
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "iv", "tte", "open_interest", "strike"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        _, _, _, net_spot, flip, regime = _scan_gamma_net(
            grp, rate, div_yield, contract_size, spot_range, n_points
        )
        spot = float(grp["underlying_price"].iloc[0])
        dist = (flip / spot - 1.0) * 100.0 if not np.isnan(flip) else float("nan")
        rows.append(
            {
                "expiry": exp,
                "underlying_price": spot,
                "net_gex": net_spot,
                "flip_spot": flip,
                "flip_distance_pct": dist,
                "regime": regime,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def vega_exposure(chain: pd.DataFrame, contract_size: float = 100.0) -> pd.DataFrame:
    """Per-expiry dealer vega exposure (VEX) from open interest and Greeks.

    The volatility sibling of ``delta_exposure`` and ``gamma_exposure``. Under the
    same dealer convention - long call, short put - per option

        vex_i = sign * vega_i * open_interest_i * contract_size

    with ``sign = +1`` for calls and ``-1`` for puts. ``vega`` from ``enrich`` is
    already per 1% vol move, so VEX is dollars of P&L the writing side gains or
    loses per one volatility point. Positive net VEX means dealers carry long vega
    and benefit when implied vol rises (and tend to sell vol into spikes);
    negative means they are short vega and get squeezed when vol pops.
    ``vega_wall_strike`` is the strike holding the most gross dollar vega (both
    sides, sign ignored) - usually near the money, where vega is largest and vol
    hedging concentrates.

    Needs the ``vega`` column from ``enrich`` plus ``open_interest``. Rows with
    NaN vega or no open interest contribute nothing; an expiry with no usable vega
    is skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        An enriched chain (see ``enrich``); needs ``vega`` and ``open_interest``.
    contract_size : float
        Shares per contract (default: 100).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, call_vex, put_vex, net_vex,
        vega_wall_strike. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "call_vex",
        "put_vex",
        "net_vex",
        "vega_wall_strike",
    ]
    needed = {"kind", "vega", "open_interest", "strike", "underlying_price"}
    if chain.empty or not needed.issubset(chain.columns):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "vega", "open_interest", "strike"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        spot = float(grp["underlying_price"].iloc[0])
        sign = np.where(grp["_kind"].to_numpy() == "call", 1.0, -1.0)
        vex = sign * grp["vega"].to_numpy() * grp["open_interest"].to_numpy() * contract_size
        gross = np.abs(grp["vega"].to_numpy()) * grp["open_interest"].to_numpy() * contract_size
        grp = grp.assign(_vex=vex, _gross=gross)
        call_vex = float(grp.loc[grp["_kind"] == "call", "_vex"].sum())
        put_vex = float(grp.loc[grp["_kind"] == "put", "_vex"].sum())

        by_strike = grp.groupby("strike")["_gross"].sum().sort_index()
        if (by_strike > 0).any():
            wall = float(by_strike.idxmax())
        else:
            continue

        rows.append(
            {
                "expiry": exp,
                "underlying_price": spot,
                "call_vex": call_vex,
                "put_vex": put_vex,
                "net_vex": call_vex + put_vex,
                "vega_wall_strike": wall,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def vega_exposure_by_strike(chain: pd.DataFrame, contract_size: float = 100.0) -> pd.DataFrame:
    """Per-strike dealer vega exposure, the profile behind ``vega_exposure``.

    ``vega_exposure`` nets VEX to one row per expiry; this keeps the strike axis
    the way ``gamma_exposure_by_strike`` does for GEX. For each expiry and strike
    it returns ``call_vex``, ``put_vex``, ``net_vex`` and a ``cumulative_net_vex``
    running up the strikes, plus ``is_vega_wall`` on the strike carrying the most
    gross dollar vega. Charted against strike it's the VEX profile, and the sign
    change in ``cumulative_net_vex`` brackets the strike where net dealer vega
    flips.

    Same scaling and NaN rules as ``vega_exposure``: per option
    ``sign * vega * open_interest * contract_size`` with ``sign`` +1 for calls and
    -1 for puts, and ``vega`` already per 1% vol move (no spot scaling, unlike
    GEX). Rows with NaN vega or no open interest contribute nothing; an expiry
    with no usable vega is skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        An enriched chain (see ``enrich``); needs ``vega`` and ``open_interest``.
    contract_size : float
        Shares per contract (default: 100).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, strike, call_vex, put_vex, net_vex,
        cumulative_net_vex, is_vega_wall. One row per (expiry, strike), sorted by
        expiry then strike.
    """
    cols = [
        "expiry",
        "underlying_price",
        "strike",
        "call_vex",
        "put_vex",
        "net_vex",
        "cumulative_net_vex",
        "is_vega_wall",
    ]
    needed = {"kind", "vega", "open_interest", "strike", "underlying_price"}
    if chain.empty or not needed.issubset(chain.columns):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "vega", "open_interest", "strike"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        spot = float(grp["underlying_price"].iloc[0])
        sign = np.where(grp["_kind"].to_numpy() == "call", 1.0, -1.0)
        grp = grp.assign(
            _vex=sign * grp["vega"].to_numpy() * grp["open_interest"].to_numpy() * contract_size,
            _gross=np.abs(grp["vega"].to_numpy()) * grp["open_interest"].to_numpy() * contract_size,
        )
        gross_by_strike = grp.groupby("strike")["_gross"].sum()
        if not (gross_by_strike > 0).any():
            continue
        wall = float(gross_by_strike.idxmax())

        call_vex = grp.loc[grp["_kind"] == "call"].groupby("strike")["_vex"].sum()
        put_vex = grp.loc[grp["_kind"] == "put"].groupby("strike")["_vex"].sum()

        cum = 0.0
        for s in np.sort(grp["strike"].unique()):
            c = float(call_vex.get(s, 0.0))
            p = float(put_vex.get(s, 0.0))
            net = c + p
            cum += net
            rows.append(
                {
                    "expiry": exp,
                    "underlying_price": spot,
                    "strike": float(s),
                    "call_vex": c,
                    "put_vex": p,
                    "net_vex": net,
                    "cumulative_net_vex": cum,
                    "is_vega_wall": float(s) == wall,
                }
            )

    return pd.DataFrame(rows, columns=cols)


def theta_exposure(chain: pd.DataFrame, contract_size: float = 100.0) -> pd.DataFrame:
    """Per-expiry dealer theta exposure (TEX) from open interest and Greeks.

    The time-decay sibling of ``vega_exposure``. Under the same dealer convention -
    long call, short put - per option

        tex_i = sign * theta_i * open_interest_i * contract_size

    with ``sign = +1`` for calls and ``-1`` for puts. ``theta`` from ``enrich`` is
    the per-day decay, so TEX is dollars the writing side earns or pays each day the
    spot sits still. Long options carry negative theta, so the long-call leg shows up
    negative and the short-put leg positive. ``theta_wall_strike`` is the strike
    holding the most gross dollar theta (both sides, sign ignored) - near the money,
    where theta is largest.

    Needs the ``theta`` column from ``enrich`` plus ``open_interest``. Rows with NaN
    theta or no open interest contribute nothing; an expiry with no usable theta is
    skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        An enriched chain (see ``enrich``); needs ``theta`` and ``open_interest``.
    contract_size : float
        Shares per contract (default: 100).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, call_tex, put_tex, net_tex,
        theta_wall_strike. One row per expiry, sorted by expiry.
    """
    cols = [
        "expiry",
        "underlying_price",
        "call_tex",
        "put_tex",
        "net_tex",
        "theta_wall_strike",
    ]
    needed = {"kind", "theta", "open_interest", "strike", "underlying_price"}
    if chain.empty or not needed.issubset(chain.columns):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "theta", "open_interest", "strike"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        spot = float(grp["underlying_price"].iloc[0])
        sign = np.where(grp["_kind"].to_numpy() == "call", 1.0, -1.0)
        tex = sign * grp["theta"].to_numpy() * grp["open_interest"].to_numpy() * contract_size
        gross = np.abs(grp["theta"].to_numpy()) * grp["open_interest"].to_numpy() * contract_size
        grp = grp.assign(_tex=tex, _gross=gross)
        call_tex = float(grp.loc[grp["_kind"] == "call", "_tex"].sum())
        put_tex = float(grp.loc[grp["_kind"] == "put", "_tex"].sum())

        by_strike = grp.groupby("strike")["_gross"].sum().sort_index()
        if (by_strike > 0).any():
            wall = float(by_strike.idxmax())
        else:
            continue

        rows.append(
            {
                "expiry": exp,
                "underlying_price": spot,
                "call_tex": call_tex,
                "put_tex": put_tex,
                "net_tex": call_tex + put_tex,
                "theta_wall_strike": wall,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def theta_exposure_by_strike(chain: pd.DataFrame, contract_size: float = 100.0) -> pd.DataFrame:
    """Per-strike dealer theta exposure, the profile behind ``theta_exposure``.

    ``theta_exposure`` nets TEX to one row per expiry; this keeps the strike axis
    like ``vega_exposure_by_strike`` does for VEX. For each expiry and strike it
    returns ``call_tex``, ``put_tex``, ``net_tex`` and a ``cumulative_net_tex``
    running up the strikes, plus ``is_theta_wall`` on the strike carrying the most
    gross dollar theta. Charted against strike it's the TEX profile, peaking near
    the money where theta is largest, and the sign change in ``cumulative_net_tex``
    brackets the strike where net dealer theta flips.

    Same scaling and NaN rules as ``theta_exposure``: per option
    ``sign * theta * open_interest * contract_size`` with ``sign`` +1 for calls and
    -1 for puts, and ``theta`` already the per-day decay (no spot scaling, unlike
    GEX). Rows with NaN theta or no open interest contribute nothing; an expiry
    with no usable theta is skipped.

    Parameters
    ----------
    chain : pd.DataFrame
        An enriched chain (see ``enrich``); needs ``theta`` and ``open_interest``.
    contract_size : float
        Shares per contract (default: 100).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, underlying_price, strike, call_tex, put_tex, net_tex,
        cumulative_net_tex, is_theta_wall. One row per (expiry, strike), sorted by
        expiry then strike.
    """
    cols = [
        "expiry",
        "underlying_price",
        "strike",
        "call_tex",
        "put_tex",
        "net_tex",
        "cumulative_net_tex",
        "is_theta_wall",
    ]
    needed = {"kind", "theta", "open_interest", "strike", "underlying_price"}
    if chain.empty or not needed.issubset(chain.columns):
        return pd.DataFrame(columns=cols)

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "theta", "open_interest", "strike"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for exp, grp in df.groupby("expiry", sort=True):
        spot = float(grp["underlying_price"].iloc[0])
        sign = np.where(grp["_kind"].to_numpy() == "call", 1.0, -1.0)
        grp = grp.assign(
            _tex=sign * grp["theta"].to_numpy() * grp["open_interest"].to_numpy() * contract_size,
            _gross=np.abs(grp["theta"].to_numpy())
            * grp["open_interest"].to_numpy()
            * contract_size,
        )
        gross_by_strike = grp.groupby("strike")["_gross"].sum()
        if not (gross_by_strike > 0).any():
            continue
        wall = float(gross_by_strike.idxmax())

        call_tex = grp.loc[grp["_kind"] == "call"].groupby("strike")["_tex"].sum()
        put_tex = grp.loc[grp["_kind"] == "put"].groupby("strike")["_tex"].sum()

        cum = 0.0
        for s in np.sort(grp["strike"].unique()):
            c = float(call_tex.get(s, 0.0))
            p = float(put_tex.get(s, 0.0))
            net = c + p
            cum += net
            rows.append(
                {
                    "expiry": exp,
                    "underlying_price": spot,
                    "strike": float(s),
                    "call_tex": c,
                    "put_tex": p,
                    "net_tex": net,
                    "cumulative_net_tex": cum,
                    "is_theta_wall": float(s) == wall,
                }
            )

    return pd.DataFrame(rows, columns=cols)
