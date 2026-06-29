"""Visualization functions for options analytics."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from opticore import Leg
from opticore import greeks as oc_greeks


def _get_plt():
    """Lazy import matplotlib."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install opticore[viz]"
        )


def smile(
    enriched_df,
    expiry: Optional[str] = None,
    x: str = "strike",
    ax=None,
):
    """Plot implied volatility smile from an enriched chain DataFrame.

    Parameters
    ----------
    enriched_df : pd.DataFrame
        DataFrame with 'strike', 'expiry', 'iv', 'kind' columns
        (output of oc.enrich()).
    expiry : str, pd.Timestamp, or None
        Specific expiry date to plot (e.g. '2026-06-20' or a Timestamp).
        If None, plots all.
    x : str
        X-axis variable: 'strike' or 'moneyness'.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Standard matplotlib convention. Use ``ax`` to add annotations,
        ``fig`` to save / display.
    """
    plt = _get_plt()
    import pandas as pd

    df = enriched_df.copy()

    # Filter valid IV
    df = df[df["iv"].notna() & (df["iv"] > 0) & (df["iv"] < 5.0)]

    # Use calls only for cleaner smile
    df = df[df["kind"].str.lower().isin(["call", "c"])]

    if expiry is not None:
        # Normalize both sides to UTC-midnight Timestamps so str/Timestamp
        # inputs match either schema (legacy "YYYYMMDD" strings or Timestamps).
        target = pd.to_datetime(expiry, utc=True).normalize()
        exp_norm = pd.to_datetime(df["expiry"], utc=True).dt.normalize()
        df = df[exp_norm == target]

    if df.empty:
        raise ValueError("No data to plot after filtering.")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    x_col = x if x in df.columns else "strike"

    # Group by expiry and plot each
    for exp, group in df.groupby("expiry"):
        group = group.sort_values(x_col)
        label = pd.Timestamp(exp).strftime("%Y-%m-%d") if hasattr(exp, "strftime") else str(exp)
        ax.plot(group[x_col], group["iv"] * 100, "o-", markersize=4, label=label)

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title("IV Smile")
    ax.legend(title="Expiry", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def payoff(
    legs: Sequence[Leg],
    spot_range: Optional[tuple[float, float]] = None,
    num_points: int = 200,
    ax=None,
):
    """Plot strategy payoff diagram.

    Parameters
    ----------
    legs : list of Leg
        Strategy legs. Each Leg(kind, strike, qty, premium).
    spot_range : tuple or None
        (low, high) for the x-axis. Auto-computed if None.
    num_points : int
        Number of points to plot.
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Standard matplotlib convention.

    Examples
    --------
    >>> import opticore as oc
    >>> fig, ax = oc.plot.payoff([
    ...     oc.Leg("call", strike=105, qty=1, premium=3.50),
    ...     oc.Leg("put",  strike=95,  qty=1, premium=2.10),
    ... ])
    """
    plt = _get_plt()

    if not legs:
        raise ValueError("At least one leg is required.")

    # Determine spot range
    strikes = [leg.strike for leg in legs]
    if spot_range is None:
        mid = np.mean(strikes)
        span = max(np.ptp(strikes) * 1.5, mid * 0.2)
        spot_range = (mid - span, mid + span)

    spots = np.linspace(spot_range[0], spot_range[1], num_points)

    # Compute payoff at expiry for each leg
    total_payoff = np.zeros_like(spots)
    total_cost = 0.0

    for leg in legs:
        if leg.kind.lower() in ("call", "c"):
            intrinsic = np.maximum(spots - leg.strike, 0)
        else:
            intrinsic = np.maximum(leg.strike - spots, 0)

        total_payoff += leg.qty * intrinsic
        total_cost += leg.qty * leg.premium

    net_pnl = total_payoff - total_cost

    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    ax.plot(spots, net_pnl, "b-", linewidth=2, label="P&L at Expiry")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    # Mark break-even points
    sign_changes = np.where(np.diff(np.sign(net_pnl)))[0]
    for idx in sign_changes:
        # Linear interpolation for exact break-even
        x0, x1 = spots[idx], spots[idx + 1]
        y0, y1 = net_pnl[idx], net_pnl[idx + 1]
        be = x0 - y0 * (x1 - x0) / (y1 - y0)
        ax.axvline(x=be, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.annotate(f"BE: {be:.1f}", xy=(be, 0), fontsize=9, ha="center", va="bottom", color="red")

    # Mark strikes
    for leg in legs:
        ax.axvline(x=leg.strike, color="gray", linestyle=":", alpha=0.4)

    # Fill profit/loss regions
    ax.fill_between(spots, net_pnl, 0, where=(net_pnl > 0), alpha=0.1, color="green")
    ax.fill_between(spots, net_pnl, 0, where=(net_pnl < 0), alpha=0.1, color="red")

    ax.set_xlabel("Underlying Price at Expiry")
    ax.set_ylabel("Profit / Loss")
    ax.set_title("Strategy Payoff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def greek(
    greek_name: str,
    spot_range: tuple[float, float],
    strike: float,
    expiry: float,
    rate: float,
    vol: float,
    kind: str = "both",
    div_yield: float = 0.0,
    num_points: int = 200,
    ax=None,
):
    """Plot a Greek as a function of spot price.

    Parameters
    ----------
    greek_name : str
        One of: 'delta', 'gamma', 'theta', 'vega', 'rho', 'price'.
    spot_range : tuple
        (low, high) range for the underlying price.
    strike, expiry, rate, vol, div_yield : float
        BSM parameters.
    kind : str
        'call', 'put', or 'both' (overlays both on same axes).
    num_points : int
        Number of points to compute.

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Standard matplotlib convention.
    """
    plt = _get_plt()

    spots = np.linspace(spot_range[0], spot_range[1], num_points)
    valid_greeks = {"price", "delta", "gamma", "theta", "vega", "rho"}
    if greek_name not in valid_greeks:
        raise ValueError(f"greek must be one of {valid_greeks}, got: {greek_name!r}")

    kinds = []
    if kind.lower() in ("call", "c", "both"):
        kinds.append(("call", True))
    if kind.lower() in ("put", "p", "both"):
        kinds.append(("put", False))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    for label, is_call in kinds:
        values = []
        for s in spots:
            g = oc_greeks(s, strike, expiry, rate, vol, label, div_yield)
            values.append(getattr(g, greek_name))

        ax.plot(spots, values, linewidth=2, label=label.capitalize())

    ax.axvline(x=strike, color="gray", linestyle=":", alpha=0.5, label=f"Strike ({strike})")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel(greek_name.capitalize())
    ax.set_title(f"{greek_name.capitalize()} vs Spot Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def liquidity(
    chain,
    relative: bool = True,
    ax=None,
):
    """Plot per-expiry bid-ask spread from a chain DataFrame.

    Bars show the median spread for each expiry (the typical tax on a fill),
    with a marker for that expiry's widest relative spread so a single
    untradeable strike hiding behind a decent median still shows up. Shorter
    bars are the liquid expiries worth working an order in.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``oc.liquidity``; needs ``bid`` and ``ask``.
    relative : bool
        Plot the spread relative to mid (percent of price) when True, the
        default, so cheap and dear options compare fairly. When False, plot the
        absolute spread in price terms.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Standard matplotlib convention. Use ``ax`` to add annotations,
        ``fig`` to save / display.
    """
    plt = _get_plt()
    import pandas as pd

    from opticore.chain import liquidity as compute_liquidity

    liq = compute_liquidity(chain)
    if liq.empty:
        raise ValueError("No liquidity data to plot after filtering.")

    labels = [
        pd.Timestamp(e).strftime("%Y-%m-%d") if hasattr(e, "strftime") else str(e)
        for e in liq["expiry"]
    ]
    if relative:
        median = liq["median_rel_spread"] * 100
        widest = liq["max_rel_spread"] * 100
        ylabel = "Bid-Ask Spread (% of mid)"
    else:
        median = liq["median_spread"]
        widest = None
        ylabel = "Bid-Ask Spread (price)"

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    pos = np.arange(len(labels))
    ax.bar(pos, median, color="tab:blue", alpha=0.8, label="median")
    if widest is not None:
        ax.scatter(pos, widest, color="tab:red", marker="_", s=200, zorder=3, label="widest")

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title("Liquidity by Expiry")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig, ax


def term_structure(
    chain,
    rate: float = 0.045,
    div_yield: float = 0.0,
    price_col: str = "mid",
    fit: bool = True,
    ax=None,
):
    """Plot the ATM implied-vol term structure from a chain DataFrame.

    One point per expiry, ATM IV against time to expiry, so you can read
    contango (curve rising with tenor) or backwardation (front bid up) at a
    glance. With ``fit`` the least-squares line from ``oc.term_slope`` is drawn
    over the points and its shape is labelled.

    Parameters
    ----------
    chain : pd.DataFrame
        Same schema as ``oc.atm_iv``; needs the columns ``enrich`` reads.
    rate : float
        Risk-free rate passed through to ``atm_iv`` (default: 0.045).
    div_yield : float
        Continuous dividend yield passed through to ``atm_iv`` (default: 0.0).
    price_col : str
        Which price ``atm_iv`` should use (default: 'mid').
    fit : bool
        Overlay the fitted term-structure line and shape label when there are
        at least two expiries. Default True.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Standard matplotlib convention. Use ``ax`` to add annotations,
        ``fig`` to save / display.
    """
    plt = _get_plt()

    from opticore.chain import atm_iv as compute_atm_iv
    from opticore.chain import term_slope

    atm = compute_atm_iv(chain, rate=rate, div_yield=div_yield, price_col=price_col)
    if atm.empty:
        raise ValueError("No term-structure data to plot after filtering.")

    atm = atm.sort_values("tte")
    tte = atm["tte"].to_numpy()
    iv = atm["atm_iv"].to_numpy() * 100

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    ax.plot(tte, iv, "o-", color="tab:blue", markersize=5, label="ATM IV")

    if fit and len(atm) >= 2:
        ts = term_slope(atm)
        line = (ts.front_iv + ts.slope * (tte - ts.front_tte)) * 100
        ax.plot(tte, line, "--", color="tab:red", alpha=0.7, label=f"fit ({ts.shape})")

    ax.set_xlabel("Time to Expiry (years)")
    ax.set_ylabel("ATM Implied Volatility (%)")
    ax.set_title("IV Term Structure")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def gamma_profile(
    chain,
    expiry=None,
    rate: float = 0.045,
    div_yield: float = 0.0,
    contract_size: float = 100.0,
    spot_range: float = 0.2,
    n_points: int = 81,
    ax=None,
):
    """Plot the net dealer gamma profile and flip level for one expiry.

    Sweeps net GEX across the same spot grid ``oc.gamma_flip`` scans, draws the
    curve, marks the current spot and the flip level, and shades the
    price-dampening (net long gamma, positive) and price-amplifying (net short
    gamma, negative) regions so the shape around spot is readable at a glance.
    Defaults to the nearest expiry; pass ``expiry`` to pick another, like
    ``oc.plot.smile``.

    Parameters
    ----------
    chain : pd.DataFrame
        An enriched chain (see ``oc.enrich``); needs ``iv``, ``tte``,
        ``open_interest``, ``strike``, ``kind`` and ``underlying_price``.
    expiry : str, pd.Timestamp, or None
        Expiry to plot. If None, the nearest (soonest) expiry is used.
    rate, div_yield, contract_size, spot_range, n_points :
        Passed straight through to the gamma scan; match what you gave
        ``gamma_flip`` / ``enrich`` so the curve and the flip table agree.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    (fig, ax) : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Standard matplotlib convention.
    """
    plt = _get_plt()
    import pandas as pd

    from opticore.chain import _scan_gamma_net

    needed = {"kind", "iv", "tte", "open_interest", "strike", "underlying_price"}
    missing = needed - set(chain.columns)
    if missing:
        raise ValueError(f"chain is missing columns for gamma profile: {sorted(missing)}")

    df = chain.copy()
    df["_kind"] = (
        df["kind"].str.lower().map({"call": "call", "c": "call", "put": "put", "p": "put"})
    )
    df = df.dropna(subset=["_kind", "iv", "tte", "open_interest", "strike"])
    if df.empty:
        raise ValueError("No usable gamma data to plot after filtering.")

    if expiry is not None:
        target = pd.to_datetime(expiry, utc=True).normalize()
        exp_norm = pd.to_datetime(df["expiry"], utc=True).dt.normalize()
        grp = df[exp_norm == target]
        if grp.empty:
            raise ValueError(f"No rows for expiry {expiry!r}.")
        label = pd.Timestamp(target).strftime("%Y-%m-%d")
    else:
        # nearest = soonest expiry = smallest time to expiry
        nearest = df.loc[df["tte"].idxmin(), "expiry"]
        grp = df[df["expiry"] == nearest]
        label = (
            pd.Timestamp(nearest).strftime("%Y-%m-%d")
            if hasattr(nearest, "strftime")
            else str(nearest)
        )

    grid, net, _gross, _net_spot, flip, regime = _scan_gamma_net(
        grp, rate, div_yield, contract_size, spot_range, n_points
    )
    spot = float(grp["underlying_price"].iloc[0])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    ax.axhline(0, color="gray", linewidth=1, alpha=0.6)
    ax.plot(grid, net, color="tab:blue", linewidth=2, label="Net GEX")
    ax.fill_between(
        grid, net, 0, where=net >= 0, color="tab:green", alpha=0.2, label="dampening (net long)"
    )
    ax.fill_between(
        grid, net, 0, where=net < 0, color="tab:red", alpha=0.2, label="amplifying (net short)"
    )
    ax.axvline(spot, color="black", linestyle="--", alpha=0.7, label=f"Spot ({spot:.2f})")
    if not np.isnan(flip):
        ax.axvline(flip, color="tab:purple", linestyle=":", linewidth=2, label=f"Flip ({flip:.2f})")

    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Net Dealer GEX")
    ax.set_title(f"Gamma Profile {label} (regime: {regime})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax
