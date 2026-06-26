"""Type stubs for opticore.chain.

Mirrors the runtime signatures in chain.py with explicit return types so
mypy / IDE users see real types instead of `Any`. The stubs in
``__init__.pyi`` re-export these — but having a dedicated stub for
``opticore.chain`` means `from opticore.chain import enrich` is also typed.
"""

from __future__ import annotations

from typing import Any, NamedTuple, TypedDict

import pandas as pd

class ConnectionStatus(TypedDict):
    """Return shape of ``check_connection``.

    Using a TypedDict (rather than ``dict``) lets mypy/IDEs autocomplete
    the keys and flag typos like ``status["accuont"]``.
    """

    connected: bool
    account: str | None
    server_version: int | None
    message: str

def check_connection(
    host: str = ...,
    port: int = ...,
    client_id: int = ...,
    timeout: float = ...,
) -> ConnectionStatus: ...
def fetch_chain(
    symbol: str,
    provider: str = ...,
    max_expiries: int = ...,
    strike_count: int = ...,
    timeout: float = ...,
    **provider_kwargs: Any,
) -> pd.DataFrame: ...
def enrich(
    chain: pd.DataFrame,
    rate: float = ...,
    div_yield: float = ...,
    price_col: str = ...,
    include_theo: bool = ...,
) -> pd.DataFrame: ...
def parity_check(
    chain: pd.DataFrame,
    rate: float = ...,
    div_yield: float = ...,
    price_col: str = ...,
) -> pd.DataFrame: ...
def implied_forward(
    chain: pd.DataFrame,
    rate: float = ...,
    n_atm_strikes: int = ...,
    price_col: str = ...,
) -> pd.DataFrame: ...
def atm_iv(
    chain: pd.DataFrame,
    rate: float = ...,
    div_yield: float = ...,
    price_col: str = ...,
) -> pd.DataFrame: ...

class TermSlope(NamedTuple):
    slope: float
    shape: str
    front_iv: float
    back_iv: float
    front_tte: float
    back_tte: float

def term_slope(atm: pd.DataFrame, flat_tol: float = ...) -> TermSlope: ...
def iv_skew(
    chain: pd.DataFrame,
    rate: float = ...,
    div_yield: float = ...,
    price_col: str = ...,
) -> pd.DataFrame: ...
def rr_bf(
    chain: pd.DataFrame,
    rate: float = ...,
    div_yield: float = ...,
    price_col: str = ...,
) -> pd.DataFrame: ...
def straddle(
    chain: pd.DataFrame,
    price_col: str = ...,
) -> pd.DataFrame: ...
def max_pain(
    chain: pd.DataFrame,
) -> pd.DataFrame: ...
def pcr(
    chain: pd.DataFrame,
) -> pd.DataFrame: ...
def turnover(
    chain: pd.DataFrame,
) -> pd.DataFrame: ...
def liquidity(
    chain: pd.DataFrame,
) -> pd.DataFrame: ...
def dollar_volume(
    chain: pd.DataFrame,
    price_col: str = ...,
    contract_size: float = ...,
) -> pd.DataFrame: ...
def oi_walls(
    chain: pd.DataFrame,
) -> pd.DataFrame: ...
def oi_profile(
    chain: pd.DataFrame,
) -> pd.DataFrame: ...
def volume_profile(
    chain: pd.DataFrame,
) -> pd.DataFrame: ...
def volume_walls(
    chain: pd.DataFrame,
) -> pd.DataFrame: ...
