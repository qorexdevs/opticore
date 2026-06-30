"""Render the chart images shown in the README.

Run after changing the plotting code or the sample chain:
    python scripts/render_readme_charts.py

Everything uses the bundled synthetic SPY chain (provider="sample"), so this
runs with no account and no network. Output goes to docs/assets/ and is
committed so the README renders on GitHub and PyPI without a build step.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import opticore as oc

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chain = oc.fetch_chain(provider="sample", symbol="SPY")
    enriched = oc.enrich(chain, rate=0.045, div_yield=0.013)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    oc.plot.exposure_profile(enriched, greek="gamma", ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "dealer-gamma-exposure.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    oc.plot.smile(enriched, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "vol-smile.png", dpi=140)
    plt.close(fig)

    print(f"wrote {OUT / 'dealer-gamma-exposure.png'}")
    print(f"wrote {OUT / 'vol-smile.png'}")


if __name__ == "__main__":
    main()
