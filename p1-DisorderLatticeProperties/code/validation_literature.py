"""Published orthotropic-lattice C(T) geometry-calibration comparison.

Omidi and St-Pierre (2023), IJSS 270, 112233, Appendix C, Tables C.1/C.2.
This checks the reported continuum calibration attached to their snub-square
lattice experiments. It does not reproduce discrete-beam fracture toughness
or establish a plastic-work coefficient.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from resources.calculations import calc_FaW, calc_FaW_aniso


DOI = "10.1016/j.ijsolstr.2023.112233"
SOURCE_URL = (
    "https://acris.aalto.fi/ws/portalfiles/portal/106210409/"
    "1_s2.0_S0020768323001300_main.pdf"
)
# Table C.2; rho is the specimen relative density, chi is the paper's
# orthotropy parameter denoted rho. Table C.1 / Eq. D.2 give lambda = 1.
PUBLISHED = ((0.20, 7.65, 1.15), (0.25, 4.74, 1.10), (0.30, 3.16, 1.06))
A_OVER_W = 49.0 / 196.0  # Snub-square C(T), Figure 3b, not an assumed 0.5.


def published_ct_comparison() -> list[dict[str, float]]:
    """Compare Ortega's factor with Table C.2 at the source's crack ratio.

    Use the published chi directly, without silently applying another
    plane-strain mapping. S is a normalized compliance giving precisely
    lambda=1 and the reported chi; its absolute modulus cancels from K's
    dimensionless geometry factor. This is a source-parameter comparison,
    not a reconstruction of an unstated out-of-plane material completion.
    """
    astm = float(calc_FaW(A_OVER_W, 1.0))
    rows = []
    for density, chi, published_y in PUBLISHED:
        compliance = np.array(
            [[1.0, -0.433, 0.0], [-0.433, 1.0, 0.0],
             [0.0, 0.0, 2.0 * (chi + 0.433)]]
        )
        ortega = float(calc_FaW_aniso(A_OVER_W, 1.0, S=compliance))
        ortega_y = ortega / astm
        rows.append({
            "relative_density": density,
            "lambda": 1.0,
            "chi": chi,
            "a_over_W": A_OVER_W,
            "published_Y": published_y,
            "ortega_Y": ortega_y,
            "difference_percent": 100.0 * (ortega_y / published_y - 1.0),
        })
    return rows


def plot_published_ct(output_dir: Path) -> tuple[Path, Path]:
    """Write one PNG and one SVG, with the published points unchanged."""
    rows = published_ct_comparison()
    x = np.array([r["relative_density"] for r in rows])
    published = np.array([r["published_Y"] for r in rows])
    ortega = np.array([r["ortega_Y"] for r in rows])
    error = np.array([r["difference_percent"] for r in rows])
    with plt.rc_context({"font.size": 11, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1),
                                 gridspec_kw={"width_ratios": [1.55, 1]})
        ax, residual = axes
        ax.plot(x, published, "^", color="black", mfc="none", mew=1.5,
                ms=10, label="Omidi & St-Pierre (2023)")
        ax.plot(x, ortega, "o", color="#c62828", ms=5.5,
                label="Current study (Ortega)")
        for density, chi, _ in PUBLISHED:
            ax.annotate(rf"$\chi={chi:.2f}$", (density, max(
                r["ortega_Y"] for r in rows if r["relative_density"] == density)),
                xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
        ax.set_ylabel(r"C(T) correction, $Y=f_{\mathrm{CT}}/f_{\mathrm{ASTM}}$")
        ax.set_ylim(1.02, 1.28)
        ax.legend(loc="lower left", frameon=False, fontsize=9)
        residual.plot(x, error, "o", color="#c62828", ms=6)
        for density, value in zip(x, error):
            residual.annotate(f"{value:+.2f}%", (density, value),
                              xytext=(0, 9), textcoords="offset points", ha="center")
        residual.axhline(0, color="0.55", lw=0.8)
        residual.set_ylabel("Difference from published calibration (%)")
        residual.set_ylim(-0.5, 8)
        for axis in axes:
            axis.set_xlabel(r"Relative density, $\bar{\rho}$")
            axis.set_xticks(x, ["20%", "25%", "30%"])
            axis.set_xlim(0.19, 0.31)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=0.2)
        fig.suptitle("Published lattice C(T) geometry calibration", y=0.98)
        fig.text(0.5, 0.025,
                 rf"Snub-square; $a/W=0.25$, $\lambda=1$. Source: {DOI}, Table C.2.",
                 ha="center", fontsize=8.5)
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / "published-lattice-ct-calibration"
        png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
        fig.savefig(png, dpi=240)
        fig.savefig(svg)
        plt.close(fig)
    return png, svg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for row in published_ct_comparison():
        print(row)
    for path in plot_published_ct(args.output_dir):
        print(path.resolve())
