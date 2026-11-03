"""Real-data layout previews, explicitly separate from recalculated paper results.

Read current saved properties for original IDs1-1000 and the ordered reference.
No source data, processing defaults, validation plots or manuscript is changed.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TOPOLOGIES = ("tri", "kagome", "hex", "FCC")
NAMES = {"tri": "Triangular", "kagome": "Kagome", "hex": "Hexagonal", "FCC": "FCC"}
COLORS = {"tri": "#c43b3b", "kagome": "#333333", "hex": "#3465a4", "FCC": "#67853b"}
DESIGNS = (("full", "Full disorder", "", "#3974a8"),
           ("mechanism", "Mechanism-based disorder", "Target-xs", "#d48332"))


def read_frame(path, role, sources, **kwargs):
    path = Path(path)
    contents = path.read_bytes()
    source_hash = hashlib.sha256(contents).hexdigest()
    key = str(path.resolve())
    if key in sources and sources[key]["sha256"] != source_hash:
        raise ValueError("Source changed between preview reads: " + key)
    sources[key] = {"path": key, "sha256": source_hash, "role": role}
    return pd.read_csv(io.StringIO(contents.decode("utf-8-sig")), **kwargs)


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values) & (values > 0)]
    if not len(finite):
        return {key: None for key in ("min", "q05", "q25", "median", "q75", "q95", "max")}
    return dict(zip(("min", "q05", "q25", "median", "q75", "q95", "max"),
                    (float(x) for x in np.quantile(finite, (0, .05, .25, .5, .75, .95, 1)))))


def property_records(data_root, sources):
    points, summaries, periodic = [], [], {}
    for topology in TOPOLOGIES:
        for design, _, part, _ in DESIGNS:
            folder = data_root / "disNodes" / part / "0.2" / topology / "MLdata"
            for mode, column, factor, unit in (("FT", "K_JIC", 1/math.sqrt(1000), "MPa sqrt(m)"),
                                              ("UT", "Ductility", 100., "%")):
                path = folder / (mode + "-disNodes-allProps.csv")
                frame = read_frame(path, "current_saved_" + mode + "_properties", sources, index_col=0)
                ids = pd.to_numeric(frame.index, errors="raise").to_numpy(dtype=float)
                if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all() or len(set(ids)) != len(ids):
                    raise ValueError("Non-unique/integer saved property identities: " + str(path))
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)*factor
                reference = values[ids == 0]
                if len(reference) != 1 or not np.isfinite(reference[0]) or reference[0] <= 0:
                    raise ValueError("Missing positive periodic reference: " + str(path))
                key = (topology, mode)
                if key in periodic and not math.isclose(periodic[key], float(reference[0]), rel_tol=1e-10):
                    raise ValueError("Periodic reference differs between disorder designs: " + str(key))
                periodic[key] = float(reference[0])
                selected = (ids >= 1) & (ids <= 1000)
                subset = values[selected]
                for sim_id, value in zip(ids[selected], subset):
                    points.append({"topology": topology, "mode": mode, "design": design,
                                   "sim_id": int(sim_id), "metric": column, "value": float(value) if np.isfinite(value) else None,
                                   "unit": unit, "plotted": bool(np.isfinite(value) and value > 0),
                                   "source": str(path), "source_sha256": sources[str(path.resolve())]["sha256"],
                                   "status": "current_saved_value_pending_recalculation"})
                summaries.append({"topology": topology, "mode": mode, "design": design, "unit": unit,
                                  "expected_original_ids": 1000, "present_saved_ids": int(selected.sum()),
                                  "missing_saved_ids": int(1000-selected.sum()),
                                  "finite_positive_values": int((np.isfinite(subset) & (subset > 0)).sum()),
                                  "nonfinite_saved_values": int((~np.isfinite(subset)).sum()),
                                  "nonpositive_saved_values": int((np.isfinite(subset) & (subset <= 0)).sum()),
                                  "periodic_id0_value": periodic[key], **quantiles(subset)})
    return points, summaries, periodic


def plot_ranges(ax, summaries, periodic, mode):
    from matplotlib.lines import Line2D
    for position, topology in enumerate(TOPOLOGIES):
        for offset, (design, _, _, color) in zip((-.15, .15), DESIGNS):
            row = next(r for r in summaries if r["mode"] == mode and r["topology"] == topology and r["design"] == design)
            x = position+offset
            if row["median"] is not None:
                ax.vlines(x, row["min"], row["max"], color=color, alpha=.4, lw=1)
                ax.vlines(x, row["q05"], row["q95"], color=color, alpha=.55, lw=6)
                ax.vlines(x, row["q25"], row["q75"], color=color, alpha=.9, lw=12)
                ax.plot(x, row["median"], "o", ms=4, mfc="white", mec=color, mew=1.2)
            ax.text(x, -.10, str(row["finite_positive_values"]), color=color, ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=8)
        ax.plot(position, periodic[topology, mode], "D", color="#111111", ms=6, zorder=5)
    ax.set_xticks(range(4), [NAMES[t] for t in TOPOLOGIES])
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.text(-.055, -.10, "n =", transform=ax.transAxes, va="top", fontsize=8)
    return [Line2D([], [], color=color, lw=6, label=name) for _, name, _, color in DESIGNS] + [
        Line2D([], [], marker="D", color="#111111", ls="none", ms=6, label="Periodic, ID 0")]


def save_figure(fig, output, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / (name+"."+extension), dpi=250)


def run(root, data_root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root, data_root, output = Path(root), Path(data_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    pilot_path = root / "audit/shared/uat-elastic-window-pilot.json"
    pilot_bytes = pilot_path.read_bytes()
    pilot = json.loads(pilot_bytes)
    sources[str(pilot_path.resolve())] = {"path": str(pilot_path.resolve()), "role": "prior_source_bound_window_pilot",
                                        "sha256": hashlib.sha256(pilot_bytes).hexdigest()}
    points, summaries, periodic = property_records(data_root, sources)
    fit_rows, curve_rows, crossing_rows = [], [], []
    style = {"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
             "pdf.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.1))
        fit_colors = {5: "#c46d23", 8: "#3c85af", 12: "#864d99"}
        for ax, topology in zip(axes.flat, TOPOLOGIES):
            text = []
            for kind in ("ordered", "full_id1"):
                case = next(r for r in pilot["cases"] if r["lat"] == topology and r["kind"] == kind)
                raw = read_frame(case["path"], "raw_UAT_initial_fit_example", sources, header=None).iloc[:, :2].to_numpy(float)
                if sources[str(Path(case["path"]).resolve())]["sha256"] != case["sha256"]:
                    raise ValueError("Initial-fit source differs from its previous audited bytes")
                if not np.isfinite(raw[:14]).all():
                    raise ValueError("Fit example lacks a genuine finite initial interval")
                x, y = raw[:14].T
                ax.plot(100*x, y, "o-", color="#555555" if kind == "ordered" else "#bca384", ms=3,
                        lw=.9, alpha=.8)
                for n in ((12,) if kind == "ordered" else (5, 8, 12)):
                    slope, intercept = np.polyfit(x[:n], y[:n], 1)
                    color = "#555555" if kind == "ordered" else fit_colors[n]
                    ax.plot(100*x[:n], slope*x[:n]+intercept, color=color,
                            ls="--" if kind == "ordered" else "-", lw=1.3)
                    fit_rows.append({"topology": topology, "kind": kind, "samples": n, "E_app_MPa": float(slope),
                                     "intercept_MPa": float(intercept), "last_strain": float(x[n-1]),
                                     "source": case["path"], "source_sha256": case["sha256"],
                                     "status": "illustrative_window_not_an_accepted_modulus"})
                    text.append(("Ordered, n=12" if kind == "ordered" else "Full ID 1, n="+str(n)) +
                                ": {:.3f} GPa".format(slope/1000))
                for sample, (strain, stress) in enumerate(zip(x, y)):
                    curve_rows.append({"figure": "elastic_fit", "topology": topology, "kind": kind,
                                       "sample": sample, "strain": float(strain), "stress_MPa": float(stress)})
            ax.set_title(NAMES[topology])
            ax.set_xlabel("Grip-displacement/body-height strain (%)")
            ax.set_ylabel("Stress (MPa)")
            ax.grid(alpha=.18)
            ax.text(.03, .97, "\n".join(text), transform=ax.transAxes, va="top", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86})
        fig.suptitle("Real UAT initial-loading examples: window sensitivity", y=.985, fontsize=12)
        handles = [Line2D([], [], color="#555555", marker="o", ms=3, label="Ordered raw response"),
                   Line2D([], [], color="#bca384", marker="o", ms=3, label="Full-disorder ID 1 raw response")]
        handles += [Line2D([], [], color=color, label="Full ID 1 fit: {} points".format(n)) for n, color in fit_colors.items()]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .045), ncol=3, frameon=False)
        fig.text(.5, .014, "Illustrative fits with an intercept; not accepted elastic intervals. E_app includes grip compliance.",
                 ha="center", fontsize=8)
        fig.subplots_adjust(top=.92, bottom=.20, left=.09, right=.98, hspace=.42, wspace=.28)
        save_figure(fig, output, "uat-elastic-fit-examples")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.2, 4.9))
        handles = plot_ranges(ax, summaries, periodic, "FT")
        ax.set_ylabel(r"Saved $K_J$ (MPa$\sqrt{\mathrm{m}}$)")
        ax.set_title("FT population layout preview | Ti, nominal 10 mm / 20%")
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .13), ncol=3, frameon=False)
        fig.text(.5, .072, "Median circle; thick IQR; 5-95% interval; thin full min-max. Numbers show available saved values.",
                 ha="center", fontsize=8)
        fig.text(.5, .025, "CURRENT SAVED VALUES - RECALCULATION PENDING. No new M or toughness calculation; no FCC literature benchmark.",
                 ha="center", fontsize=8, color="#913e25")
        fig.subplots_adjust(left=.11, right=.98, top=.90, bottom=.30)
        save_figure(fig, output, "ft-population-layout-preview")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
        ax = axes[0]
        for topology in TOPOLOGIES:
            path = data_root / "disNodes/0.2" / topology / "MLdata/UT-UT-disNodes-allOUT.csv"
            frame = read_frame(path, "saved_processed_periodic_UAT_curve", sources, index_col=0, nrows=2)
            if frame.index.tolist() != [0, 0]:
                raise ValueError("Expected strain row and periodic ID0 response in saved UAT table")
            x, y = frame.iloc[0, 1:].to_numpy(float), frame.iloc[1, 1:].to_numpy(float)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Saved periodic preview contains nonfinite values")
            peak = int(np.argmax(y))
            zero = np.flatnonzero(y[peak:] <= 0)
            stop = int(peak+zero[0]) if len(zero) else len(y)
            crossing = np.flatnonzero(y[peak:stop] <= .25*y[peak])
            cross = int(peak+crossing[0]) if len(crossing) else None
            valid_cross = bool(cross is not None and y[cross] > 0)
            ax.plot(100*x[:stop], y[:stop], color=COLORS[topology], label=NAMES[topology], lw=1.2)
            if valid_cross:
                ax.plot(100*x[cross], y[cross], "o", mfc="white", mec=COLORS[topology], ms=5)
                if not math.isclose(100*x[cross], periodic[topology, "UT"], rel_tol=1e-9):
                    raise ValueError("Saved periodic ductility and saved processed crossing do not match")
            crossing_rows.append({"topology": topology, "saved_25pct_crossing_observed_positive": valid_cross,
                                  "sample": cross, "strain_percent": float(100*x[cross]) if valid_cross else None,
                                  "stress_MPa": float(y[cross]) if valid_cross else None,
                                  "display_samples": stop, "source": str(path),
                                  "limit": "Saved processed curve; zero-tail truncation is display-only, not proof of native termination."})
            for sample in range(stop):
                curve_rows.append({"figure": "periodic_UAT_saved_curve", "topology": topology, "kind": "ordered",
                                   "sample": sample, "strain": float(x[sample]), "stress_MPa": float(y[sample])})
        ax.set_xlabel("Saved grip-displacement/body-height strain (%)")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title("(a) Saved periodic Ti responses")
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=8)
        handles = plot_ranges(axes[1], summaries, periodic, "UT")
        axes[1].set_ylabel(r"Saved $\varepsilon_{25}$ (%)")
        axes[1].set_title("(b) Saved ductility distributions")
        handles += [Line2D([], [], color="#555555", marker="o", mfc="white", ls="none", label="Saved 25% crossing")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .13), ncol=4, frameon=False)
        fig.text(.5, .073, "Population intervals: median/IQR, 5-95%, full min-max. Native-tail and censoring audit pending.",
                 ha="center", fontsize=8)
        fig.text(.5, .025, "CURRENT SAVED VALUES - RECALCULATION PENDING. Al/triangular validation remains a separate material-labelled panel.",
                 ha="center", fontsize=8, color="#913e25")
        fig.subplots_adjust(left=.07, right=.98, top=.91, bottom=.32, wspace=.30)
        save_figure(fig, output, "uat-population-layout-preview")
        plt.close(fig)

    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != row["sha256"] for path, row in sources.items()):
        raise ValueError("A preview source changed during generation")
    write_rows(output / "population-preview-points.csv", points)
    write_rows(output / "population-preview-summary.csv", summaries)
    write_rows(output / "elastic-fit-example-results.csv", fit_rows)
    write_rows(output / "curve-preview-points.csv", curve_rows)
    write_rows(output / "periodic-UAT-crossings.csv", crossing_rows)
    metadata = {"status": "Exploratory real-data examples; current saved values, pending final reprocessing",
                "population_scope": "Original IDs1-1000 separately per topology/design/metric; ID0 reference only. No paired-complete-case or outlier filtering.",
                "normalisation": {"FT": "Saved MPa sqrt(mm) divided by sqrt(1000)", "UT": "Saved strain multiplied by100"},
                "range_definition": "Median/IQR, central5-95percentiles and full observed min-max; population spread, not confidence or measurement uncertainty.",
                "eligibility": "Finite positive values of each current saved metric; existing source omissions remain missing, not recovered or imputed.",
                "sources": list(sources.values()), "summaries": summaries, "periodic_UAT_crossings": crossing_rows,
                "limits": ["Historical metrics may change substantially after the accepted scientific corrections.",
                           "No new modulus/toughness method, source substitution, field reconstruction or FEA execution.",
                           "Initial fit windows are sensitivity examples and do not establish pre-plastic linearity or body material modulus.",
                           "Periodic curves are saved processed responses; display truncation does not attest native solver history length.",
                           "No synthetic stress-strain bands or padded missing observations; Al benchmark data are not pooled with Ti.",
                           "These previews do not replace current paper validation exports or establish FCC benchmark agreement."]}
    (output / "preview-metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Z:/p1/sims/Ti/Validation"))
    parser.add_argument("--data-root", type=Path, default=Path("Z:/p1/data/Ti"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run(args.root, args.data_root, args.out or args.root / "audit/previews")
    print("Wrote3 labelled real-data preview figures from{} source files.".format(len(result["sources"])))
