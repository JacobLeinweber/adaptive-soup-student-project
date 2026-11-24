"""Figures and the results page for project_docs/, built only from the tables in results/.

The main part of the page shows the shared starting point (since meeting 3: every seed starts from the supplied
seed-52 source-only model), from the tables in results/shared_init/tables/. The domain probe exists only for the
previous variant (every seed from its own source-only model) and is read from results/tables/.

The script reads no raw data, no labels and no checkpoints. It merges every T*_runs_all_splits.csv of a folder
(and the confusion, domain-probe and cost tables), checks that rows which occur in several tables agree, and writes:

    project_docs/figures/fig1_weighted_f1_by_group.png        shared start
    project_docs/figures/fig2_method_minus_control.png        shared start
    project_docs/figures/fig3_difference_to_baseline.png      shared start
    project_docs/figures/fig4_domain_probe.png                previous variant
    project_docs/figures/fig5_confusion_MHEALTH_to_PAMAP2.png shared start
    project_docs/figures/fig6_training_time.png               shared start
    project_docs/results.md

If the tables of the alignment-weight sensitivity (T7_*.csv, seed 52) exist, make_sensitivity_figures.py adds
figures 7 to 9 and its own section. If the stage-3 tables of the shared start exist, make_soup_figures.py adds
figures 10 to 12 and its section. make_shared_init_figures.py adds the section on the previous variant with
figures 13 to 15 (old against new); figure 4 follows in the same section. The figure numbers are fixed
(FIGURE_NUMBERS), because the meeting-3 brief refers to them.

All values are on the target test split unless stated otherwise. A group value (for example
"Cross-dataset") is first averaged over the transfers of the group within each seed; mean and
sample standard deviation are then taken over the five seeds.

Example: python make_figures.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"  # start per seed (previous variant), sensitivity, comparison of the two starts
MAIN_TABLES = ROOT / "results" / "shared_init" / "tables"  # shared start: the main part of the page
OUT = ROOT / "project_docs"
FIGURES = OUT / "figures"
SUPPLIED_SEED = 52  # seed of the supplied source-only checkpoints, the shared start of every seed
PREVIOUS_SECTION = "Previous variant: own starting point per seed"
# A figure keeps its number when the page changes, because the meeting-3 brief refers to figures by number.
FIGURE_NUMBERS = {"groups": 1, "paired": 2, "heatmap": 3, "probe": 4, "confusion": 5, "costs": 6,
                  "sensitivity": 7, "soup": 10, "comparison": 13}

CONTROL_SUFFIX = "lambda0_target_pass"  # a control: alignment weight 0, target windows still pass through the model
BASELINE = "baseline"
METRICS = {"weighted_f1": "Weighted F1", "macro_f1": "Macro-F1", "accuracy": "Accuracy"}
METHOD_NAMES = {"deep_coral": "Deep CORAL", "dann": "DANN", "dsan": "DSAN"}
CONTROL_NAMES = {"deep_coral": "CORAL control", "dann": "DANN control", "dsan": "DSAN control"}
OTHER_NAMES = {BASELINE: "Baseline", "source_only_continued": "Source-only continued"}
# One fixed colour per method (a control takes a light tint of its method), grey for the baseline.
METHOD_COLORS = {"deep_coral": "#2a78d6", "dann": "#eb6834", "dsan": "#1baf7a"}
SPARE_COLORS = ("#eda100", "#e87ba4", "#008300", "#6250d6")
GREY, INK, INK_SOFT, GRID, AXIS, SURFACE = "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FAMILY_NAMES = {"cross_dataset": "Cross-dataset", "cross_gender": "Cross-gender", "cross_age": "Cross-age"}
SUBGROUPS = {"below30": "<30", "age30plus": "≥30"}
CONFUSION_TRANSFER = "MHEALTH_to_PAMAP2"


# ---------------------------------------------------------------- loading

def read_merged(tables: Path, pattern: str, keys: list[str], values: list[str]) -> pd.DataFrame:
    """Read all tables of the folder that match the pattern; rows with the same key must carry the same values."""
    paths = sorted(tables.glob(pattern))
    if not paths:
        raise SystemExit(f"No table matches {tables.relative_to(ROOT)}/{pattern}")
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    distinct = data.groupby(keys, sort=False)[values].nunique(dropna=False)
    conflicts = distinct[(distinct > 1).any(axis=1)]
    if len(conflicts):
        raise SystemExit(f"Tables {pattern} disagree for {len(conflicts)} keys, first:\n{conflicts.head()}")
    return data.drop_duplicates(keys).reset_index(drop=True)


class Results:
    """All merged tables of one folder and the variants found in them."""

    def __init__(self, tables: Path) -> None:
        key = ["transfer", "seed", "variant"]
        self.runs = read_merged(tables, "T*_runs_all_splits.csv", [*key, "split"], [*METRICS, "n_windows"])
        self.confusion = read_merged(tables, "T*_confusion_target_test.csv", [*key, "true_index", "predicted_index"], ["count"])
        # The domain probe was computed only for the start per seed, not again for the shared start.
        has_probe = any(tables.glob("T*_domain_probe_runs.csv"))
        self.probe = read_merged(tables, "T*_domain_probe_runs.csv", key, ["probe_accuracy"]) if has_probe else None
        self.costs = read_merged(tables, "T*_costs_per_run.csv", key, ["train_seconds"])

        latest = pd.read_csv(sorted(tables.glob("T*_runs_all_splits.csv"))[-1])  # keeps the order of the transfers
        self.transfers = list(dict.fromkeys(latest["transfer"]))
        self.family = dict(zip(latest["transfer"], latest["family"]))
        self.families = list(dict.fromkeys(latest["family"]))
        self.seeds = sorted(self.runs["seed"].unique())

        found = set(self.runs["variant"])
        if BASELINE not in found:
            raise SystemExit("No baseline in the tables")
        controls = {}
        for variant in found:
            if variant.endswith(CONTROL_SUFFIX):  # "lambda0_target_pass" alone is the control of Deep CORAL
                controls[variant[: -len(CONTROL_SUFFIX)].rstrip("_") or "deep_coral"] = variant
        known = [m for m in METHOD_NAMES if m in controls and m in found]
        self.methods = known + sorted(m for m in controls if m in found and m not in known)
        self.control = {m: controls[m] for m in self.methods}
        self.variants = [BASELINE] + [v for m in self.methods for v in (m, self.control[m])]
        self.extra = sorted(found - set(self.variants))  # shown only in the detailed table

        self.name, self.color, self.is_control = dict(OTHER_NAMES), {BASELINE: GREY}, {}
        spare = iter(SPARE_COLORS)
        for method in self.methods:
            self.name[method] = METHOD_NAMES.get(method, method)
            self.name[self.control[method]] = CONTROL_NAMES.get(method, f"{self.name[method]} control")
            self.color[method] = METHOD_COLORS.get(method) or next(spare)
            self.color[self.control[method]] = self.color[method]
            self.is_control[self.control[method]] = True
        for variant in self.extra:
            self.name.setdefault(variant, variant)

        self.groups = [("All %d" % len(self.transfers), self.transfers)]
        for family in self.families:
            members = [t for t in self.transfers if self.family[t] == family]
            self.groups.append((f"{FAMILY_NAMES.get(family, family)} ({len(members)})", members))

    def per_seed(self, members: list[str], metric: str = "weighted_f1", split: str = "target_test") -> pd.DataFrame:
        """Seed x variant, in percent; a group is averaged over its transfers within each seed."""
        rows = self.runs[(self.runs["split"] == split) & self.runs["transfer"].isin(members)]
        table = rows.pivot(index=["transfer", "seed"], columns="variant", values=metric)
        return table.groupby("seed").mean() * 100

    def probe_per_seed(self, members: list[str]) -> pd.DataFrame:
        rows = self.probe[self.probe["transfer"].isin(members)]
        return rows.pivot(index=["transfer", "seed"], columns="variant", values="probe_accuracy").groupby("seed").mean() * 100


def transfer_label(name: str) -> str:
    """'RealWorld_below30_to_RealWorld_age30plus' -> 'RealWorld <30 → ≥30'."""
    source, target = name.split("_to_")
    s, t = source.split("_", 1), target.split("_", 1)
    if len(s) == 2 and len(t) == 2 and s[0] == t[0]:
        return f"{s[0]} {SUBGROUPS.get(s[1], s[1])} → {SUBGROUPS.get(t[1], t[1])}"
    return f"{source} → {target}"


def paired(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """Mean and sample sd of the per-seed difference a - b, and the number of seeds with a > b."""
    difference = a - b
    return difference.mean(), difference.std(ddof=1), int((difference > 0).sum())


# ---------------------------------------------------------------- figure helpers

def tint(color: str, share: float = 0.6) -> tuple:
    """Mix a colour with white."""
    return tuple(1 - (1 - channel) * (1 - share) for channel in to_rgb(color))


def bar_style(results: Results, variant: str) -> dict:
    color = results.color[variant]
    if results.is_control.get(variant):  # light tint and hatch, so that a control does not depend on colour alone
        return {"facecolor": tint(color), "edgecolor": color, "hatch": "////", "linewidth": 0.8}
    return {"facecolor": color, "edgecolor": SURFACE, "linewidth": 0.8}


def new_figure(rows: int, columns: int, size: tuple, **kwargs):
    figure, axes = plt.subplots(rows, columns, figsize=size, facecolor=SURFACE, **kwargs)
    for axis in np.atleast_1d(axes).ravel():
        axis.set_facecolor(SURFACE)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(AXIS)
        axis.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    return figure, axes


def save(figure, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: project_docs/figures/{name}")
    return f"figures/{name}"


# ---------------------------------------------------------------- figures

def figure_groups(results: Results) -> str:
    """Weighted F1 per group and variant, error bars = sd over the seeds."""
    figure, axis = new_figure(1, 1, (11.5, 4.8))
    width = 0.8 / len(results.variants)
    for i, variant in enumerate(results.variants):
        for g, (_, members) in enumerate(results.groups):
            values = results.per_seed(members)[variant]
            x = g + (i - (len(results.variants) - 1) / 2) * width
            axis.bar(x, values.mean(), width * 0.92, label=results.name[variant] if g == 0 else None, **bar_style(results, variant))
            axis.errorbar(x, values.mean(), yerr=values.std(ddof=1), color=INK, linewidth=0.9, capsize=2)
            if not results.is_control.get(variant):
                axis.text(x, values.mean() + values.std(ddof=1) + 1.0, f"{values.mean():.1f}", ha="center", va="bottom",
                          fontsize=8, color=INK_SOFT)
    axis.set_xticks(range(len(results.groups)), [name for name, _ in results.groups], fontsize=10, color=INK)
    axis.set_ylabel("Weighted F1 on target test (%)", color=INK_SOFT, fontsize=10)
    axis.set_ylim(0, 100)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(ncol=len(results.variants), frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    figure.suptitle("Weighted F1 on the target test split, shared start, mean ± SD over five seeds", fontsize=12, color=INK,
                    y=1.04)
    return save(figure, "fig1_weighted_f1_by_group.png")


def figure_paired(results: Results) -> str:
    """Method minus its own control per transfer: one point per seed and the mean."""
    rows = [(transfer_label(t), [t]) for t in results.transfers] + [results.groups[0]]
    figure, axes = new_figure(1, len(results.methods), (4.2 * len(results.methods) + 1.2, 6.2), sharex=True, sharey=True)
    positions = list(range(len(rows), 0, -1))
    positions[-1] -= 0.5  # a gap before the row for all transfers
    for axis, method in zip(np.atleast_1d(axes), results.methods):
        color = results.color[method]
        for y, (_, members) in zip(positions, rows):
            table = results.per_seed(members)
            difference = table[method] - table[results.control[method]]
            axis.scatter(difference, [y] * len(difference), s=22, color=color, alpha=0.55, linewidths=0, zorder=2)
            axis.scatter(difference.mean(), y, s=70, marker="D", color=color, edgecolors=INK, linewidths=1.0, zorder=3)
        axis.axvline(0, color=INK_SOFT, linewidth=1.0, zorder=1)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_title(f"{results.name[method]} − {results.name[results.control[method]]}", fontsize=11, color=INK)
        axis.set_xlabel("Difference in weighted F1 (points)", fontsize=9, color=INK_SOFT)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(positions, [name for name, _ in rows], fontsize=9, color=INK)
    first.scatter([], [], s=22, color=GREY, alpha=0.55, linewidths=0, label="one seed")
    first.scatter([], [], s=70, marker="D", color=GREY, edgecolors=INK, linewidths=1.0, label="mean of five seeds")
    figure.legend(frameon=False, fontsize=9, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    figure.suptitle("Each method minus its own control run (target test, shared start), paired by seed", fontsize=12,
                    color=INK, y=1.04)
    return save(figure, "fig2_method_minus_control.png")


def figure_heatmap(results: Results) -> str:
    """Transfers x methods: mean difference to the baseline, values in the cells."""
    rows = [(transfer_label(t), [t]) for t in results.transfers] + results.groups
    values = np.array([[paired(results.per_seed(members)[method], results.per_seed(members)[BASELINE])[0]
                        for method in results.methods] for _, members in rows])
    limit = float(np.abs(values).max())
    colormap = LinearSegmentedColormap.from_list("diverging", ["#e34948", "#f0efec", "#2a78d6"])
    figure, axis = new_figure(1, 1, (1.9 * len(results.methods) + 3.2, 0.42 * len(rows) + 1.6))
    image = axis.imshow(values, cmap=colormap, vmin=-limit, vmax=limit, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            strong = abs(values[i, j]) > 0.6 * limit
            axis.text(j, i, signed(values[i, j]), ha="center", va="center", fontsize=10,
                      color="#ffffff" if strong else INK)
    axis.set_xticks(range(len(results.methods)), [results.name[m] for m in results.methods], fontsize=10, color=INK)
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(rows)), [name for name, _ in rows], fontsize=9, color=INK)
    axis.axhline(len(results.transfers) - 0.5, color=SURFACE, linewidth=6)
    for side in axis.spines.values():
        side.set_visible(False)
    bar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.03)
    bar.set_label("Weighted F1: method − baseline (points)", fontsize=9, color=INK_SOFT)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, colors=INK_SOFT, length=0)
    axis.set_title("Difference to the baseline on target test, shared start, mean over five seeds", fontsize=12, color=INK,
                   pad=28)
    return save(figure, "fig3_difference_to_baseline.png")


def figure_probe(results: Results) -> str:
    """Domain-probe accuracy per variant, mean ± sd, with the chance level."""
    variants = [v for v in results.variants if v in set(results.probe["variant"])]
    figure, axes = new_figure(1, len(results.groups), (3.1 * len(results.groups) + 1.6, 3.6), sharex=True, sharey=True)
    positions = range(len(variants), 0, -1)
    for axis, (group, members) in zip(np.atleast_1d(axes), results.groups):
        table = results.probe_per_seed(members)
        for y, variant in zip(positions, variants):
            mean, sd = table[variant].mean(), table[variant].std(ddof=1)
            control = results.is_control.get(variant)
            axis.errorbar(mean, y, xerr=sd, color=INK, linewidth=0.9, capsize=2, zorder=2)
            axis.scatter(mean, y, s=60, color=SURFACE if control else results.color[variant], edgecolors=results.color[variant],
                         linewidths=1.6, zorder=3)
            axis.text(mean + sd + 3.0, y, f"{mean:.1f}", va="center", fontsize=8, color=INK_SOFT)
        axis.axvline(50, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
        axis.text(50.8, len(variants) + 0.55, "chance", fontsize=8, color=INK_SOFT, va="center")
        axis.set_xlim(45, 108)
        axis.set_ylim(0.4, len(variants) + 0.9)
        axis.set_xticks([50, 60, 70, 80, 90, 100])
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_title(group, fontsize=10, color=INK)
        axis.set_xlabel("Probe accuracy (%)", fontsize=9, color=INK_SOFT)
    np.atleast_1d(axes)[0].set_yticks(list(positions), [results.name[v] for v in variants], fontsize=9, color=INK)
    figure.suptitle("Domain probe: can a linear classifier tell source from target features? Mean ± SD over five seeds",
                    fontsize=12, color=INK, y=1.04)
    return save(figure, "fig4_domain_probe.png")


def figure_confusion(results: Results) -> tuple[str, list[str]]:
    """Row-normalised confusion matrices, averaged over the seeds: baseline, best and worst method of one transfer."""
    scores = results.per_seed([CONFUSION_TRANSFER])
    ranked = sorted(results.methods, key=lambda m: scores[m].mean())
    shown = list(dict.fromkeys([BASELINE, ranked[-1], ranked[0]]))
    cells = results.confusion[results.confusion["transfer"] == CONFUSION_TRANSFER]
    names = cells.drop_duplicates("true_index").sort_values("true_index")["true_name"].tolist()
    colormap = LinearSegmentedColormap.from_list("sequential", [SURFACE, "#86b6ef", "#2a78d6", "#0d366b"])
    figure, axes = new_figure(1, len(shown), (4.3 * len(shown), 4.6), sharey=True)
    for axis, variant in zip(np.atleast_1d(axes), shown):
        per_seed = []
        for _, run in cells[cells["variant"] == variant].groupby("seed"):
            matrix = run.pivot(index="true_index", columns="predicted_index", values="count").to_numpy(dtype=float)
            per_seed.append(matrix / matrix.sum(axis=1, keepdims=True))
        mean = np.mean(per_seed, axis=0) * 100
        axis.imshow(mean, cmap=colormap, vmin=0, vmax=100)
        for i in range(len(names)):
            for j in range(len(names)):
                axis.text(j, i, f"{mean[i, j]:.0f}", ha="center", va="center", fontsize=10,
                          color="#ffffff" if mean[i, j] > 55 else INK)
        axis.set_xticks(range(len(names)), names, fontsize=9, color=INK, rotation=30, ha="right")
        axis.set_xlabel("Predicted", fontsize=9, color=INK_SOFT)
        role = "" if variant == BASELINE else (" (best method)" if variant == ranked[-1] else " (worst method)")
        axis.set_title(f"{results.name[variant]}{role}\nweighted F1 {scores[variant].mean():.1f} ± {scores[variant].std(ddof=1):.1f} %",
                       fontsize=10, color=INK)
        for side in axis.spines.values():
            side.set_visible(False)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(range(len(names)), names, fontsize=9, color=INK)
    first.set_ylabel("True", fontsize=9, color=INK_SOFT)
    figure.suptitle(f"{transfer_label(CONFUSION_TRANSFER)}, target test, shared start: share of each true class (%), "
                    "mean over five seeds", fontsize=12, color=INK, y=1.03)
    return save(figure, f"fig5_confusion_{CONFUSION_TRANSFER}.png"), shown


def figure_costs(results: Results) -> str:
    """Training time per variant: sum over all runs and mean per run."""
    timed = results.costs.dropna(subset=["train_seconds"])
    variants = [v for v in results.variants if v in set(timed["variant"])]
    seconds = timed.groupby("variant")["train_seconds"]
    figure, axes = new_figure(1, 2, (11.5, 3.6), sharey=True)
    positions = range(len(variants), 0, -1)
    panels = ((seconds.sum() / 60, "Total training time (min)"), (seconds.mean(), "Mean training time per run (s)"))
    for axis, (values, label) in zip(axes, panels):
        for y, variant in zip(positions, variants):
            axis.barh(y, values[variant], 0.7, **bar_style(results, variant))
            axis.text(values[variant] + values.max() * 0.015, y, f"{values[variant]:.1f}", va="center", fontsize=9, color=INK_SOFT)
        axis.set_xlim(0, values.max() * 1.12)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel(label, fontsize=9, color=INK_SOFT)
    axes[0].set_yticks(list(positions), [f"{results.name[v]} ({seconds.count()[v]} runs)" for v in variants], fontsize=9, color=INK)
    figure.suptitle("Training time on Apple M1 (MPS), training loop only, shared start", fontsize=12, color=INK, y=1.02)
    return save(figure, "fig6_training_time.png")


# ---------------------------------------------------------------- results page

def signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "−")


def cell(values: pd.Series) -> str:
    return f"{values.mean():.1f} ± {values.std(ddof=1):.1f}"


def difference_cell(a: pd.Series, b: pd.Series) -> str:
    mean, sd, higher = paired(a, b)
    return f"{signed(mean)} ± {sd:.1f} ({higher}/{len(a)})"


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows]) + "\n"


def observations(results: Results) -> list[str]:
    """Statements that follow directly from the numbers; every value is computed here."""
    name, everything = results.name, results.groups[0][1]
    overall, lines = results.per_seed(everything), []
    lines.append("Averaged over all transfers, weighted F1 on target test is "
                 + ", ".join(f"{cell(overall[v])} % for {name[v]}" for v in [BASELINE, *results.methods]) + ".")
    for method in results.methods:
        control = results.control[method]
        groups = "; ".join(f"{group}: {difference_cell(results.per_seed(m)[method], results.per_seed(m)[control])}"
                           for group, m in results.groups[1:])
        lines.append(f"{name[method]} minus its own control: {difference_cell(overall[method], overall[control])} points over all "
                     f"transfers (in brackets: seeds in which the method is higher). By group: {groups}.")
    lines.append("The control runs (continued training with a target forward pass, alignment weight 0) minus the baseline: "
                 + ", ".join(f"{difference_cell(overall[results.control[m]], overall[BASELINE])} ({name[results.control[m]]})"
                             for m in results.methods) + " points.")
    for i, first in enumerate(results.methods):
        for second in results.methods[i + 1:]:
            groups = "; ".join(f"{group}: {difference_cell(results.per_seed(m)[second], results.per_seed(m)[first])}"
                               for group, m in results.groups[1:])
            lines.append(f"{name[second]} minus {name[first]}: {difference_cell(overall[second], overall[first])} points over all "
                         f"transfers. By group: {groups}.")
    means = pd.DataFrame({t: results.per_seed([t]).mean() for t in results.transfers}).T
    best = means[[BASELINE, *results.methods]].idxmax(axis=1).value_counts()
    lines.append("Highest mean weighted F1 per transfer (baseline and methods only): "
                 + ", ".join(f"{name[v]} on {count}" for v, count in best.items()) + f" of {len(results.transfers)} transfers.")
    below = [t for t in results.transfers if all(means.loc[t, m] < means.loc[t, BASELINE] for m in results.methods)]
    if below:
        parts = [f"{transfer_label(t)} (" + ", ".join(f"{name[m]} {signed(means.loc[t, m] - means.loc[t, BASELINE])}"
                                                       for m in results.methods) + " points)" for t in below]
        lines.append(f"On {listed(parts)} every method has a lower mean weighted F1 than the baseline.")
    for method in results.methods:
        gain = (means[method] - means[BASELINE]).sort_values(ascending=False)
        lines.append(f"{name[method]}: the three transfers with the largest gain over the baseline ("
                     + ", ".join(f"{transfer_label(t)} {signed(gain[t])}" for t in gain.index[:3])
                     + f") carry {gain.iloc[:3].sum() / gain.sum() * 100:.0f} % of the summed mean gain.")
    source = results.per_seed(everything, "macro_f1", "source_test")
    lines.append("Macro-F1 on the source test split, averaged over all transfers and seeds: "
                 + ", ".join(f"{name[v]} {source[v].mean():.1f} %" for v in [BASELINE, *results.methods]) + ".")
    timed = results.costs.dropna(subset=["train_seconds"]).groupby("variant")["train_seconds"].mean()
    untimed = "" if BASELINE in timed else f"; the baseline is the supplied seed-{SUPPLIED_SEED} model and was not trained here"
    lines.append("Mean training time per run: " + ", ".join(f"{name[v]} {timed[v]:.0f} s" for v in results.variants if v in timed)
                 + f" (see the note on Figure {FIGURE_NUMBERS['costs']}{untimed}).")
    return lines


def probe_observations(results: Results) -> list[str]:
    """The domain-probe statements; only the tables of the start per seed contain the probe."""
    name, probe = results.name, results.probe_per_seed(results.groups[0][1])
    available = [v for v in results.variants if v in probe]
    return ["Domain probe over all transfers (accuracy of a linear classifier that separates source from target features; "
            "50 % = not separable): " + ", ".join(f"{name[v]} {cell(probe[v])} %" for v in available) + ".",
            "Domain probe, method minus its own control, paired by seed: "
            + ", ".join(f"{name[m]} {signed(paired(probe[m], probe[results.control[m]])[0])} ± "
                        f"{paired(probe[m], probe[results.control[m]])[1]:.1f} points" for m in results.methods if m in probe) + "."]


def anchor(title: str) -> str:
    """Link target of a markdown heading, as GitHub builds it."""
    return re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")


def listed(values: list) -> str:
    """[100, 445, 545] -> '100, 445 and 545'."""
    items = [str(v) for v in values]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def trained_again(results: Results, previous: Results) -> tuple[list[int], int, float]:
    """Seeds whose runs were trained again for the shared start, their number of runs, and their summed training time
    divided by that of the same training loops (variant, transfer, seed) with the start per seed."""
    key = ["transfer", "seed", "variant"]
    again = results.costs[(results.costs["seed"] != SUPPLIED_SEED) & (results.costs["variant"] != BASELINE)]
    before = previous.costs.merge(again[key], on=key, how="inner")
    if len(before) != len(again) or again["train_seconds"].isna().any() or before["train_seconds"].isna().any():
        raise SystemExit("The training times of the runs trained again do not match the runs with the start per seed")
    return sorted(again["seed"].unique()), len(again), again["train_seconds"].sum() / before["train_seconds"].sum()


PROBE_CAPTION = ("Accuracy of a logistic regression (5-fold cross-validation) that separates source-train from target-train "
                 "windows on the frozen 96-dimensional features; controls are drawn as open circles.")


def probe_section(previous: Results, path: str) -> list[str]:
    """Figure 4 with its observations, as the last part of the section on the previous variant."""
    number = FIGURE_NUMBERS["probe"]
    return ["### Domain probe", "",
            "The domain probe was computed only for the previous variant (five seeds, each from its own source-only model) "
            f"and not again for the shared start; for seed {SUPPLIED_SEED} both variants have the same runs.", "",
            f"**Figure {number}.** {PROBE_CAPTION}", "", f"![Figure {number}]({path})", "",
            "**Observations (domain probe).** Only statements that follow directly from the tables; no causes are claimed.", "",
            *(f"- {line}" for line in probe_observations(previous)), ""]


def results_page(results: Results, figures: dict[str, str], confusion_variants: list[str], again: tuple[list[int], int, float],
                 extra: list[str] = ()) -> None:
    name = results.name
    group_names = [group for group, _ in results.groups]
    seeds_again, runs_again, time_ratio = again
    previous_link = f"[{PREVIOUS_SECTION}](#{anchor(PREVIOUS_SECTION)})"
    page = ["# Results", "",
            f"This page is generated by `make_figures.py` from the tables in `{MAIN_TABLES.relative_to(ROOT)}/` (shared start) "
            f"and `{TABLES.relative_to(ROOT)}/` (alignment-weight sensitivity, previous variant, comparison of the two starts). "
            "Do not edit it by hand.", "",
            "## Setup", "",
            f"Every branch starts from the supplied seed-{SUPPLIED_SEED} source-only model of its transfer, for all "
            f"{len(results.seeds)} seeds ({', '.join(str(s) for s in results.seeds)}) and for methods and controls alike "
            "(shared starting point, requested by the supervisor in meeting 3). The seed only steers the randomness of the "
            "adaptation: the order of the source and target batches, dropout and the initialisation of the DANN discriminator. "
            f"Every method continues training on {len(results.transfers)} transfers with labelled source data and unlabelled "
            "target training windows. Each method has its own control run: the same loop, the same target forward pass and the "
            "same source loss weight, but with the alignment weight set to 0. Target labels are read only after all training "
            "has finished, to compute the metrics below on the target test split of the last epoch.", "",
            f"The seed-{SUPPLIED_SEED} runs are those of stage 2, which already started from this model; the seeds "
            f"{listed(seeds_again)} were trained again ({runs_again} runs, `results/shared_init/`). The baseline is the same "
            "model for every seed, so its SD over the seeds is 0. The previous variant, in which every seed started from its "
            f"own source-only model, is compared with the shared start in the section {previous_link}.", "",
            "Values are mean ± sample standard deviation over the seeds, in percent. A group value is first averaged over the "
            "transfers of the group within each seed.", ""]

    page += ["## Overview", ""]
    for metric, title in METRICS.items():
        rows = [[name[v], *(cell(results.per_seed(members, metric)[v]) for _, members in results.groups)] for v in results.variants]
        page += [f"**{title}, target test**", "", markdown_table(["Variant", *group_names], rows)]

    page += ["## Paired differences", "",
             "Difference in weighted F1 on target test in percentage points, paired by seed: mean ± SD of the five per-seed "
             "differences, and in brackets the number of seeds in which the method is higher.", ""]
    rows = []
    for method in results.methods:
        for label, reference in (("own control", results.control[method]), ("baseline", BASELINE)):
            rows.append([f"{name[method]} − {label}", *(difference_cell(results.per_seed(m)[method], results.per_seed(m)[reference])
                                                         for _, m in results.groups)])
    page += [markdown_table(["Difference", *group_names], rows)]
    for label, pick in (("own control", lambda m: results.control[m]), ("the baseline", lambda m: BASELINE)):
        rows = [[transfer_label(t), *(difference_cell(results.per_seed([t])[m], results.per_seed([t])[pick(m)]) for m in results.methods)]
                for t in results.transfers]
        page += [f"Per transfer, each method minus {label}:", "", markdown_table(["Transfer", *(name[m] for m in results.methods)], rows)]

    shown = ", ".join(name[v] for v in confusion_variants)
    captions = {
        "groups": "Weighted F1 on target test per group of transfers for the baseline, the methods and their controls (hatched); "
                  "error bars show the SD over the seeds.",
        "paired": "Each method minus its own control per transfer: one small point per seed, the diamond is the mean of the seeds; "
                  "points right of the zero line mean the method is higher.",
        "heatmap": "Mean difference in weighted F1 between each method and the baseline for every transfer and group; "
                   "blue cells are above the baseline, red cells below.",
        "confusion": f"Row-normalised confusion matrices on target test for {transfer_label(CONFUSION_TRANSFER)}, averaged over the "
                     f"seeds, for {shown} (best and worst method by mean weighted F1 on this transfer).",
        "costs": "Training time of the training loop on an Apple M1 with MPS, summed over all runs of a variant and as mean per run. "
                 f"The baseline is the supplied seed-{SUPPLIED_SEED} model and has no training time. Seed {SUPPLIED_SEED}: the "
                 f"runs of stage 2 under macOS 26.5. The seeds {listed(seeds_again)} were trained again in one batch under macOS "
                 f"27.2.1 and took {time_ratio:.2f} times as long as the same training loops of the previous variant (stage 2, "
                 "macOS 26.5); the cause is not established (see [runs_log.md](runs_log.md)).",
    }
    page += ["## Figures", ""]
    for key in ("groups", "paired", "heatmap", "probe", "confusion", "costs"):
        number = FIGURE_NUMBERS[key]
        if key == "probe":  # exists only for the previous variant
            page += [f"*Figure {number} (domain probe) was computed only for the previous variant and is shown in the section "
                     f"{previous_link}.*", ""]
            continue
        page += [f"**Figure {number}.** {captions[key]}", "", f"![Figure {number}]({figures[key]})", ""]

    page += ["## Observations", "", "Only statements that follow directly from the tables; no causes are claimed.", ""]
    page += [f"- {line}" for line in observations(results)] + [""]

    page += ["## Detailed table", "", "Weighted F1 on target test per transfer, mean ± SD over the seeds, all variants found in the tables.", ""]
    detailed = results.variants + results.extra
    rows = [[transfer_label(t), *(cell(results.per_seed([t])[v]) for v in detailed)] for t in results.transfers]
    page += [markdown_table(["Transfer", *(name[v] for v in detailed)], rows)]
    if results.extra:
        page += [", ".join(name[v] for v in results.extra) + ": continued training without any target data (alignment weight 0, "
                 "no target forward pass), run with the source loss weight of Deep CORAL.", ""]

    page += list(extra)
    page += ["## Limitations", "",
             "Five seeds that all start from one source-only model, the last epoch without model selection, loss weights taken "
             "over from another benchmark without tuning, and small target test sets on some transfers limit what these numbers "
             "show. See [limitations.md](limitations.md).", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.md").write_text("\n".join(page), encoding="utf-8")
    print("Saved: project_docs/results.md")


def main() -> None:
    results, previous = Results(MAIN_TABLES), Results(TABLES)  # shared start; start per seed (for the domain probe)
    print("Variants:", ", ".join(results.variants), "| detailed only:", ", ".join(results.extra) or "none")
    confusion, confusion_variants = figure_confusion(results)
    figures = {"groups": figure_groups(results), "paired": figure_paired(results), "heatmap": figure_heatmap(results),
               "confusion": confusion, "costs": figure_costs(results)}
    extra = []
    if (TABLES / "T7_sensitivity_summary.csv").is_file():
        from make_sensitivity_figures import sensitivity_section  # imports helpers of this module
        extra = sensitivity_section(first_figure=FIGURE_NUMBERS["sensitivity"])
    if (MAIN_TABLES / "T8_summary_target_test.csv").is_file():
        from make_soup_figures import soup_section  # imports helpers of this module
        extra = [*extra, *soup_section(first_figure=FIGURE_NUMBERS["soup"])]
    from make_shared_init_figures import previous_variant_section  # imports helpers of this module
    extra = [*extra, *previous_variant_section(first_figure=FIGURE_NUMBERS["comparison"]),
             *probe_section(previous, figure_probe(previous))]
    results_page(results, figures, confusion_variants, trained_again(results, previous), extra)


if __name__ == "__main__":
    main()
