"""Figures and the results-page section on the previous variant: stage 2 and stage 3 with the start per seed, next to the
shared starting point (T11) that the rest of the page shows.

Built only from the T11 tables in results/tables/ (written by make_shared_init_tables.py); reads no raw data, no labels
and no checkpoints. make_figures.py calls previous_variant_section(), so that project_docs/results.md keeps a single
generator. "Old": every seed starts from its own source-only model (previous variant); "new": every seed starts from
the supplied seed-52 source-only model (shared start, since meeting 3). Figures:
    project_docs/figures/fig13_shared_init_groups.png
    project_docs/figures/fig14_shared_init_method_minus_control.png
    project_docs/figures/fig15_shared_init_soup.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from make_figures import (CONTROL_NAMES, GREY, GRID, INK, INK_SOFT, METHOD_COLORS, METHOD_NAMES, PREVIOUS_SECTION,
                          SUPPLIED_SEED, SURFACE, TABLES, markdown_table, new_figure, save, signed, transfer_label)
from make_soup_figures import COLORS as SOUP_COLORS
from make_soup_figures import FAMILY_NAMES, GROUP_NAMES, key
from make_soup_figures import NAMES as SOUP_NAMES

CONTROLS = {"deep_coral": "lambda0_target_pass", "dann": "dann_lambda0_target_pass", "dsan": "dsan_lambda0_target_pass"}
BRANCHES = ["baseline", *(v for pair in CONTROLS.items() for v in pair)]
NAMES = {"baseline": "Baseline", **METHOD_NAMES, **{CONTROLS[m]: CONTROL_NAMES[m] for m in CONTROLS}}
COLORS = {"baseline": GREY, **METHOD_COLORS, **{CONTROLS[m]: METHOD_COLORS[m] for m in CONTROLS}}
START_NAMES = {"old": "own source-only model per seed (previous variant)", "new": "shared start: seed-52 model for all seeds"}
SOUP_KINDS = ["baseline", "branch_mean", "ensemble", "bn_avg", "bn_target"]
SOUP_TABLE_KINDS = ["baseline", "branch_mean", "best_src_ce", "best_src_macro_f1", "oracle", "ensemble", "bn_avg", "bn_target"]
SOUP_COMPARISONS = {"1_soup_minus_branch_mean": "Soup − mean of the branches",
                    "2a_soup_minus_best_src_ce": "Soup − best branch by SRC (CE)",
                    "2b_soup_minus_best_src_macro_f1": "Soup − best branch by SRC (macro-F1)",
                    "2c_soup_minus_oracle_best_branch": "Soup − best branch by target test (oracle, uses target labels)",
                    "3_soup_minus_baseline": "Soup − baseline",
                    "5_soup_minus_ensemble": "Soup − output ensemble"}


class Compare:
    """The T11 tables; scores in percent."""

    def __init__(self) -> None:
        self.branches = pd.read_csv(TABLES / "T11_branches_old_vs_new.csv")
        self.method_differences = pd.read_csv(TABLES / "T11_method_differences_old_vs_new.csv")
        self.soup = pd.read_csv(TABLES / "T11_soup_old_vs_new.csv")
        self.soup_differences = pd.read_csv(TABLES / "T11_soup_differences_old_vs_new.csv")
        self.reproduction = pd.read_csv(TABLES / "T11_seed52_soup_reproduction.csv")
        self.transfers = list(dict.fromkeys(self.branches[self.branches["scope"] == "transfer"]["name"]))
        self.groups = ["all", "cross_dataset", "cross_gender", "cross_age"]

    @staticmethod
    def _row(table: pd.DataFrame, **match) -> pd.Series:
        rows = table
        for column, value in match.items():
            rows = rows[rows[column] == value]
        if len(rows) != 1:
            raise SystemExit(f"expected one row for {match}, found {len(rows)}")
        return rows.iloc[0]

    def value(self, table: pd.DataFrame, name: str, variant: str, start: str, column: str = "weighted_f1") -> tuple[float, float]:
        row = self._row(table, name=name, variant=variant)
        return row[f"{start}_{column}_mean"] * 100, row[f"{start}_{column}_sd"] * 100

    def method_difference(self, name: str, method: str, reference: str, start: str) -> tuple[float, float, int]:
        row = self._row(self.method_differences, name=name, method=method, reference=reference)
        return (row[f"{start}_weighted_f1_diff_mean"] * 100, row[f"{start}_weighted_f1_diff_sd"] * 100,
                int(row[f"{start}_weighted_f1_seeds_method_higher"]))

    def soup_difference(self, name: str, comparison: str, a: str, start: str) -> tuple[float, float, int, int, int]:
        row = self._row(self.soup_differences, name=name, comparison=comparison, a=a)
        return (row[f"{start}_weighted_f1_diff_mean"] * 100, row[f"{start}_weighted_f1_diff_sd"] * 100,
                int(row[f"{start}_weighted_f1_seeds_a_higher"]), int(row[f"{start}_weighted_f1_pairs_a_higher"]),
                int(row["n_pairs"]))

    def cell(self, table: pd.DataFrame, name: str, variant: str) -> str:
        (old, old_sd), (new, new_sd) = (self.value(table, name, variant, start) for start in ("old", "new"))
        return f"{old:.1f} ± {old_sd:.1f} → {new:.1f} ± {new_sd:.1f}"

    def method_cell(self, name: str, method: str, reference: str) -> str:
        parts = [self.method_difference(name, method, reference, start) for start in ("old", "new")]
        return " → ".join(f"{signed(m)} ± {sd:.1f} ({k}/5)" for m, sd, k in parts)

    def soup_cell(self, name: str, comparison: str, a: str) -> str:
        parts = [self.soup_difference(name, comparison, a, start) for start in ("old", "new")]
        return " → ".join(f"{signed(m)} ± {sd:.1f} ({k}/5)" for m, sd, k, _, _ in parts)


def legend_handles(color: str = INK_SOFT) -> list[Line2D]:
    return [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=SURFACE if start == "old" else color,
                   markeredgecolor=color, markeredgewidth=1.5, label=START_NAMES[start]) for start in ("old", "new")]


def pair_of_points(axis, values: list[tuple[float, float]], y: float, color: str, marker: str = "o", size: float = 34) -> None:
    """Old (open) above, new (filled) below, each with its SD as a horizontal line."""
    for (mean, sd), offset, filled in zip(values, (0.17, -0.17), (False, True)):
        axis.errorbar(mean, y + offset, xerr=sd, color=color, linewidth=1.0, capsize=0, zorder=2)
        axis.scatter(mean, y + offset, s=size, marker=marker, color=color if filled else SURFACE, edgecolors=color,
                     linewidths=1.5, zorder=3)


def style(axis, xlabel: str, zero: bool = False) -> None:
    if zero:
        axis.axvline(0, color=INK_SOFT, linewidth=1.0, zorder=1)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlabel(xlabel, fontsize=8.5, color=INK_SOFT)


def figure_groups(data: Compare) -> str:
    figure, axes = new_figure(1, len(data.groups), (14.5, 4.6), sharey=True)
    positions = np.arange(len(BRANCHES))[::-1]
    for axis, group in zip(axes, data.groups):
        for y, variant in zip(positions, BRANCHES):
            values = [data.value(data.branches, group, variant, start) for start in ("old", "new")]
            pair_of_points(axis, values, y, COLORS[variant], "s" if variant in CONTROLS.values() else "o")
        style(axis, "Weighted F1 on target test (%)")
        axis.set_title(FAMILY_NAMES[group], fontsize=10.5, color=INK)
    axes[0].set_yticks(positions, [NAMES[v] for v in BRANCHES], fontsize=9.5, color=INK)
    figure.legend(handles=legend_handles(), frameon=False, fontsize=9, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    figure.suptitle("Stage 2, old and shared start: weighted F1 on target test, mean ± SD over five seeds "
                    "(methods: circles, controls: squares)", fontsize=12, color=INK, y=1.1)
    return save(figure, "fig13_shared_init_groups.png")


def figure_method_minus_control(data: Compare) -> str:
    rows = [(transfer_label(t), t) for t in data.transfers] + [("All 12", "all")]
    positions = list(range(len(rows), 0, -1))
    positions[-1] -= 0.5
    figure, axes = new_figure(1, len(CONTROLS), (13.5, 6.6), sharey=True)
    for axis, method in zip(axes, CONTROLS):
        for y, (_, scope) in zip(positions, rows):
            values = [data.method_difference(scope, method, "own_control", start)[:2] for start in ("old", "new")]
            pair_of_points(axis, values, y, METHOD_COLORS[method], "D" if scope == "all" else "o", 60 if scope == "all" else 34)
        style(axis, "Difference in weighted F1 (points)", zero=True)
        axis.set_title(f"{METHOD_NAMES[method]} − {CONTROL_NAMES[method]}", fontsize=10, color=INK)
    axes[0].set_yticks(positions, [label for label, _ in rows], fontsize=9, color=INK)
    figure.legend(handles=legend_handles(), frameon=False, fontsize=9, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    figure.suptitle("Each method minus its own control (target test), old and shared start, mean ± SD over five seeds",
                    fontsize=12, color=INK, y=1.06)
    return save(figure, "fig14_shared_init_method_minus_control.png")


def figure_soup(data: Compare) -> str:
    figure, axes = new_figure(2, len(data.groups), (14.5, 6.4), sharey=True, sharex="col")
    positions = np.arange(len(SOUP_KINDS))[::-1]
    for row, group in enumerate(("methods", "controls")):
        for axis, scope in zip(axes[row], data.groups):
            for y, kind in zip(positions, SOUP_KINDS):
                values = [data.value(data.soup, scope, key(kind, group), start) for start in ("old", "new")]
                pair_of_points(axis, values, y, SOUP_COLORS[kind])
            style(axis, "Weighted F1 on target test (%)" if row == 1 else "")
            if row == 0:
                axis.set_title(FAMILY_NAMES[scope], fontsize=10.5, color=INK)
        axes[row][0].set_yticks(positions, [SOUP_NAMES[k] for k in SOUP_KINDS], fontsize=9, color=INK)
        axes[row][0].set_ylabel(GROUP_NAMES[group].replace(" (", "\n("), fontsize=9, color=INK_SOFT)
    figure.legend(handles=legend_handles(), frameon=False, fontsize=9, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    figure.suptitle("Stage 3, old and shared start: soups, output ensemble and their references (target test), "
                    "mean ± SD over five seeds", fontsize=12, color=INK, y=1.07)
    return save(figure, "fig15_shared_init_soup.png")


def observations(data: Compare) -> list[str]:
    """Statements that follow directly from the T11 tables; every value is computed here."""
    lines = []
    branches = ", ".join(f"{NAMES[v]} {data.cell(data.branches, 'all', v)} %" for v in BRANCHES)
    lines.append(f"Stage 2, all 12 transfers, old → new: {branches}.")
    for start in ("old", "new"):
        sds = [data.value(data.branches, "all", v, start)[1] for v in BRANCHES[1:]]
        lines.append(f"SD over the seeds (all 12) of the six branches, {START_NAMES[start]}: "
                     f"{min(sds):.1f} to {max(sds):.1f} points, mean {np.mean(sds):.1f}.")
    lines.append("Each method minus its own control, all 12, old → new: "
                 + "; ".join(f"{METHOD_NAMES[m]} {data.method_cell('all', m, 'own_control')}" for m in CONTROLS) + ".")
    lines.append("Each method minus the baseline, all 12, old → new: "
                 + "; ".join(f"{METHOD_NAMES[m]} {data.method_cell('all', m, 'baseline')}" for m in CONTROLS) + ".")
    for group in ("methods", "controls"):
        values = ", ".join(f"{SOUP_NAMES[k]} {data.cell(data.soup, 'all', key(k, group))} %"
                           for k in ("baseline", "branch_mean", "ensemble", "bn_avg", "bn_target"))
        lines.append(f"{GROUP_NAMES[group]}, all 12 transfers, old → new: {values}.")
        for bn in ("bn_avg", "bn_target"):
            a = key(bn, group)
            parts = "; ".join(f"{label.split(' − ')[1]} {data.soup_cell('all', c, a)}"
                              for c, label in SOUP_COMPARISONS.items() if c in ("1_soup_minus_branch_mean", "3_soup_minus_baseline",
                                                                                 "5_soup_minus_ensemble"))
            pairs = [data.soup_difference("all", "3_soup_minus_baseline", a, start) for start in ("old", "new")]
            lines.append(f"{SOUP_NAMES[bn]} ({group}), all 12, old → new, soup minus {parts}; transfer-seed pairs above "
                         f"the baseline {pairs[0][3]}/{pairs[0][4]} → {pairs[1][3]}/{pairs[1][4]}.")
    for bn in ("bn_avg", "bn_target"):
        lines.append(f"Soup of the methods minus soup of the controls ({SOUP_NAMES[bn].split(', ')[1]}), all 12, old → new: "
                     f"{data.soup_cell('all', '4_soup_methods_minus_soup_controls', key(bn, 'methods'))}; the same for the means "
                     f"of the branches: {data.soup_cell('all', 'context_branch_mean_methods_minus_controls', 'branch_mean_methods')}.")
    reproduction = data.reproduction
    computed_again = reproduction[reproduction["variant"].str.startswith(("soup_", "ensemble_"))]
    lines.append(f"Seed 52, where both starts use the same branches: {int(reproduction['rows_identical'].sum())} of "
                 f"{int(reproduction['rows_transfer_x_split'].sum())} metric rows (transfer × split) of the baseline, the "
                 f"branches, the soups and the ensembles are identical to stage 3, among them all "
                 f"{int(computed_again['rows_identical'].sum())} of {int(computed_again['rows_transfer_x_split'].sum())} rows of "
                 f"the soups and ensembles computed again; largest difference "
                 f"{reproduction['max_abs_difference_any_metric'].max() * 100:.2f} points.")
    return lines


def previous_variant_section(first_figure: int = 13) -> list[str]:
    data = Compare()
    figures = [figure_groups(data), figure_method_minus_control(data), figure_soup(data)]
    names = [FAMILY_NAMES[g] for g in data.groups]
    page = [f"## {PREVIOUS_SECTION}", "",
            f"Until meeting 3, every seed started from its own source-only model (seed {SUPPLIED_SEED} from the supplied "
            "checkpoint, the other seeds from source-only models trained in this project with the supplied training code). "
            "Since meeting 3 all branches of all seeds start from the same source-only checkpoint, as the supervisor "
            "requested, so that the initialisation cannot shape the soup; the sections above therefore show the shared start, "
            "and the previous variant is kept here for comparison. Everything else is the same in both variants: AdaTime "
            "weights, 30 epochs, model after the last epoch, the same controls, the same stage-3 code and checks. The full "
            "tables of the previous variant are in `results/tables/` (stage 2: `T3_*.csv`, source-only continued: `T5_*.csv`, "
            "stage 3: `T8_*.csv`); the results page as it was before the switch is in commit `26279be`.", "",
            "### Old against new", "",
            f"*Old*: previous variant. *New*: shared start (the sections above). The seed-{SUPPLIED_SEED} runs are the same in "
            "both. Cells show old → new, mean ± sample SD over the five seeds in percent; a group value is first averaged over "
            "its transfers within each seed. Differences are paired by transfer and seed, in brackets the number of seeds in "
            "which the first variant is higher. Tables: `results/tables/T11_*.csv`.", ""]
    rows = [[NAMES[v], *(data.cell(data.branches, g, v) for g in data.groups)] for v in BRANCHES]
    page += ["**Stage 2: weighted F1 on target test (%), old → new**", "", markdown_table(["Variant", *names], rows)]
    rows = [[f"{METHOD_NAMES[m]} − {label}", *(data.method_cell(g, m, reference) for g in data.groups)]
            for m in CONTROLS for reference, label in (("own_control", "own control"), ("baseline", "baseline"))]
    page += ["**Stage 2: paired differences in weighted F1 on target test (points), old → new**", "",
             markdown_table(["Difference", *names], rows)]
    for group in ("methods", "controls"):
        rows = [[SOUP_NAMES[k], *(data.cell(data.soup, g, key(k, group)) for g in data.groups)] for k in SOUP_TABLE_KINDS]
        page += [f"**Stage 3: weighted F1 on target test (%), {GROUP_NAMES[group]}, old → new**", "",
                 markdown_table(["Variant", *names], rows)]
    rows = []
    for group in ("methods", "controls"):
        for bn in ("bn_avg", "bn_target"):
            for comparison, label in SOUP_COMPARISONS.items():
                rows.append([f"{'Methods' if group == 'methods' else 'Controls'}, {SOUP_NAMES[bn].split(', ')[1]}: {label}",
                             *(data.soup_cell(g, comparison, key(bn, group)) for g in data.groups)])
        rows.append([f"{'Methods' if group == 'methods' else 'Controls'}: BN re-estimated − BN averaged",
                     *(data.soup_cell(g, "6_bn_target_minus_bn_avg", key("bn_target", group)) for g in data.groups)])
    for bn in ("bn_avg", "bn_target"):
        rows.append([f"Soup of the methods − soup of the controls, {SOUP_NAMES[bn].split(', ')[1]}",
                     *(data.soup_cell(g, "4_soup_methods_minus_soup_controls", key(bn, "methods")) for g in data.groups)])
    page += ["**Stage 3: paired differences in weighted F1 on target test (points), old → new**", "",
             markdown_table(["Difference", *names], rows)]
    captions = [
        "Stage 2 per group of transfers: the baseline, the three methods (circles) and their controls (squares); open "
        "markers: old start, filled markers: shared start. Lines: SD over the seeds.",
        "Each method minus its own control per transfer, open: old start, filled: shared start; points right of zero mean "
        "the method is higher.",
        "Stage 3 per group of transfers: baseline, mean of the three branches, output ensemble and the two soups; top: soup "
        "of the methods, bottom: soup of the controls. Open: old start, filled: shared start.",
    ]
    for k, (path, caption) in enumerate(zip(figures, captions)):
        page += [f"**Figure {first_figure + k}.** {caption}", "", f"![Figure {first_figure + k}]({path})", ""]
    page += ["**Observations (old against new).** Only statements that follow directly from the T11 tables; no causes are "
             "claimed.", ""]
    page += [f"- {line}" for line in observations(data)] + [""]
    return page
