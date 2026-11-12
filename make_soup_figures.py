"""Figures and the results-page section for the uniform soup of the final checkpoints (stage 3, "at the end").

Built only from the T8 tables in results/tables/ (written by make_soup_tables.py); reads no raw data, no labels and
no checkpoints. make_figures.py calls soup_section() when these tables exist, so that project_docs/results.md keeps
a single generator. Figures:

    project_docs/figures/fig10_soup_by_group.png
    project_docs/figures/fig11_soup_differences_per_transfer.png
    project_docs/figures/fig12_soup_variants_per_transfer.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from make_figures import (GREY, GRID, INK, INK_SOFT, SURFACE, TABLES, markdown_table, new_figure, save, signed, tint,
                          transfer_label)

GROUP_NAMES = {"methods": "Soup of the methods (Deep CORAL + DANN + DSAN)",
               "controls": "Soup of the controls (their three λ = 0 runs)"}
FAMILY_NAMES = {"all": "All 12", "cross_dataset": "Cross-dataset (6)", "cross_gender": "Cross-gender (4)", "cross_age": "Cross-age (2)"}
TICK_NAMES = {"all": "All 12", "cross_dataset": "Cross-dataset\n(6)", "cross_gender": "Cross-gender\n(4)", "cross_age": "Cross-age\n(2)"}
# Slots 4 to 7 of the categorical palette; slots 1 to 3 are the colours of the three methods in figures 1 to 9.
COLORS = {"branch_mean": "#eda100", "ensemble": "#e87ba4", "bn_avg": "#6250d6", "bn_target": "#008300", "baseline": GREY}
NAMES = {"baseline": "Baseline", "branch_mean": "Mean of the 3 branches", "best_src_ce": "Best branch by SRC (CE)",
         "best_src_macro_f1": "Best branch by SRC (macro-F1)", "oracle": "Best branch by target test (oracle)",
         "ensemble": "Output ensemble", "bn_avg": "Soup, BN averaged", "bn_target": "Soup, BN re-estimated on target"}
COMPARISONS = {"1_soup_minus_branch_mean": "Soup − mean of the branches",
               "2a_soup_minus_best_src_ce": "Soup − best branch by SRC (CE)",
               "2b_soup_minus_best_src_macro_f1": "Soup − best branch by SRC (macro-F1)",
               "2c_soup_minus_oracle_best_branch": "Soup − best branch by target test (oracle, uses target labels)",
               "3_soup_minus_baseline": "Soup − baseline",
               "5_soup_minus_ensemble": "Soup − output ensemble"}


def key(kind: str, group: str) -> str:
    """Variant name in the T8 tables."""
    soup = f"soup_{group}"
    return {"baseline": "baseline", "bn_avg": f"{soup}_bn_avg", "bn_target": f"{soup}_bn_target",
            "ensemble": f"ensemble_{group}"}.get(kind, f"{kind}_{group}")


class Soup:
    """The T8 tables; scores in percent."""

    def __init__(self) -> None:
        self.summary = pd.read_csv(TABLES / "T8_summary_target_test.csv")
        self.differences = pd.read_csv(TABLES / "T8_differences_target_test.csv")
        self.selection = pd.read_csv(TABLES / "T8_selection.csv")
        self.costs = pd.read_csv(TABLES / "T8_costs_summary.csv").set_index("variant")
        self.commits = pd.read_csv(TABLES / "T8_commits.csv").set_index("outputs")
        self.criteria = pd.read_csv(TABLES / "T8_criteria.csv")
        self.transfers = list(dict.fromkeys(self.summary[self.summary["scope"] == "transfer"]["name"]))
        self.groups = ["all", "cross_dataset", "cross_gender", "cross_age"]

    def value(self, name: str, variant: str, column: str = "weighted_f1") -> tuple[float, float]:
        row = self.summary[(self.summary["name"] == name) & (self.summary["variant"] == variant)].iloc[0]
        return row[f"{column}_mean"] * 100, row[f"{column}_sd"] * 100

    def difference(self, name: str, comparison: str, a: str) -> pd.Series:
        rows = self.differences[(self.differences["name"] == name) & (self.differences["comparison"] == comparison)
                                & (self.differences["a"] == a)]
        return rows.iloc[0]

    def cell(self, name: str, variant: str) -> str:
        mean, sd = self.value(name, variant)
        return f"{mean:.1f} ± {sd:.1f}"

    def difference_cell(self, name: str, comparison: str, a: str, pairs: bool = False) -> str:
        d = self.difference(name, comparison, a)
        text = f"{signed(d['weighted_f1_diff_mean'] * 100)} ± {d['weighted_f1_diff_sd'] * 100:.1f} ({d['weighted_f1_seeds_a_higher']}/{d['n_seeds']})"
        return text + (f", {d['weighted_f1_pairs_a_higher']}/{d['n_pairs']} pairs" if pairs else "")


def figure_groups(data: Soup) -> str:
    kinds = ["baseline", "branch_mean", "ensemble", "bn_avg", "bn_target"]
    figure, axes = new_figure(1, 2, (13.0, 4.6), sharey=True)
    width = 0.8 / len(kinds)
    for axis, group in zip(axes, ("methods", "controls")):
        for i, kind in enumerate(kinds):
            for g, name in enumerate(data.groups):
                mean, sd = data.value(name, key(kind, group))
                x = g + (i - (len(kinds) - 1) / 2) * width
                style = {"facecolor": COLORS[kind], "edgecolor": SURFACE, "linewidth": 0.8}
                if group == "controls" and kind != "baseline":
                    style = {"facecolor": tint(COLORS[kind]), "edgecolor": COLORS[kind], "hatch": "////", "linewidth": 0.8}
                axis.bar(x, mean, width * 0.9, label=NAMES[kind] if g == 0 else None, **style)
                axis.errorbar(x, mean, yerr=sd, color=INK, linewidth=0.9, capsize=2)
                if kind in ("branch_mean", "bn_avg", "bn_target"):
                    axis.text(x, mean + sd + 1.0, f"{mean:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK_SOFT,
                              rotation=90)
        axis.set_xticks(range(len(data.groups)), [TICK_NAMES[n] for n in data.groups], fontsize=9.5, color=INK)
        axis.set_ylim(0, 100)
        axis.yaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_title(GROUP_NAMES[group], fontsize=10.5, color=INK)
    axes[0].set_ylabel("Weighted F1 on target test (%)", color=INK_SOFT, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=len(kinds), frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    figure.suptitle("Uniform soup of the final checkpoints: weighted F1 on target test, mean ± SD over five seeds",
                    fontsize=12, color=INK, y=1.09)
    return save(figure, "fig10_soup_by_group.png")


def dot_panels(data: Soup, columns: list[tuple[str, str, list[tuple]]], title: str, name: str, shared_legend: bool) -> str:
    """One panel per comparison; per transfer row the mean ± SD over the seeds of each (label, a) series."""
    rows = [(transfer_label(t), t) for t in data.transfers] + [("All 12", "all")]
    positions = list(range(len(rows), 0, -1))
    positions[-1] -= 0.5
    figure, axes = new_figure(1, len(columns), (3.3 * len(columns) + 1.6, 6.6), sharey=True)
    for axis, (comparison, heading, series) in zip(np.atleast_1d(axes), columns):
        n = len(series)
        for k, (label, a, color, filled) in enumerate(series):
            offset = (k - (n - 1) / 2) * 0.28
            for y, (_, scope) in zip(positions, rows):
                d = data.difference(scope, comparison, a)
                mean, sd = d["weighted_f1_diff_mean"] * 100, d["weighted_f1_diff_sd"] * 100
                axis.errorbar(mean, y + offset, xerr=sd, color=color, linewidth=1.0, capsize=0, zorder=2)
                axis.scatter(mean, y + offset, s=36 if scope != "all" else 60, marker="o" if scope != "all" else "D",
                             color=color if filled else SURFACE, edgecolors=color, linewidths=1.6, zorder=3,
                             label=label if y == positions[0] else None)
        axis.axvline(0, color=INK_SOFT, linewidth=1.0, zorder=1)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_title(heading, fontsize=9.5, color=INK, wrap=True)
        axis.set_xlabel("Difference in weighted F1 (points)", fontsize=8.5, color=INK_SOFT)
        if not shared_legend:
            axis.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(positions, [label for label, _ in rows], fontsize=9, color=INK)
    if shared_legend:
        handles, labels = first.get_legend_handles_labels()
        figure.legend(handles, labels, frameon=False, fontsize=9, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    figure.suptitle(title, fontsize=12, color=INK, y=1.07 if shared_legend else 0.99)
    return save(figure, name)


def figure_differences(data: Soup) -> str:
    series = [("Soup, BN averaged", key("bn_avg", "methods"), COLORS["bn_avg"], True),
              ("Soup, BN re-estimated on target", key("bn_target", "methods"), COLORS["bn_target"], True)]
    columns = [(c, COMPARISONS[c].replace(" (oracle, uses target labels)", "\n(oracle, uses target labels)"), series)
               for c in ("1_soup_minus_branch_mean", "2a_soup_minus_best_src_ce", "2b_soup_minus_best_src_macro_f1",
                         "2c_soup_minus_oracle_best_branch", "3_soup_minus_baseline")]
    return dot_panels(data, columns, "Soup of the three methods against its branches and the baseline (target test), "
                                     "mean ± SD over five seeds", "fig11_soup_differences_per_transfer.png", True)


def figure_variants(data: Soup) -> str:
    columns = [
        ("4_soup_methods_minus_soup_controls", "Soup of the methods − soup of the controls",
         [("BN averaged", key("bn_avg", "methods"), COLORS["bn_avg"], True),
          ("BN re-estimated on target", key("bn_target", "methods"), COLORS["bn_target"], True)]),
        ("6_bn_target_minus_bn_avg", "BN re-estimated − BN averaged",
         [("soup of the methods", key("bn_target", "methods"), INK_SOFT, True),
          ("soup of the controls", key("bn_target", "controls"), INK_SOFT, False)]),
        ("5_soup_minus_ensemble", "Soup of the methods − output ensemble",
         [("BN averaged", key("bn_avg", "methods"), COLORS["bn_avg"], True),
          ("BN re-estimated on target", key("bn_target", "methods"), COLORS["bn_target"], True)]),
    ]
    return dot_panels(data, columns, "Alignment, BatchNorm statistics and weight vs. output averaging (target test), "
                                     "mean ± SD over five seeds", "fig12_soup_variants_per_transfer.png", False)


def observations(data: Soup) -> list[str]:
    """Statements that follow directly from the T8 tables; every value is computed here."""
    lines = []
    for group in ("methods", "controls"):
        values = ", ".join(f"{NAMES[k]} {data.cell('all', key(k, group))} %"
                           for k in ("baseline", "branch_mean", "best_src_ce", "best_src_macro_f1", "oracle", "ensemble",
                                     "bn_avg", "bn_target"))
        lines.append(f"{GROUP_NAMES[group]}, all 12 transfers: {values}.")
    for bn in ("bn_avg", "bn_target"):
        a = key(bn, "methods")
        parts = "; ".join(f"{COMPARISONS[c].split(' − ')[1]} {data.difference_cell('all', c, a, pairs=True)}" for c in COMPARISONS)
        lines.append(f"{NAMES[bn]} (methods), differences over all transfers: {parts}.")
        groups = "; ".join(f"{FAMILY_NAMES[g]} {data.difference_cell(g, '1_soup_minus_branch_mean', a)}" for g in data.groups[1:])
        lines.append(f"{NAMES[bn]} (methods) minus the mean of its branches by group: {groups}.")
    for group in ("methods", "controls"):
        lines.append(f"BN re-estimated minus BN averaged, {group}: "
                     f"{data.difference_cell('all', '6_bn_target_minus_bn_avg', key('bn_target', group), pairs=True)}; "
                     f"source test weighted F1 {data.value('all', key('bn_avg', group), 'source_test_weighted_f1')[0]:.1f} % "
                     f"(averaged) and {data.value('all', key('bn_target', group), 'source_test_weighted_f1')[0]:.1f} % "
                     f"(re-estimated), mean of the branches {data.value('all', key('branch_mean', group), 'source_test_weighted_f1')[0]:.1f} %.")
    src = data.criteria.groupby("variant")[["src_val_ce", "src_val_macro_f1"]].mean()
    for group in ("methods", "controls"):
        branches = [v for v in src.index if v in set(data.selection[data.selection["group"] == group]["chosen_branch"])]
        lines.append(f"Without target labels, on source validation (SRC, mean over transfers and seeds), {group}: macro-F1 of the "
                     f"branches {src.loc[branches, 'src_val_macro_f1'].min() * 100:.1f} to {src.loc[branches, 'src_val_macro_f1'].max() * 100:.1f} %, "
                     f"soup {src.loc[key('bn_avg', group), 'src_val_macro_f1'] * 100:.1f} % (BN averaged) and "
                     f"{src.loc[key('bn_target', group), 'src_val_macro_f1'] * 100:.1f} % (re-estimated), ensemble "
                     f"{src.loc[key('ensemble', group), 'src_val_macro_f1'] * 100:.1f} %; cross-entropy of the branches "
                     f"{src.loc[branches, 'src_val_ce'].min():.2f} to {src.loc[branches, 'src_val_ce'].max():.2f}, soup "
                     f"{src.loc[key('bn_avg', group), 'src_val_ce']:.2f} and {src.loc[key('bn_target', group), 'src_val_ce']:.2f}.")
    for bn in ("bn_avg", "bn_target"):
        lines.append(f"Soup of the methods minus soup of the controls ({NAMES[bn].split(', ')[1]}): "
                     f"{data.difference_cell('all', '4_soup_methods_minus_soup_controls', key(bn, 'methods'), pairs=True)}; "
                     "the same difference between the means of the branches: "
                     f"{data.difference_cell('all', 'context_branch_mean_methods_minus_controls', 'branch_mean_methods', pairs=True)}.")
    above = {}
    for bn in ("bn_avg", "bn_target"):
        above[bn] = [t for t in data.transfers
                     if data.difference(t, "2c_soup_minus_oracle_best_branch", key(bn, "methods"))["weighted_f1_diff_mean"] > 0]
    lines.append("Transfers on which the soup of the methods is above the best branch by target test (oracle) in the mean "
                 "over seeds: " + "; ".join(f"{NAMES[bn]}: {', '.join(transfer_label(t) for t in above[bn]) or 'none'}"
                                           for bn in above) + ".")
    picks = data.selection[data.selection["group"] == "methods"].groupby("rule")["chosen_branch"].value_counts().unstack(fill_value=0)
    lines.append("Branch chosen per transfer and seed (60 cases) among Deep CORAL / DANN / DSAN: "
                 + "; ".join(f"{NAMES[r] if r in NAMES else r}: " + ", ".join(f"{b} {picks.loc[r].get(b, 0)}" for b in ("deep_coral", "dann", "dsan"))
                             for r in ("best_src_ce", "best_src_macro_f1", "oracle")) + ".")
    return lines


def soup_section(first_figure: int = 10) -> list[str]:
    data = Soup()
    figures = [figure_groups(data), figure_differences(data), figure_variants(data)]
    commit = data.commits.loc["soup", "repository_commit"][:7]
    page = ["## Stage 3: uniform soup of the final checkpoints", "",
            "**Setup.** For every transfer and seed, the three branches of stage 2 (Deep CORAL, DANN, DSAN with the AdaTime "
            "weights, last epoch) are averaged with equal weight 1/3 into one model; the same is done with their three control "
            "runs. All six runs of a seed start from the same source-only checkpoint, which is checked before averaging. "
            "Only the HAR predictor is averaged; the DANN discriminator is not part of the soup. The averaging step is the soup "
            "script of the project partner (`create_method_soup.py`). Two BatchNorm variants: *averaged* keeps the averaged "
            "running statistics; *re-estimated on target* recomputes them after averaging in one pass over the unlabelled "
            "target-train windows (cumulative average over batches of 32, weights unchanged). The output ensemble averages the "
            "softmax outputs of the same three models and needs no new weights. No target labels are used for any of this, "
            f"and no training takes place. Code commit of all soups and ensembles: `{commit}`.", "",
            "**Best branch.** Without target labels, the branch with the lowest cross-entropy or the highest macro-F1 on "
            "source validation (SRC) is chosen per transfer and seed. The oracle takes the branch with the highest weighted "
            "F1 on target test; **it uses target labels and is shown only as a reference, not as a selection rule.** The "
            "source-validation predictions of the branches were computed afterwards from their saved checkpoints "
            "(prediction only).", "",
            "Values are mean ± sample SD over the five seeds in percent; a group value is first averaged over its transfers "
            "within each seed. Differences are paired by transfer and seed; in brackets the number of seeds in which the first "
            "variant is higher, and for all transfers also the number of the 60 transfer-seed pairs.", ""]
    names = [FAMILY_NAMES[g] for g in data.groups]
    for group in ("methods", "controls"):
        rows = [[NAMES[k], *(data.cell(g, key(k, group)) for g in data.groups)]
                for k in ("baseline", "branch_mean", "best_src_ce", "best_src_macro_f1", "oracle", "ensemble", "bn_avg", "bn_target")]
        page += [f"**Weighted F1 on target test (%): {GROUP_NAMES[group]}**", "", markdown_table(["Variant", *names], rows)]
    rows = []
    for group in ("methods", "controls"):
        for bn in ("bn_avg", "bn_target"):
            for comparison, label in COMPARISONS.items():
                rows.append([f"{'Methods' if group == 'methods' else 'Controls'}, {NAMES[bn].split(', ')[1]}: {label}",
                             *(data.difference_cell(g, comparison, key(bn, group)) for g in data.groups)])
        rows.append([f"{'Methods' if group == 'methods' else 'Controls'}: BN re-estimated − BN averaged",
                     *(data.difference_cell(g, "6_bn_target_minus_bn_avg", key("bn_target", group)) for g in data.groups)])
    for bn in ("bn_avg", "bn_target"):
        rows.append([f"Soup of the methods − soup of the controls, {NAMES[bn].split(', ')[1]}",
                     *(data.difference_cell(g, "4_soup_methods_minus_soup_controls", key(bn, "methods")) for g in data.groups)])
    rows.append(["Context: mean of the method branches − mean of the control branches",
                 *(data.difference_cell(g, "context_branch_mean_methods_minus_controls", "branch_mean_methods") for g in data.groups)])
    page += ["**Paired differences in weighted F1 on target test (points)**", "", markdown_table(["Difference", *names], rows)]

    costs = data.costs
    soup_rows = [v for v in costs.index if v.startswith("soup_")]
    page += ["**Costs.** "
             f"Averaging three checkpoints takes {costs.loc[soup_rows, 'total_seconds_average'].sum() / costs.loc[soup_rows, 'outputs'].sum() * 1000:.0f} ms "
             f"per soup; re-estimating the BatchNorm statistics takes {costs.loc[[v for v in soup_rows if v.endswith('bn_target')], 'total_seconds_bn_recompute'].sum() / costs.loc[[v for v in soup_rows if v.endswith('bn_target')], 'outputs'].sum():.1f} s "
             "per soup (one pass over target train in batches of 32 on MPS). Training the three branches of one soup took "
             f"{costs.loc[key('bn_avg', 'methods'), 'total_minutes_training_of_ingredients'] / costs.loc[key('bn_avg', 'methods'), 'outputs'] * 60:.0f} s "
             "on average. A soup is evaluated at the cost of one model, the ensemble at the cost of three. Details: "
             "`results/tables/T8_costs.csv`, `T8_costs_summary.csv`.", ""]
    captions = [
        "Weighted F1 on target test per group: the baseline, the mean of the three branches, the output ensemble and the two "
        "soups; left the soup of the methods, right the soup of the control runs (hatched). Error bars: SD over the seeds.",
        "Soup of the three methods minus its references per transfer, for both BatchNorm variants; points right of zero mean "
        "the soup is higher. The fourth panel uses target labels to choose the branch and is a reference only.",
        "Left: soup of the methods minus soup of the controls (what the alignment losses contribute inside the soup). "
        "Middle: re-estimated minus averaged BatchNorm statistics (filled: methods, open: controls). Right: soup minus "
        "output ensemble of the same three models.",
    ]
    for k, (path, caption) in enumerate(zip(figures, captions)):
        page += [f"**Figure {first_figure + k}.** {caption}", "", f"![Figure {first_figure + k}]({path})", ""]
    page += ["**Observations (soup).** Only statements that follow directly from the T8 tables; no causes are claimed.", ""]
    page += [f"- {line}" for line in observations(data)] + [""]
    return page
