"""Figures and the results-page section for the alignment-weight sensitivity (paper convention, seed 52).

Built only from the T7 tables in results/tables/ (written by make_sensitivity_tables.py); reads no raw data,
no labels and no checkpoints. make_figures.py calls sensitivity_section() when these tables exist, so that
project_docs/results.md keeps a single generator. Figures:

    project_docs/figures/fig7_lambda_sensitivity.png
    project_docs/figures/fig8_paper_vs_adatime.png
    project_docs/figures/fig9_lambda_per_transfer.png

Target values are shown as a sensitivity curve only; the marked choices come from the label-free criteria.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from make_figures import (AXIS, GREY, GRID, INK, INK_SOFT, METHOD_COLORS, METHOD_NAMES, SURFACE, TABLES, markdown_table,
                          new_figure, save, signed, transfer_label)

SEED = 52
FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
CONTROL_X = -3.2  # position of the shared control left of log2(0.25) = -2, with a gap for the axis break
# Label-free criteria shown in the figures and the page, with marker and label (DEV is not used, see make_dev_check.py).
CRITERION_STYLE = {"src_ce": ("v", "SRC (source-val cross-entropy)"), "src_macro_f1": ("^", "SRC (source-val macro-F1)")}
ADATIME_CONTROL = {"deep_coral": "lambda0_target_pass", "dann": "dann_lambda0_target_pass", "dsan": "dsan_lambda0_target_pass"}


def position(factor: float) -> float:
    return CONTROL_X if factor == 0 else float(np.log2(factor))


def factor_label(factor: float) -> str:
    return "0 (control)" if factor == 0 else f"{factor:g}"


class Sensitivity:
    """The T7 tables, in percent where they hold scores."""

    def __init__(self) -> None:
        self.summary = pd.read_csv(TABLES / "T7_sensitivity_summary.csv")
        self.selection = pd.read_csv(TABLES / "T7_selection.csv")
        self.differences = pd.read_csv(TABLES / "T7_adatime_vs_paper.csv")
        self.metrics = pd.read_csv(TABLES / "T7_metrics_all_splits.csv")
        self.commits = pd.read_csv(TABLES / "T7_commits.csv")
        self.dev_check = pd.read_csv(TABLES / "T7_dev_check_summary.csv").set_index("check")["value"]
        self.methods = [m for m in METHOD_NAMES if m in set(self.summary["method"])]
        self.criteria = [c for c in CRITERION_STYLE if c in set(self.selection["criterion"])]
        earlier = pd.read_csv(TABLES / "T3_runs_all_splits.csv")
        rows = earlier[(earlier["seed"] == SEED) & (earlier["variant"] == "baseline") & (earlier["split"] == "target_test")]
        self.baseline = rows["weighted_f1"].mean() * 100
        self.n_transfers = int(self.summary["n_transfers"].max())
        self.transfers = list(dict.fromkeys(self.metrics["transfer"]))

    def row(self, variant: str) -> pd.Series:
        return self.summary.set_index("variant").loc[variant]

    def curve(self, method: str, column: str) -> pd.DataFrame:
        """Mean and sd over the transfers of one column for the shared control and the five factors."""
        variants = ["paper_control", *(f"paper_{method}_x{f:g}" for f in FACTORS)]
        table = self.summary.set_index("variant").loc[variants, [f"{column}_mean", f"{column}_sd", "factor"]]
        return table.rename(columns={f"{column}_mean": "mean", f"{column}_sd": "sd"})

    def chosen(self, method: str, criterion: str, candidates: str = "factors") -> float:
        rows = self.selection[(self.selection["method"] == method) & (self.selection["criterion"] == criterion)
                              & (self.selection["candidates"] == candidates) & (self.selection["scope"] == "mean_over_transfers")]
        return float(rows["chosen_factor"].iloc[0])

    def per_transfer_choice(self, method: str, criterion: str, candidates: str = "factors") -> pd.Series:
        rows = self.selection[(self.selection["method"] == method) & (self.selection["criterion"] == criterion)
                              & (self.selection["candidates"] == candidates) & (self.selection["scope"] == "per_transfer")]
        return rows.set_index("transfer")["chosen_factor"]

    def per_transfer_mean(self, method: str, criterion: str, candidates: str = "factors") -> pd.Series:
        rows = self.selection[(self.selection["method"] == method) & (self.selection["criterion"] == criterion)
                              & (self.selection["candidates"] == candidates)
                              & (self.selection["scope"] == "per_transfer_choice_mean")]
        return rows.iloc[0]

    def target(self, variant: str) -> pd.Series:
        """Weighted F1 on target test per transfer, in percent."""
        rows = self.metrics[(self.metrics["variant"] == variant) & (self.metrics["split"] == "target_test")]
        return rows.set_index("transfer")["weighted_f1"].reindex(self.transfers) * 100

    def difference(self, comparison: str, method: str, split: str = "target_test", metric: str = "weighted_f1") -> pd.Series:
        rows = self.differences[(self.differences["comparison"] == comparison) & (self.differences["method"] == method)
                                & (self.differences["split"] == split) & (self.differences["metric"] == metric)]
        return rows.iloc[0]


# ---------------------------------------------------------------- figures

def style_factor_axis(axis, show_labels: bool) -> None:
    ticks = [CONTROL_X, *(position(f) for f in FACTORS)]
    axis.set_xticks(ticks, [factor_label(0), *(factor_label(f) for f in FACTORS)] if show_labels else [""] * len(ticks),
                    fontsize=8, color=INK)
    axis.set_xlim(CONTROL_X - 0.6, 2.5)
    axis.axvspan(CONTROL_X + 0.35, -2.35, color=SURFACE, zorder=0)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for x in (CONTROL_X + 0.45, -2.45):  # axis break between the control and the log-scaled factors
        axis.plot([x - 0.06, x + 0.06], [-0.012, 0.012], transform=axis.get_xaxis_transform(), color=AXIS,
                  linewidth=1.0, clip_on=False)


def draw_curve(axis, data: Sensitivity, method: str, column: str, scale: float, marks: bool) -> None:
    color = METHOD_COLORS[method]
    curve = data.curve(method, column)
    x = [position(f) for f in curve["factor"]]
    y = curve["mean"].to_numpy() * scale
    axis.plot(x[1:], y[1:], color=color, linewidth=1.8, marker="o", markersize=5, zorder=3)
    axis.plot(x[:2], y[:2], color=color, linewidth=1.0, linestyle=(0, (2, 2)), zorder=2)
    axis.scatter(x[0], y[0], s=34, color=SURFACE, edgecolors=color, linewidths=1.6, zorder=4)
    adatime = data.row(method)[f"{column}_mean"] * scale
    control = data.row(ADATIME_CONTROL[method])[f"{column}_mean"] * scale
    axis.scatter([position(1.0), CONTROL_X], [adatime, control], s=46, marker="D", color=SURFACE, edgecolors=INK_SOFT,
                 linewidths=1.2, zorder=4)
    if marks:
        for k, criterion in enumerate(data.criteria):
            factor = data.chosen(method, criterion)
            value = curve.set_index("factor").loc[factor, "mean"] * scale
            marker, _ = CRITERION_STYLE[criterion]
            offset = (k - (len(data.criteria) - 1) / 2) * 0.16
            axis.scatter(position(factor) + offset, value, s=70, marker=marker, color=INK, edgecolors=SURFACE,
                         linewidths=0.8, zorder=5)


def figure_sensitivity(data: Sensitivity) -> str:
    rows = [("target_test_weighted_f1", 100, "Weighted F1, target test (%)", True),
            ("src_val_ce", 1, "Cross-entropy, source val\n(SRC, lower is better)", False),
            ("src_val_macro_f1", 100, "Macro-F1, source val (%)\n(SRC, higher is better)", False)]
    # One y scale per row, so that a change of a few points is not drawn as large as the collapse of a method.
    figure, axes = new_figure(len(rows), len(data.methods), (4.1 * len(data.methods) + 0.8, 2.7 * len(rows) + 0.6),
                              sharex=True, sharey="row", squeeze=False)
    for j, method in enumerate(data.methods):
        for i, (column, scale, label, marks) in enumerate(rows):
            axis = axes[i, j]
            draw_curve(axis, data, method, column, scale, marks)
            if i == 0:
                axis.axhline(data.baseline, color=GREY, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
                base = data.row(f"paper_{method}_x1")["alignment_weight"]
                axis.set_title(f"{METHOD_NAMES[method]}  (base λ = {base:.4g})", fontsize=11, color=INK)
            if j == 0:
                axis.set_ylabel(label, fontsize=9, color=INK_SOFT)
            style_factor_axis(axis, show_labels=i == len(rows) - 1)
        axes[-1, j].set_xlabel("Factor × base λ (log scale)", fontsize=9, color=INK_SOFT)
    first = axes[0, 0]
    first.plot([], [], color=GREY, linewidth=1.8, marker="o", markersize=5, label="paper convention (source weight 1)")
    first.scatter([], [], s=34, color=SURFACE, edgecolors=GREY, linewidths=1.6, label="shared control (λ = 0)")
    first.scatter([], [], s=46, marker="D", color=SURFACE, edgecolors=INK_SOFT, linewidths=1.2,
                  label="AdaTime weights (method / own control)")
    first.plot([], [], color=GREY, linewidth=1.0, linestyle=(0, (4, 3)), label="baseline (source-only)")
    for criterion in data.criteria:
        marker, label = CRITERION_STYLE[criterion]
        first.scatter([], [], s=70, marker=marker, color=INK, label=f"choice of {label}")
    figure.legend(*first.get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper center", ncol=4,
                  bbox_to_anchor=(0.5, 1.0 + 0.012 * len(rows)))
    figure.suptitle(f"Sensitivity to the alignment weight, paper convention, seed {SEED}: mean over "
                    f"{data.n_transfers} transfers", fontsize=12, color=INK, y=1.03 + 0.015 * len(rows))
    figure.tight_layout()
    return save(figure, "fig7_lambda_sensitivity.png")


def figure_adatime(data: Sensitivity) -> str:
    """Paper convention x1 minus AdaTime weights per transfer; the control panel has one point per AdaTime control."""
    panels = [(METHOD_NAMES[m], [("paper_x1_minus_adatime", m, METHOD_COLORS[m], "o")]) for m in data.methods]
    panels.append(("Shared control − AdaTime controls",
                   [("paper_control_minus_adatime_control", m, METHOD_COLORS[m], "o") for m in data.methods]))
    rows = [*(transfer_label(t) for t in data.transfers), f"All {data.n_transfers} (mean ± SD)"]
    positions = list(range(len(rows), 0, -1))
    positions[-1] -= 0.5
    figure, axes = new_figure(1, len(panels), (3.5 * len(panels) + 1.6, 6.0), sharey=True, sharex=True)
    for axis, (title, series) in zip(np.atleast_1d(axes), panels):
        for k, (comparison, method, color, marker) in enumerate(series):
            row = data.difference(comparison, method)
            a, b = data.target(row["variant_a"]), data.target(row["variant_b"])
            difference = (a - b).reindex(data.transfers)
            offset = (k - (len(series) - 1) / 2) * 0.22
            axis.scatter(difference, [p + offset for p in positions[:-1]], s=24, color=color, marker=marker, zorder=3,
                         label=METHOD_NAMES[method] if len(series) > 1 else None)
            axis.errorbar(difference.mean(), positions[-1] + offset, xerr=difference.std(ddof=1), color=color,
                          linewidth=1.0, capsize=2, zorder=2)
            axis.scatter(difference.mean(), positions[-1] + offset, s=60, marker="D", color=color, edgecolors=INK,
                         linewidths=0.9, zorder=3)
        axis.axvline(0, color=INK_SOFT, linewidth=1.0, zorder=1)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_title(title if len(series) > 1 else f"{title}: paper ×1 − AdaTime", fontsize=10, color=INK)
        axis.set_xlabel("Weighted F1, target test (points)", fontsize=9, color=INK_SOFT)
    last = np.atleast_1d(axes)[-1]
    last.legend(*last.get_legend_handles_labels(), frameon=False, fontsize=9, loc="lower center", ncol=3,
                bbox_to_anchor=(0.5, 1.06), title="AdaTime control of", title_fontsize=9)
    np.atleast_1d(axes)[0].set_yticks(positions, rows, fontsize=9, color=INK)
    figure.suptitle(f"Source weight 1 instead of the AdaTime source weights, same λ ratio, seed {SEED}", fontsize=12,
                    color=INK, y=1.08)
    return save(figure, "fig8_paper_vs_adatime.png")


def figure_per_transfer(data: Sensitivity) -> str:
    """Transfers x factors: weighted F1 on target test minus the shared control, with the per-transfer SRC choice framed."""
    control = data.target("paper_control")
    values = {m: np.array([(data.target(f"paper_{m}_x{f:g}") - control).to_numpy() for f in FACTORS]).T for m in data.methods}
    limit = max(float(np.nanmax(np.abs(v))) for v in values.values())
    colormap = LinearSegmentedColormap.from_list("diverging", ["#e34948", "#f0efec", "#2a78d6"])
    figure, axes = new_figure(1, len(data.methods), (3.3 * len(data.methods) + 2.6, 0.42 * len(data.transfers) + 1.8),
                              sharey=True)
    for axis, method in zip(np.atleast_1d(axes), data.methods):
        image = axis.imshow(values[method], cmap=colormap, vmin=-limit, vmax=limit, aspect="auto")
        for i in range(len(data.transfers)):
            for j in range(len(FACTORS)):
                value = values[method][i, j]
                axis.text(j, i, signed(value), ha="center", va="center", fontsize=8,
                          color="#ffffff" if abs(value) > 0.6 * limit else INK)
        choice = data.per_transfer_choice(method, "src_ce")
        for i, transfer in enumerate(data.transfers):
            j = FACTORS.index(float(choice[transfer]))
            axis.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.8))
        axis.set_xticks(range(len(FACTORS)), [f"×{f:g}" for f in FACTORS], fontsize=9, color=INK)
        axis.xaxis.tick_top()
        axis.set_title(METHOD_NAMES[method], fontsize=11, color=INK, pad=22)
        for side in axis.spines.values():
            side.set_visible(False)
    np.atleast_1d(axes)[0].set_yticks(range(len(data.transfers)), [transfer_label(t) for t in data.transfers], fontsize=9,
                                      color=INK)
    bar = figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label("Weighted F1, target test: factor − shared control (points)", fontsize=9, color=INK_SOFT)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, colors=INK_SOFT, length=0)
    figure.suptitle(f"Per transfer, seed {SEED}; framed: factor chosen by SRC (source-val cross-entropy) for that transfer",
                    fontsize=12, color=INK, y=1.02)
    return save(figure, "fig9_lambda_per_transfer.png")


# ---------------------------------------------------------------- results page

def pct(mean: float, sd: float) -> str:
    return f"{mean * 100:.1f} ± {sd * 100:.1f}"


def dev_paragraph(check: pd.Series) -> str:
    """Why DEV is not used, with the numbers of T7_dev_check_summary.csv."""
    value = lambda key: float(check[key])
    return ("**DEV is not used.** Deep embedded validation was ported from AdaTime, but its importance weights are not sound "
            f"on our splits. Over the {value('runs'):.0f} runs with the AdaTime weights, the domain classifier separates "
            f"source-train from target-train features with a median hold-out accuracy of "
            f"{value('median domain classifier hold-out accuracy (seed 52)') * 100:.1f} %; the mean importance weight on source "
            f"validation, which should be close to 1, lies between {value('mean importance weight, min over runs (seed 52)'):.2f} "
            f"and {value('mean importance weight, max over runs (seed 52)'):.0f} (outside [0.5, 2] in "
            f"{value('mean importance weight outside [0.5, 2] (seed 52)'):.0f} runs); DEV is negative for at least one of three "
            f"classifier seeds in {value('runs with a negative DEV for at least one seed'):.0f} runs, although the cross-entropy "
            f"cannot be negative; and in {value('pairs whose DEV preference (method or control) changes with the seed'):.0f} of "
            f"{value('method vs own control pairs'):.0f} pairs of a method and its control, DEV prefers a different run depending "
            "only on the seed of the domain classifier. Only SRC is therefore reported (details in `T7_dev_check.csv`).")


def observations(data: Sensitivity) -> list[str]:
    """Statements that follow directly from the T7 tables; every value is computed here."""
    lines = []
    control = data.row("paper_control")["target_test_weighted_f1_mean"] * 100
    for method in data.methods:
        curve = data.curve(method, "target_test_weighted_f1").iloc[1:]["mean"] * 100
        lines.append(f"{METHOD_NAMES[method]}: weighted F1 on target test lies between {curve.min():.1f} and {curve.max():.1f} % "
                     f"over the five factors (shared control {control:.1f} %); {int((curve > control).sum())} of 5 factors are "
                     "above the control.")
    source = data.metrics[(data.metrics["split"] == "source_test")]
    for method in data.methods:
        for factor in FACTORS:
            rows = source[source["variant"] == f"paper_{method}_x{factor:g}"]
            collapsed = int((rows["weighted_f1"] < 0.2).sum())
            if collapsed:
                lines.append(f"{METHOD_NAMES[method]} ×{factor:g}: in {collapsed} of {len(rows)} transfers weighted F1 on "
                             f"source test is below 20 % (the model collapses on the labelled source as well); mean SRC "
                             f"macro-F1 on source val {data.row(f'paper_{method}_x{factor:g}')['src_val_macro_f1_mean'] * 100:.1f} %.")
    choices = "; ".join(f"{METHOD_NAMES[m]} " + ", ".join(f"×{data.chosen(m, c):g} ({CRITERION_STYLE[c][1].split('(')[1][:-1]})"
                                                          for c in data.criteria) for m in data.methods)
    lines.append(f"Choices over the mean: {choices}. The two SRC variants pick the same factor for "
                 f"{sum(len({data.chosen(m, c) for c in data.criteria}) == 1 for m in data.methods)} of {len(data.methods)} methods.")
    zero = [f"{METHOD_NAMES[m]} with {CRITERION_STYLE[c][1]}" for m in data.methods for c in data.criteria
            if data.chosen(m, c, "factors_and_control") == 0]
    if zero:
        lines.append("If λ = 0 is a candidate, SRC prefers the control over every factor for " + "; ".join(zero) + ".")
    parts = []
    for method in data.methods:
        d = data.difference("paper_x1_minus_adatime", method)
        parts.append(f"{METHOD_NAMES[method]} {signed(d['diff_mean_points'])} ± {d['diff_sd_points']:.1f} "
                     f"({d['transfers_a_higher']}/{d['n_transfers']})")
    lines.append("Paper convention ×1 minus AdaTime weights on target test: " + ", ".join(parts) + " points.")
    return lines


def sensitivity_section(first_figure: int = 7) -> list[str]:
    """Figures 7 to 9 and the markdown lines of the section; every number comes from the T7 tables."""
    data = Sensitivity()
    figures = [figure_sensitivity(data), figure_adatime(data), figure_per_transfer(data)]
    n = data.n_transfers
    commit = data.commits[data.commits["convention"] == "paper"]["repository_commit"].iloc[0][:7]
    page = ["## Sensitivity to the alignment weight (paper convention, seed 52)", "",
            f"**Only seed 52 was run, and target labels were not used for any choice in this section.** Target values are "
            f"reported as a sensitivity curve only; the marked choices come from label-free criteria computed on the source "
            f"validation split. This is a sensitivity analysis, not the final choice of the alignment weight. Values are "
            f"mean ± sample SD over the {n} transfers (not over seeds, unlike the sections above).", "",
            "**Setup.** All three methods are run in the convention of their papers: loss = 1 · source cross-entropy + "
            "λ · alignment loss. The base λ of a method is the ratio of AdaTime's alignment weight to its source-loss weight "
            "for HAR, so that factor 1 has the same ratio as the runs above with the AdaTime weights. With Adam, scaling the "
            "whole loss by the source weight c changes nothing except that the weight decay acts like 1e-4 / c; factor 1 "
            "therefore differs from the AdaTime runs only in the effective weight decay. Everything else is unchanged "
            "(start model, optimiser, budget, forward passes, DANN reversal factor and discriminator, constant DSAN weight). "
            "One control run (λ = 0, source weight 1, target forward pass) serves all three methods: with these weights the "
            f"three training loops give bitwise the same model. Code commit of all runs: `{commit}`.", "",
            dev_paragraph(data.dev_check), ""]
    rows = []
    for method in data.methods:
        x1 = data.row(f"paper_{method}_x1")
        adatime = data.row(method)
        rows.append([METHOD_NAMES[method], f"{adatime['alignment_weight']:.4f}", f"{adatime['source_weight']:.4f}",
                     f"{x1['alignment_weight']:.4f}", f"{1e-4 / adatime['source_weight']:.2e}"])
    page += [markdown_table(["Method", "AdaTime alignment weight", "AdaTime source weight", "Base λ (ratio)",
                             "Effective weight decay of the AdaTime runs"], rows)]

    page += [f"**Weighted F1 on target test (%)**, mean ± SD over {n} transfers. AdaTime: the runs above with the "
             "AdaTime weights (seed 52); in the control column their own control.", ""]
    control = data.row("paper_control")
    rows = []
    for method in data.methods:
        cells = [pct(data.row(f"paper_{method}_x{f:g}")["target_test_weighted_f1_mean"],
                     data.row(f"paper_{method}_x{f:g}")["target_test_weighted_f1_sd"]) for f in FACTORS]
        rows.append([METHOD_NAMES[method], pct(control["target_test_weighted_f1_mean"], control["target_test_weighted_f1_sd"]),
                     *cells, pct(data.row(method)["target_test_weighted_f1_mean"], data.row(method)["target_test_weighted_f1_sd"]),
                     pct(data.row(ADATIME_CONTROL[method])["target_test_weighted_f1_mean"],
                         data.row(ADATIME_CONTROL[method])["target_test_weighted_f1_sd"])])
    page += [markdown_table(["Method", "Control (λ = 0)", *(f"×{f:g}" for f in FACTORS), "AdaTime weights",
                             "AdaTime control"], rows)]
    page += [f"Baseline (source-only model, seed 52): {data.baseline:.1f} % weighted F1 on target test.", ""]

    page += ["**Choices of the label-free criteria.** Over the mean: the factor with the best mean criterion over the "
             "transfers. Per transfer: each transfer takes its own best factor; the target column is then the mean over the "
             "transfers. Candidates are the five factors; in brackets the choice when the control (λ = 0) is a candidate too.", ""]
    rows = []
    for method in data.methods:
        for criterion in data.criteria:
            factor = data.chosen(method, criterion)
            with_control = data.chosen(method, criterion, "factors_and_control")
            mean_target = data.row(f"paper_{method}_x{factor:g}")["target_test_weighted_f1_mean"] * 100
            per = data.per_transfer_mean(method, criterion)
            per_control = data.per_transfer_mean(method, criterion, "factors_and_control")
            rows.append([METHOD_NAMES[method], CRITERION_STYLE[criterion][1], f"×{factor:g} (×{with_control:g})",
                         f"{mean_target:.1f}", per["chosen_factor_counts"],
                         f"{per['target_test_weighted_f1'] * 100:.1f}", per_control["chosen_factor_counts"]])
    page += [markdown_table(["Method", "Criterion", "Choice over the mean", "Target WF1 at this choice (%)",
                             "Per-transfer choices", "Target WF1, per-transfer choices (%)",
                             "Per-transfer choices with the control as candidate"], rows)]

    page += [f"**Source weight 1 instead of the AdaTime source weights**, weighted F1 on target test, paired by transfer: "
             f"mean ± SD of the {n} differences in points, in brackets the number of transfers in which the paper "
             "convention is higher.", ""]
    rows = []
    for method in data.methods:
        cells = []
        for comparison in ("paper_x1_minus_adatime", "paper_control_minus_adatime_control"):
            d = data.difference(comparison, method)
            cells.append(f"{signed(d['diff_mean_points'])} ± {d['diff_sd_points']:.1f} ({d['transfers_a_higher']}/{d['n_transfers']})")
        own = [data.difference(c, method) for c in ("method_minus_control_paper_x1", "method_minus_control_adatime")]
        cells += [f"{signed(d['diff_mean_points'])} ± {d['diff_sd_points']:.1f} ({d['transfers_a_higher']}/{d['n_transfers']})"
                  for d in own]
        rows.append([METHOD_NAMES[method], *cells])
    page += [markdown_table(["Method", "Paper ×1 − AdaTime weights", "Shared control − AdaTime control",
                             "Method − control, paper ×1", "Method − control, AdaTime weights"], rows)]

    captions = [
        f"Mean over the {n} transfers against the factor (log scale; 0 = shared control). Top: weighted F1 on target test "
        "with the factors chosen by the label-free criteria; the dashed line is the baseline. Lower rows: the criteria on "
        "source validation. Open diamonds: the runs with the AdaTime weights at the same λ ratio (method at ×1, its own "
        "control at 0).",
        "Weighted F1 on target test, paper convention at factor 1 minus the run with the AdaTime weights, per transfer; "
        "the right panel compares the shared control with each of the three AdaTime controls.",
        "Weighted F1 on target test of every factor minus the shared control, per transfer; the framed cell is the factor "
        "that SRC (source-val cross-entropy) chooses for that transfer.",
    ]
    for k, (path, caption) in enumerate(zip(figures, captions)):
        page += [f"**Figure {first_figure + k}.** {caption}", "", f"![Figure {first_figure + k}]({path})", ""]
    page += ["**Observations (sensitivity).** Only statements that follow directly from the T7 tables; one seed, so "
             "differences of a few points between factors are within the variation seen between seeds above.", ""]
    page += [f"- {line}" for line in observations(data)] + [""]
    return page
