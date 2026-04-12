# Licensed under the Apache License, Version 2.0

"""Plot helpers for broad-compartment decomposition results."""

import numpy as np
import matplotlib.pyplot as plt


def plot_decomposition_summary(
    results,
    call_summary=None,
    save_to_filename=None,
    save_dpi=300,
    top_hypotheses=6,
    top_markers_per_component=4,
):
    """Render a human-interpretable summary of decomposition logic."""
    if not results:
        return None

    best = results[0]
    hypotheses = results[:top_hypotheses]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 12),
        gridspec_kw={"width_ratios": [1.3, 1.0], "height_ratios": [1.0, 1.0]},
    )
    ax_hyp, ax_frac = axes[0]
    ax_comp, ax_mark = axes[1]

    # --- Panel 1: ranked hypotheses ---
    labels = [f"{row.cancer_type} / {row.template}" for row in hypotheses]
    scores = [row.score for row in hypotheses]
    colors = ["#1f77b4" if i == 0 else "#9ecae1" for i in range(len(hypotheses))]
    y = np.arange(len(hypotheses))
    ax_hyp.barh(y, scores, color=colors, height=0.65)
    ax_hyp.set_yticks(y)
    ax_hyp.set_yticklabels(labels, fontsize=9)
    ax_hyp.invert_yaxis()
    ax_hyp.set_xlabel("Combined decomposition score")
    ax_hyp.set_title("Hypotheses", fontweight="bold")
    for idx, row in enumerate(hypotheses):
        ax_hyp.text(
            scores[idx] + 0.005,
            idx,
            (
                f"purity={row.purity:.0%}, cancer={row.cancer_support_score:.2f}, "
                f"site={row.template_tissue_score:.2f}, factor={row.template_site_factor:.2f}, "
                f"extra={row.template_extra_fraction:.0%}"
            ),
            va="center",
            fontsize=8,
        )
    ax_hyp.spines["top"].set_visible(False)
    ax_hyp.spines["right"].set_visible(False)
    if call_summary:
        lines = []
        if call_summary.get("label_options"):
            if len(call_summary["label_options"]) == 2:
                lines.append(
                    f"Possible labels: {call_summary['label_options'][0]} or {call_summary['label_options'][1]}"
                )
            else:
                lines.append(f"Resolved label: {call_summary['label_options'][0]}")
        if call_summary.get("site_indeterminate"):
            lines.append("Site/template: indeterminate")
        elif call_summary.get("reported_site"):
            lines.append(f"Site/template: {call_summary['reported_site']}")
        if lines:
            ax_hyp.text(
                0.02,
                0.02,
                "\n".join(lines),
                transform=ax_hyp.transAxes,
                fontsize=8.5,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
            )

    # --- Panel 2: final composition ---
    frac_items = sorted(best.fractions.items(), key=lambda item: item[1], reverse=True)
    left = 0.0
    palette = [
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#d62728",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#7f7f7f",
    ]
    for idx, (name, value) in enumerate(frac_items):
        ax_frac.barh(
            [0],
            [value * 100],
            left=left * 100,
            color=palette[idx % len(palette)],
            edgecolor="none",
            height=0.55,
            label=f"{name} ({value:.0%})",
        )
        left += value
    ax_frac.set_xlim(0, 100)
    ax_frac.set_yticks([])
    ax_frac.set_xlabel("Estimated composition (%)")
    ax_frac.set_title("Best-fit composition", fontweight="bold")
    ax_frac.legend(loc="lower center", fontsize=8, ncol=2, framealpha=0.9)
    ax_frac.spines["top"].set_visible(False)
    ax_frac.spines["right"].set_visible(False)
    ax_frac.spines["left"].set_visible(False)

    # --- Panel 3: component evidence ---
    comp_df = best.component_trace.copy()
    if comp_df.empty:
        ax_comp.text(0.5, 0.5, "No component trace", ha="center", va="center", transform=ax_comp.transAxes)
        ax_comp.set_axis_off()
    else:
        comp_df = comp_df.sort_values("fraction", ascending=True).reset_index(drop=True)
        y = np.arange(len(comp_df))
        ax_comp.barh(y, comp_df["fraction"] * 100, color="#4c78a8", alpha=0.85, height=0.55)
        ax_comp.set_yticks(y)
        ax_comp.set_yticklabels(comp_df["component"], fontsize=9)
        ax_comp.set_xlabel("Fraction of sample (%)")
        ax_comp.set_title("Component support", fontweight="bold")
        for idx, row in comp_df.iterrows():
            txt = f"marker={row['marker_score']:.2f}" if row["marker_score"] is not None else "marker=n/a"
            ax_comp.text(row["fraction"] * 100 + 0.8, idx, txt, va="center", fontsize=8)
        ax_comp.spines["top"].set_visible(False)
        ax_comp.spines["right"].set_visible(False)

    # --- Panel 4: marker trace text ---
    ax_mark.set_title("Marker logic", fontweight="bold")
    ax_mark.axis("off")
    if best.marker_trace.empty:
        ax_mark.text(0.02, 0.98, "No marker trace", ha="left", va="top", fontsize=10)
    else:
        lines = []
        for component in best.component_trace["component"].tolist()[: min(6, len(best.component_trace))]:
            sub = best.marker_trace[best.marker_trace["component"] == component].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["observed_tpm", "specificity"], ascending=[False, False]).head(top_markers_per_component)
            entries = [
                f"{row.symbol}={row.observed_tpm:.1f} TPM"
                for row in sub.itertuples()
            ]
            lines.append(f"{component}: " + ", ".join(entries))

        if best.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend([f"- {warning}" for warning in best.warnings[:4]])

        ax_mark.text(
            0.02,
            0.98,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
        )

    title = f"Broad-compartment decomposition — {best.cancer_type} / {best.template}"
    if call_summary and call_summary.get("site_indeterminate"):
        title = f"Broad-compartment decomposition — {call_summary.get('label_display', best.cancer_type)} (site indeterminate)"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_to_filename:
        fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
        print(f"Saved {save_to_filename}")
    return fig
