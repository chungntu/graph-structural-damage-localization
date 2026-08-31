"""Plot the experimental-record structure and swept-sine modal-fit quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    face: str,
    edge: str,
    fontsize: float = 8.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.35,
        facecolor=face,
        edgecolor=edge,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.15,
    )


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color="#444444",
        )
    )


def _pass_labels(cache: np.lib.npyio.NpzFile) -> list[str]:
    labels = []
    healthy_counter: dict[tuple[str, str], int] = {}
    for position, recording, state, direction in zip(
        cache["positions"],
        cache["recordings"],
        cache["states"],
        cache["sweep_directions"],
    ):
        direction_symbol = "up" if direction == "up" else "down"
        if state == "UDS":
            key = (str(position), str(recording))
            if key not in healthy_counter:
                existing = [item for item in healthy_counter if item[0] == position]
                healthy_counter[key] = len(existing) + 1
            state_label = f"H{healthy_counter[key]}"
        else:
            state_label = "D"
        labels.append(f"{state_label}-{direction_symbol}")
    return labels


def make_figure(cache_path: Path, validation_json: Path, output_path: Path) -> None:
    cache = np.load(cache_path, allow_pickle=False)
    protocol = json.loads(validation_json.read_text(encoding="utf-8"))["protocol"]

    modal_frequencies = cache["modal_frequencies_hz"]
    centers = np.asarray(protocol["modal_centers_hz"], dtype=float)
    median_coherence = cache["median_coherence"]
    coherent_fraction = cache["coherent_fraction"]
    weighted_nmse = cache["weighted_nmse"]
    states = cache["states"]
    labels = _pass_labels(cache)
    x = np.arange(len(labels))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
        }
    )
    figure = plt.figure(figsize=(15.6, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.05, 1.35, 1.1),
        height_ratios=(1, 1),
    )

    protocol_axis = figure.add_subplot(grid[:, 0])
    protocol_axis.set_xlim(0, 1)
    protocol_axis.set_ylim(0, 1)
    protocol_axis.axis("off")
    protocol_axis.set_title("(a) Experimental record structure", loc="left", pad=8)

    _box(
        protocol_axis,
        0.06,
        0.84,
        0.88,
        0.09,
        "Development: P2 vertical random input",
        "#e3f2fd",
        "#1565c0",
        9.0,
    )
    _arrow(protocol_axis, (0.50, 0.84), (0.50, 0.79))
    _box(
        protocol_axis,
        0.05,
        0.64,
        0.27,
        0.13,
        "UDS01\nhealthy template",
        "#e8f5e9",
        "#2e7d32",
    )
    _box(
        protocol_axis,
        0.365,
        0.64,
        0.27,
        0.13,
        "DS1\ndevelopment check",
        "#fce4ec",
        "#c2185b",
    )
    _box(
        protocol_axis,
        0.68,
        0.64,
        0.27,
        0.13,
        "UDS02\nheld-out healthy",
        "#fff8e1",
        "#f9a825",
    )

    _box(
        protocol_axis,
        0.06,
        0.44,
        0.88,
        0.09,
        "Validation: P1 and P2 vertical swept sine",
        "#f3e5f5",
        "#7b1fa2",
        9.0,
    )
    _arrow(protocol_axis, (0.50, 0.44), (0.50, 0.39))
    _box(
        protocol_axis,
        0.07,
        0.22,
        0.40,
        0.15,
        "Healthy at each position\n2 records x 2 directions\n= 4 passages",
        "#e8f5e9",
        "#2e7d32",
    )
    _box(
        protocol_axis,
        0.53,
        0.22,
        0.40,
        0.15,
        "DS1 at each position\n1 record x 2 directions\n= 2 passages",
        "#fce4ec",
        "#c2185b",
    )
    _arrow(protocol_axis, (0.50, 0.22), (0.50, 0.16))
    _box(
        protocol_axis,
        0.19,
        0.05,
        0.62,
        0.09,
        "12 swept-sine modal fits in total",
        "#fff3e0",
        "#e65100",
        9.0,
    )

    frequency_axis = figure.add_subplot(grid[:, 1])
    deviations = modal_frequencies - centers[None, :]
    mode_colors = ("#1565c0", "#2e7d32", "#c2185b")
    markers = ("o", "s", "^")
    for mode_index, (color, marker) in enumerate(zip(mode_colors, markers)):
        frequency_axis.plot(
            x,
            deviations[:, mode_index],
            marker=marker,
            markersize=4.5,
            linewidth=1.5,
            color=color,
            label=f"Mode {mode_index + 1}",
        )
    frequency_axis.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
    frequency_axis.axvline(5.5, color="#777777", linewidth=1.0)
    frequency_axis.axvspan(-0.5, 5.5, color="#e3f2fd", alpha=0.35, zorder=0)
    frequency_axis.axvspan(5.5, 11.5, color="#fff3e0", alpha=0.35, zorder=0)
    frequency_axis.text(2.5, 0.19, "P1", ha="center", va="top", fontweight="bold")
    frequency_axis.text(8.5, 0.19, "P2", ha="center", va="top", fontweight="bold")
    frequency_axis.set_title("(b) Fitted modal frequencies across passages", loc="left")
    frequency_axis.set_ylabel("Fitted frequency - initial center (Hz)")
    frequency_axis.set_xticks(x, labels, rotation=55, ha="right")
    frequency_axis.set_xlim(-0.5, 11.5)
    frequency_axis.set_ylim(-0.19, 0.21)
    frequency_axis.grid(alpha=0.25)
    frequency_axis.legend(loc="lower left", ncol=3, frameon=True)

    coherence_axis = figure.add_subplot(grid[0, 2])
    coherence_axis.plot(
        x,
        median_coherence,
        color="#1565c0",
        marker="o",
        markersize=4,
        linewidth=1.4,
        label="median coherence",
    )
    coherence_axis.plot(
        x,
        coherent_fraction,
        color="#7b1fa2",
        marker="s",
        markersize=3.8,
        linewidth=1.2,
        label="fraction above 0.5",
    )
    coherence_axis.axhline(0.95, color="#555555", linestyle="--", linewidth=0.9)
    coherence_axis.text(
        11.35,
        0.9508,
        "median-coherence gate",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#444444",
    )
    coherence_axis.axvline(5.5, color="#777777", linewidth=1.0)
    coherence_axis.set_title("(c) Input-output coherence", loc="left")
    coherence_axis.set_ylabel("Coherence metric")
    coherence_axis.set_xlim(-0.5, 11.5)
    coherence_axis.set_ylim(0.94, 1.002)
    coherence_axis.set_xticks([])
    coherence_axis.grid(alpha=0.25)
    coherence_axis.legend(loc="lower right")

    nmse_axis = figure.add_subplot(grid[1, 2])
    point_colors = np.where(states == "DS1", "#c2185b", "#1565c0")
    nmse_axis.scatter(x, weighted_nmse, c=point_colors, s=30, zorder=3)
    nmse_axis.plot(x, weighted_nmse, color="#777777", linewidth=0.9, zorder=2)
    nmse_axis.axhline(
        float(np.median(weighted_nmse)),
        color="#e65100",
        linestyle="--",
        linewidth=1.0,
    )
    nmse_axis.axvline(5.5, color="#777777", linewidth=1.0)
    nmse_axis.set_title("(d) Three-mode fitting error", loc="left")
    nmse_axis.set_ylabel("Weighted NMSE")
    nmse_axis.set_xticks(x, labels, rotation=55, ha="right")
    nmse_axis.set_xlim(-0.5, 11.5)
    nmse_axis.set_ylim(0, 0.052)
    nmse_axis.grid(alpha=0.25)
    nmse_axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color="#1565c0", label="healthy"),
            Line2D([], [], marker="o", linestyle="none", color="#c2185b", label="DS1"),
            Line2D([], [], linestyle="--", color="#e65100", label="median"),
        ],
        loc="upper right",
        ncol=3,
        columnspacing=0.8,
        handlelength=1.6,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.cache, args.validation_json, args.output)


if __name__ == "__main__":
    main()
