# plots.py
from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
import plotly.graph_objects as go


def plot_score_heatmap(
    scores: np.ndarray,
    title: str = "Score matrix: target (rows) vs context (cols)",
    show_values: bool = True,
) -> None:
    """
    Interactive heatmap of the VxV score matrix using Plotly.

    Args:
        scores: (V, V) array where scores[t, c] = dot(Wc[t], Wnce[c])
        title: plot title
        show_values: annotate cells with numeric values (recommended for small V)
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square (V, V) matrix.")

    V = scores.shape[0]
    text = None

    if show_values:
        text = [[f"{scores[i, j]:.2f}" for j in range(V)] for i in range(V)]

    fig = go.Figure(
        data=go.Heatmap(
            z=scores,
            text=text,
            texttemplate="%{text}" if show_values else None,
            colorscale="RdBu",
            zmid=0.0,
            colorbar=dict(title="Dot product"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Context hotel ID",
        yaxis_title="Target hotel ID",
        xaxis=dict(tickmode="array", tickvals=list(range(V))),
        yaxis=dict(tickmode="array", tickvals=list(range(V))),
        yaxis_autorange="reversed",
        width=600,
        height=600,
    )

    fig.show()


def plot_training_curves(
    epochs: Iterable[int],
    mean_loss: Iterable[float],
    mean_s_pos: Optional[Iterable[float]] = None,
    mean_s_neg: Optional[Iterable[float]] = None,
    title: str = "Training curves",
) -> None:
    """
    Interactive training curves using Plotly.

    Plots:
      1) Mean loss vs epoch
      2) Mean positive / negative dot products vs epoch (if provided)
    """
    epochs = np.array(list(epochs), dtype=int)
    mean_loss = np.array(list(mean_loss), dtype=float)

    # --- Loss curve ---
    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Scatter(
            x=epochs,
            y=mean_loss,
            mode="lines+markers",
            name="Mean loss",
        )
    )

    fig_loss.update_layout(
        title=f"{title}: Loss",
        xaxis_title="Epoch",
        yaxis_title="Mean loss",
        width=700,
        height=400,
    )

    fig_loss.show()

    # --- Dot product curves ---
    if mean_s_pos is not None and mean_s_neg is not None:
        mean_s_pos = np.array(list(mean_s_pos), dtype=float)
        mean_s_neg = np.array(list(mean_s_neg), dtype=float)

        fig_logits = go.Figure()
        fig_logits.add_trace(
            go.Scatter(
                x=epochs,
                y=mean_s_pos,
                mode="lines+markers",
                name="Mean positive dot product",
            )
        )
        fig_logits.add_trace(
            go.Scatter(
                x=epochs,
                y=mean_s_neg,
                mode="lines+markers",
                name="Mean negative dot product",
            )
        )

        fig_logits.update_layout(
            title=f"{title}: Dot products",
            xaxis_title="Epoch",
            yaxis_title="Value",
            width=700,
            height=400,
        )

        fig_logits.show()
