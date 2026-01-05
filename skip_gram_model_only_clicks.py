# model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from data import generate_skipgram_pairs
from typing import Optional
from diagnostics import (
    collect_snapshots,
    TrainingSnapshot,
    compute_score_matrix,
)
from plots import plot_score_heatmap, plot_training_curves
import plotly.io as pio

pio.renderers.default = "browser"


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.
    """
    # For large negative x, exp(x) is tiny; for large positive x, exp(-x) is tiny.
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable log(sigmoid(x)).
    Using: log(sigmoid(x)) = -log(1 + exp(-x))  for x >= 0
          log(sigmoid(x)) = x - log(1 + exp(x)) for x < 0
    """
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = -np.log1p(np.exp(-x[pos]))
    out[neg] = x[neg] - np.log1p(np.exp(x[neg]))
    return out


@dataclass
class TrainStats:
    epoch: int
    mean_loss: float
    mean_s_pos: float
    mean_s_neg: float


class SkipGramNCEClickModel:
    """
    Minimal click-only skip-gram model with negative sampling.

    Parameters:
      - Wc   : input embedding matrix  (V x d)
      - Wnce : output/context matrix   (V x d)

    For a training pair (t, c):
      v = Wc[t] where t is target hotel ID
      u_pos = Wnce[c] where c is positive context pair hotel ID
      u_neg = Wnce[n_k] for k in 1..K where n_k are negative samples

      s_pos = v dot u_pos
      s_neg[k] = v dot u_neg[k]

      loss = -log(sigmoid(s_pos)) - sum_k log(sigmoid(-s_neg[k]))
    """

    def __init__(
        self,
        num_hotels: int,
        embed_dim: int,
        num_negatives: int,
        lr: float = 0.1,
        seed: int = 42,
    ):
        if num_hotels < 3:
            raise ValueError("num_hotels must be at least 3 to allow negative sampling.")
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1.")
        if num_negatives < 1:
            raise ValueError("num_negatives must be >= 1.")

        self.V = num_hotels
        self.d = embed_dim
        self.K = num_negatives
        self.lr = lr

        rng = np.random.default_rng(seed)
        # Small random init is standard for these models
        self.Wc = 0.01 * rng.standard_normal((self.V, self.d))
        self.Wnce = 0.01 * rng.standard_normal((self.V, self.d))

        self.rng = rng

    def sample_negatives(self, target_id: int, context_id: int) -> np.ndarray:
        """
        Uniform negative sampling, excluding target and true context.
        Samples without replacement (cleaner for a toy demo).
        """
        forbidden = {target_id, context_id}
        candidates = [i for i in range(self.V) if i not in forbidden]

        if len(candidates) < self.K:
            raise ValueError(
                "Not enough candidates to sample negatives. "
                "Increase num_hotels or decrease num_negatives."
            )

        neg_ids = self.rng.choice(candidates, size=self.K, replace=False)
        return neg_ids.astype(int)

    def loss_and_grads_for_pair(self, target_id: int, context_id: int) -> Tuple[float, float, float]:
        """
        Compute loss and apply an online SGD update for one (target, context) pair.

        Returns:
          loss, s_pos, mean(s_neg)
        """
        neg_ids = self.sample_negatives(target_id, context_id)

        # --- Forward pass ---
        v = self.Wc[target_id]              # (d,)
        u_pos = self.Wnce[context_id]       # (d,)
        u_neg = self.Wnce[neg_ids]          # (K, d)

        s_pos = float(np.dot(v, u_pos))                 # scalar
        s_neg = (u_neg @ v).astype(float)               # (K,)

        # loss = -logσ(s_pos) - Σ logσ(-s_neg)
        # We use log_sigmoid for numerical stability
        loss_pos = -float(log_sigmoid(np.array([s_pos]))[0])
        loss_neg = -float(np.sum(log_sigmoid(-s_neg)))
        loss = loss_pos + loss_neg

        # --- Backprop (manual gradients) ---
        #
        # d/ds_pos of [-logσ(s_pos)] = σ(s_pos) - 1
        # d/ds_neg of [-logσ(-s_neg)] = σ(s_neg)
        #
        sig_pos = float(sigmoid(np.array([s_pos]))[0])
        grad_s_pos = sig_pos - 1.0

        sig_neg = sigmoid(s_neg)           # (K,)
        grad_s_neg = sig_neg               # (K,)

        # Gradients w.r.t vectors
        #
        # s = v·u  =>  ∂s/∂v = u,  ∂s/∂u = v
        grad_v = grad_s_pos * u_pos + (grad_s_neg[:, None] * u_neg).sum(axis=0)  # (d,)
        grad_u_pos = grad_s_pos * v                                               # (d,)
        grad_u_neg = grad_s_neg[:, None] * v[None, :]                             # (K, d)

        # --- SGD update (only touched rows) ---
        self.Wc[target_id] -= self.lr * grad_v
        self.Wnce[context_id] -= self.lr * grad_u_pos
        self.Wnce[neg_ids] -= self.lr * grad_u_neg

        return loss, s_pos, float(np.mean(s_neg))

    def train(
        self,
        pairs: List[Tuple[int, int]],
        max_epochs: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
        snapshots: Optional[list[TrainingSnapshot]] = None,
        snapshot_every: int = 1,
    ) -> List[TrainStats]:
        """
        Online SGD training until convergence.

        Convergence rule:
          stop if the absolute improvement in mean epoch loss < tol

        Args:
          pairs: list of (target, context) pairs
          max_epochs: safety limit
          tol: convergence threshold
          verbose: prints epoch summaries

        Returns:
          List of per-epoch training stats.
        """
        if len(pairs) == 0:
            raise ValueError("No training pairs provided.")

        history: List[TrainStats] = []
        prev_mean_loss = None

        for epoch in range(1, max_epochs + 1):
            # Shuffle pairs each epoch
            idx = self.rng.permutation(len(pairs))

            losses = []
            s_pos_list = []
            s_neg_list = []

            for i in idx:
                t, c = pairs[i]
                loss, s_pos, mean_s_neg = self.loss_and_grads_for_pair(t, c)
                losses.append(loss)
                s_pos_list.append(s_pos)
                s_neg_list.append(mean_s_neg)

            mean_loss = float(np.mean(losses))
            mean_s_pos = float(np.mean(s_pos_list))
            mean_s_neg = float(np.mean(s_neg_list))

            history.append(TrainStats(epoch, mean_loss, mean_s_pos, mean_s_neg))

            if snapshots is not None and (epoch % snapshot_every == 0):
                collect_snapshots(epoch=epoch, Wc=self.Wc, Wnce=self.Wnce, store=snapshots, copy_arrays=True)

            if verbose and (epoch == 1 or epoch % 10 == 0):
                print(
                    f"Epoch {epoch:3d} | mean_loss={mean_loss:.6f} "
                    f"| mean_s_pos={mean_s_pos:+.4f} | mean_s_neg={mean_s_neg:+.4f}"
                )

            if prev_mean_loss is not None:
                improvement = abs(prev_mean_loss - mean_loss)
                if improvement < tol:
                    if verbose:
                        print(
                            f"Converged at epoch {epoch} "
                            f"(abs loss improvement {improvement:.2e} < tol {tol:.2e})."
                        )
                    break

            prev_mean_loss = mean_loss

        return history


def main() -> None:
    # Two sessions, each with three clicks.
    # Use at least 4 hotels in the vocabulary so negatives exist cleanly.
    sessions = [
        [0, 1, 2],
        [2, 1, 3],
    ]
    window_size = 1

    pairs = generate_skipgram_pairs(sessions, window_size=window_size)
    print("Sessions:", sessions)
    print("Window size:", window_size)
    print("Training pairs (target, context):", pairs)
    print("Number of pairs:", len(pairs))
    print()

    # Model hyperparameters (small, simple)
    num_hotels = 4     # hotels are IDs 0..3
    embed_dim = 5
    num_negatives = 2
    lr = 0.5

    model = SkipGramNCEClickModel(
        num_hotels=num_hotels,
        embed_dim=embed_dim,
        num_negatives=num_negatives,
        lr=lr,
        seed=42,
    )

    history = model.train(
        pairs=pairs,
        max_epochs=2000,
        tol=1e-7,
        verbose=True,
    )

    scores = compute_score_matrix(model.Wc, model.Wnce)
    plot_score_heatmap(scores, show_values=True)

    epochs = [h.epoch for h in history]
    mean_loss = [h.mean_loss for h in history]
    mean_s_pos = [h.mean_s_pos for h in history]
    mean_s_neg = [h.mean_s_neg for h in history]
    plot_training_curves(epochs, mean_loss, mean_s_pos, mean_s_neg)

    print("\nFinal matrices (rounded):")
    print("\nWc (input embeddings):")
    print(model.Wc.round(4))
    print("\nWnce (output/context embeddings):")
    print(model.Wnce.round(4))

    # A tiny “what did it learn?” peek:
    # Show dot products for a few likely-positive pairs from the sessions.
    def score(t: int, c: int) -> float:
        return float(model.Wc[t] @ model.Wnce[c])

    print("\nDot products for observed context pairs:")
    observed = list(dict.fromkeys(pairs))  # unique in original order
    for t, c in observed:
        print(f"score(target={t}, context={c}) = {score(t,c):+.4f}")


if __name__ == "__main__":
    main()
