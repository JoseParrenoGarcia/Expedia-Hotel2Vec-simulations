# model_enriched.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# Reuse your existing utilities (same ones used in click-only code)
from diagnostics import collect_snapshots, TrainingSnapshot
from plots import plot_score_heatmap, plot_training_curves
from data import generate_skipgram_pairs


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.
    """
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


def compute_score_matrix_from_embeddings(Ve_all: np.ndarray, Wnce: np.ndarray) -> np.ndarray:
    """
    Score matrix for enriched model:
        scores[target, context] = Ve[target] dot Wnce[context]

    Shapes:
        Ve_all : (V, d_e)
        Wnce   : (V, d_e)
        scores : (V, V)
    """
    if Ve_all.ndim != 2 or Wnce.ndim != 2:
        raise ValueError("Ve_all and Wnce must be 2D arrays.")
    if Ve_all.shape[1] != Wnce.shape[1]:
        raise ValueError("Embedding dimensions must match between Ve_all and Wnce.")
    return Ve_all @ Wnce.T


class Hotel2VecNCEModel:
    """
    Click + Amenities + Geo version of the skip-gram NCE model.

    Conceptually:
      - Click tower:      vc = Wc[target_id]                            (d_c,)
      - Amenity tower:    va = A[target_id] @ Wa                        (d_a,)
      - Geo tower:        vg = G[target_id] @ Wg                        (d_g,)
      - Concatenate:      h  = [vc || va || vg]                         (d_c+d_a+d_g,)
      - Enrich:           ve = h @ We                                   (d_e,)
      - Score contexts:   s = ve dot Wnce[context_id] and ve dot Wnce[neg_id]

    Loss (Equation 3 style):
      loss = -log(sigmoid(s_pos)) - sum_k log(sigmoid(-s_neg[k]))

    What is trainable?
      - Wc   : (V, d_c)      click embeddings (lookup table)
      - Wa   : (a_in, d_a)   amenity projection
      - Wg   : (g_in, d_g)   geo projection
      - We   : (d_c+d_a+d_g, d_e)  enrichment projection
      - Wnce : (V, d_e)      output/context embeddings

    What is fixed input?
      - A : amenity features per hotel (V, a_in)
      - G : geo features per hotel     (V, g_in)
    """

    def __init__(
        self,
        num_hotels: int,
        click_dim: int,
        amenity_in_dim: int,
        amenity_dim: int,
        geo_in_dim: int,
        geo_dim: int,
        enriched_dim: int,
        num_negatives: int,
        lr: float = 0.1,
        seed: int = 42,
    ):
        if num_hotels < 3:
            raise ValueError("num_hotels must be at least 3 to allow negative sampling.")
        if num_negatives < 1:
            raise ValueError("num_negatives must be >= 1.")
        if click_dim < 1 or amenity_dim < 1 or geo_dim < 1 or enriched_dim < 1:
            raise ValueError("All embedding dims must be >= 1.")
        if amenity_in_dim < 1 or geo_in_dim < 1:
            raise ValueError("Input feature dims must be >= 1.")

        self.V = num_hotels
        self.K = num_negatives
        self.lr = lr

        self.d_c = click_dim
        self.a_in = amenity_in_dim
        self.d_a = amenity_dim
        self.g_in = geo_in_dim
        self.d_g = geo_dim
        self.d_e = enriched_dim

        self.concat_dim = self.d_c + self.d_a + self.d_g

        rng = np.random.default_rng(seed)

        # Trainable parameters
        self.Wc = 0.01 * rng.standard_normal((self.V, self.d_c))                  # (V, d_c)
        self.Wa = 0.01 * rng.standard_normal((self.a_in, self.d_a))               # (a_in, d_a)
        self.Wg = 0.01 * rng.standard_normal((self.g_in, self.d_g))               # (g_in, d_g)
        self.We = 0.01 * rng.standard_normal((self.concat_dim, self.d_e))         # (concat_dim, d_e)
        self.Wnce = 0.01 * rng.standard_normal((self.V, self.d_e))                # (V, d_e)

        self.rng = rng

        # Fixed per-hotel features (you will set these via set_features)
        self.A = None  # shape (V, a_in)
        self.G = None  # shape (V, g_in)

    def set_features(self, amenity_features: np.ndarray, geo_features: np.ndarray) -> None:
        """
        Set fixed hotel features used by the amenity and geo towers.

        Args:
            amenity_features: (V, a_in)
            geo_features:     (V, g_in)
        """
        if amenity_features.shape != (self.V, self.a_in):
            raise ValueError(f"amenity_features must have shape {(self.V, self.a_in)}.")
        if geo_features.shape != (self.V, self.g_in):
            raise ValueError(f"geo_features must have shape {(self.V, self.g_in)}.")
        self.A = amenity_features.astype(float)
        self.G = geo_features.astype(float)

    def sample_negatives(self, target_id: int, context_id: int) -> np.ndarray:
        """
        Uniform negative sampling, excluding target and true context.
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

    def _forward_target(self, target_id: int) -> Tuple[np.ndarray, dict]:
        """
        Forward pass for the target hotel only, producing Ve.

        Returns:
            ve: (d_e,)
            cache: dict for backprop
        """
        if self.A is None or self.G is None:
            raise ValueError("Features not set. Call set_features(...) before training.")

        # Click tower (lookup)
        vc = self.Wc[target_id]  # (d_c,)

        # Amenity tower
        a = self.A[target_id]            # (a_in,)
        va = a @ self.Wa                 # (d_a,)

        # Geo tower
        g = self.G[target_id]            # (g_in,)
        vg = g @ self.Wg                 # (d_g,)

        # Concatenate
        h = np.concatenate([vc, va, vg], axis=0)  # (concat_dim,)

        # Enrich
        ve = h @ self.We  # (d_e,)

        cache = {
            "target_id": target_id,
            "vc": vc,
            "a": a,
            "va": va,
            "g": g,
            "vg": vg,
            "h": h,
            "ve": ve,
        }
        return ve, cache

    def loss_and_grads_for_pair(self, target_id: int, context_id: int) -> Tuple[float, float, float]:
        """
        Compute loss and apply an online SGD update for one (target, context) pair.

        Returns:
          loss, s_pos, mean(s_neg)
        """
        neg_ids = self.sample_negatives(target_id, context_id)

        # --- Forward ---
        ve, cache = self._forward_target(target_id)     # (d_e,)
        u_pos = self.Wnce[context_id]                   # (d_e,)
        u_neg = self.Wnce[neg_ids]                      # (K, d_e)

        s_pos = float(np.dot(ve, u_pos))                # scalar
        s_neg = (u_neg @ ve).astype(float)              # (K,)

        # loss = -logσ(s_pos) - Σ logσ(-s_neg)
        loss_pos = -float(log_sigmoid(np.array([s_pos]))[0])
        loss_neg = -float(np.sum(log_sigmoid(-s_neg)))
        loss = loss_pos + loss_neg

        # --- Backprop: logits -> ve, Wnce rows ---
        sig_pos = float(sigmoid(np.array([s_pos]))[0])
        grad_s_pos = sig_pos - 1.0                      # scalar

        sig_neg = sigmoid(s_neg)                        # (K,)
        grad_s_neg = sig_neg                            # (K,)

        # Gradients w.r.t ve and output embeddings
        grad_ve = grad_s_pos * u_pos + (grad_s_neg[:, None] * u_neg).sum(axis=0)  # (d_e,)
        grad_u_pos = grad_s_pos * ve                                            # (d_e,)
        grad_u_neg = grad_s_neg[:, None] * ve[None, :]                          # (K, d_e)

        # --- Backprop: ve = h @ We ---
        h = cache["h"]                                      # (concat_dim,)
        grad_We = np.outer(h, grad_ve)                      # (concat_dim, d_e)
        grad_h = self.We @ grad_ve                          # (concat_dim,)

        # Split grad_h into [grad_vc, grad_va, grad_vg]
        grad_vc = grad_h[: self.d_c]                        # (d_c,)
        grad_va = grad_h[self.d_c : self.d_c + self.d_a]    # (d_a,)
        grad_vg = grad_h[self.d_c + self.d_a :]             # (d_g,)

        # --- Backprop: va = a @ Wa, vg = g @ Wg ---
        a = cache["a"]                                      # (a_in,)
        g = cache["g"]                                      # (g_in,)

        grad_Wa = np.outer(a, grad_va)                      # (a_in, d_a)
        grad_Wg = np.outer(g, grad_vg)                      # (g_in, d_g)

        # --- SGD updates (only touched items where possible) ---
        # Target click embedding row
        self.Wc[target_id] -= self.lr * grad_vc

        # Shared projection matrices
        self.Wa -= self.lr * grad_Wa
        self.Wg -= self.lr * grad_Wg
        self.We -= self.lr * grad_We

        # Output/context embeddings: only update touched rows
        self.Wnce[context_id] -= self.lr * grad_u_pos
        self.Wnce[neg_ids] -= self.lr * grad_u_neg

        return loss, s_pos, float(np.mean(s_neg))

    def compute_all_target_embeddings(self) -> np.ndarray:
        """
        Compute Ve for all hotels (V, d_e), using current parameters.
        Useful for heatmaps and later diagnostics.
        """
        if self.A is None or self.G is None:
            raise ValueError("Features not set. Call set_features(...) before using this method.")

        Ve_all = np.zeros((self.V, self.d_e), dtype=float)
        for hid in range(self.V):
            ve, _ = self._forward_target(hid)
            Ve_all[hid] = ve
        return Ve_all

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
        """
        if self.A is None or self.G is None:
            raise ValueError("Features not set. Call set_features(...) before training.")
        if len(pairs) == 0:
            raise ValueError("No training pairs provided.")

        history: List[TrainStats] = []
        prev_mean_loss = None

        for epoch in range(1, max_epochs + 1):
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

            # Snapshot hook (stores Wc/Wnce like your click-only version)
            # You can extend TrainingSnapshot later to include Wa/Wg/We if you want.
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

import numpy as np

def l2_normalise(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(v, ord=2))
    return v / (norm + eps)

def relu(v: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, v)

def pretty(x: np.ndarray, precision: int = 4) -> str:
    return np.array2string(
        x,
        precision=precision,
        suppress_small=True,
        floatmode="fixed",
    )

def print_hotel2vec_forward_trace(
    model,
    target_id: int,
    context_id: int,
    apply_l2: bool = True,
    apply_relu: bool = True,
    print_full_matrices: bool = False,
) -> None:
    """
    Print all vectors/matrices needed to fill your diagram for one (target, context) example.

    Args:
        model: Hotel2VecNCEModel (trained or untrained, but you said ideally trained)
        target_id: target hotel ID (ht)
        context_id: context hotel ID (hc)
        apply_l2: whether to show L2 normalised versions (as per paper)
        apply_relu: whether to show ReLU versions (as per paper)
        print_full_matrices: if True, prints entire W matrices, otherwise prints shapes + relevant rows
    """
    if model.A is None or model.G is None:
        raise ValueError("Model features are not set. Call model.set_features(...) first.")

    np.set_printoptions(suppress=True)

    V = model.V
    print("\n" + "=" * 90)
    print(f"Hotel2Vec forward trace for pair (target={target_id}, context={context_id})")
    print("=" * 90)

    # -------------------------
    # 1) CLICK PATH
    # -------------------------
    print("\n[Clicks]")
    Ic = np.zeros(V, dtype=int)
    Ic[target_id] = 1
    print(f"Ic (one-hot, length {V}):\n{Ic}")

    if print_full_matrices:
        print(f"\nWc (trainable, shape {model.Wc.shape}):\n{pretty(model.Wc)}")
    else:
        print(f"\nWc shape: {model.Wc.shape}")
        print(f"Wc[target] row (this is Vc before norm/ReLU):\n{pretty(model.Wc[target_id])}")

    Vc = model.Wc[target_id].copy()
    print(f"\nVc (embedding, shape {Vc.shape}):\n{pretty(Vc)}")

    Vc_norm = l2_normalise(Vc) if apply_l2 else Vc
    Vc_relu = relu(Vc_norm) if apply_relu else Vc_norm
    print("\nVc after normalisation + ReLU:")
    if apply_l2:
        print(f"Vc_norm:\n{pretty(Vc_norm)}")
    if apply_relu:
        print(f"Vc_relu:\n{pretty(Vc_relu)}")

    # -------------------------
    # 2) AMENITY PATH
    # -------------------------
    print("\n[Amenities]")
    Ia = model.A[target_id].copy()
    print(f"Ia (amenity features, shape {Ia.shape}):\n{pretty(Ia)}")

    if print_full_matrices:
        print(f"\nWa (trainable, shape {model.Wa.shape}):\n{pretty(model.Wa)}")
    else:
        print(f"\nWa shape: {model.Wa.shape}")

    Va = Ia @ model.Wa
    print(f"\nVa = Ia @ Wa (shape {Va.shape}):\n{pretty(Va)}")

    Va_norm = l2_normalise(Va) if apply_l2 else Va
    Va_relu = relu(Va_norm) if apply_relu else Va_norm
    print("\nVa after normalisation + ReLU:")
    if apply_l2:
        print(f"Va_norm:\n{pretty(Va_norm)}")
    if apply_relu:
        print(f"Va_relu:\n{pretty(Va_relu)}")

    # -------------------------
    # 3) GEO PATH
    # -------------------------
    print("\n[Geography]")
    Ig = model.G[target_id].copy()
    print(f"Ig (geo features, shape {Ig.shape}):\n{pretty(Ig)}")

    if print_full_matrices:
        print(f"\nWg (trainable, shape {model.Wg.shape}):\n{pretty(model.Wg)}")
    else:
        print(f"\nWg shape: {model.Wg.shape}")

    Vg = Ig @ model.Wg
    print(f"\nVg = Ig @ Wg (shape {Vg.shape}):\n{pretty(Vg)}")

    Vg_norm = l2_normalise(Vg) if apply_l2 else Vg
    Vg_relu = relu(Vg_norm) if apply_relu else Vg_norm
    print("\nVg after normalisation + ReLU:")
    if apply_l2:
        print(f"Vg_norm:\n{pretty(Vg_norm)}")
    if apply_relu:
        print(f"Vg_relu:\n{pretty(Vg_relu)}")

    # -------------------------
    # 4) CONCATENATION
    # -------------------------
    print("\n[Concatenation]")
    concat = np.concatenate([Vc_relu, Va_relu, Vg_relu], axis=0)
    print(f"V_c+a+g (shape {concat.shape}):\n{pretty(concat)}")

    # -------------------------
    # 5) ENRICHMENT (We -> Ve)
    # -------------------------
    print("\n[Enrichment]")
    if print_full_matrices:
        print(f"We (trainable, shape {model.We.shape}):\n{pretty(model.We)}")
    else:
        print(f"We shape: {model.We.shape}")

    Ve = concat @ model.We
    print(f"\nVe = (V_c+a+g) @ We (shape {Ve.shape}):\n{pretty(Ve)}")

    # -------------------------
    # 6) CONTEXT SCORING (optional, but useful)
    # -------------------------
    print("\n[Context scoring]")
    u_context = model.Wnce[context_id]
    if print_full_matrices:
        print(f"Wnce (trainable, shape {model.Wnce.shape}):\n{pretty(model.Wnce)}")
    else:
        print(f"Wnce shape: {model.Wnce.shape}")
        print(f"Wnce[context] row:\n{pretty(u_context)}")

    score = float(Ve @ u_context)
    print(f"\nScore s = Ve · Wnce[context] = {score:.6f}")

    print("\n" + "=" * 90)



def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    """
    Minimal demo to prove the enriched model runs end-to-end.
    This is intentionally tiny, mirroring your earlier toy scenario.
    """
    # Click sessions (same as your simple example)
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

    # Toy model sizes
    num_hotels = 4
    click_dim = 5

    # For the toy demo, we use small dense features rather than sparse one-hot features.
    # In a real system:
    # - amenities would be a structured feature vector (multi-hot / numeric)
    # - geo could be lat/lon bins, region IDs, etc.
    amenity_in_dim = 6
    amenity_dim = 3
    geo_in_dim = 2
    geo_dim = 2

    enriched_dim = 5
    num_negatives = 2
    lr = 0.3

    model = Hotel2VecNCEModel(
        num_hotels=num_hotels,
        click_dim=click_dim,
        amenity_in_dim=amenity_in_dim,
        amenity_dim=amenity_dim,
        geo_in_dim=geo_in_dim,
        geo_dim=geo_dim,
        enriched_dim=enriched_dim,
        num_negatives=num_negatives,
        lr=lr,
        seed=42,
    )

    # Fixed features per hotel (toy numbers, deterministic)
    rng = np.random.default_rng(123)
    amenity_features = rng.normal(size=(num_hotels, amenity_in_dim))
    geo_features = rng.normal(size=(num_hotels, geo_in_dim))
    model.set_features(amenity_features, geo_features)

    # ---- Debug / inspection: single forward pass ----

    target_id = 2
    context_id = 1

    history = model.train(
        pairs=pairs,
        max_epochs=2500,
        tol=1e-7,
        verbose=True,
    )

    print_hotel2vec_forward_trace(
        model=model,
        target_id=2,
        context_id=1,
        apply_l2=True,
        apply_relu=True,
        print_full_matrices=True,  # set True if you really want full W printed
    )

    # Heatmap of scores using Ve (enriched) vs Wnce
    Ve_all = model.compute_all_target_embeddings()
    scores = compute_score_matrix_from_embeddings(Ve_all, model.Wnce)
    # plot_score_heatmap(scores, show_values=True)

    # Training curves
    epochs = [h.epoch for h in history]
    mean_loss = [h.mean_loss for h in history]
    mean_s_pos = [h.mean_s_pos for h in history]
    mean_s_neg = [h.mean_s_neg for h in history]
    # plot_training_curves(epochs, mean_loss, mean_s_pos, mean_s_neg)

    print("\nFinal matrices (rounded):")
    print("\nWc (click embeddings):")
    print(model.Wc.round(4))
    print("\nWa (amenity projection):")
    print(model.Wa.round(4))
    print("\nWg (geo projection):")
    print(model.Wg.round(4))
    print("\nWe (enrichment projection):")
    print(model.We.round(4))
    print("\nWnce (output/context embeddings):")
    print(model.Wnce.round(4))


if __name__ == "__main__":
    main()
