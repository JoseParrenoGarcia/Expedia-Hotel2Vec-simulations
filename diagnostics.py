# diagnostics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


def compute_score_matrix(Wc: np.ndarray, Wnce: np.ndarray) -> np.ndarray:
    """
    Compute a full score matrix S where:
        S[target, context] = Wc[target] dot Wnce[context]

    Shapes:
        Wc   : (V, d)
        Wnce : (V, d)
        S    : (V, V)
    """
    if Wc.ndim != 2 or Wnce.ndim != 2:
        raise ValueError("Wc and Wnce must be 2D arrays.")
    if Wc.shape[1] != Wnce.shape[1]:
        raise ValueError("Embedding dimensions must match: Wc.shape[1] == Wnce.shape[1].")
    return Wc @ Wnce.T


@dataclass
class TrainingSnapshot:
    """
    Stores model state at the end of an epoch.
    Keeping it explicit makes it easy to extend later (amenities/geo).
    """
    epoch: int
    Wc: np.ndarray
    Wnce: np.ndarray


def collect_snapshots(
    epoch: int,
    Wc: np.ndarray,
    Wnce: np.ndarray,
    store: List[TrainingSnapshot],
    copy_arrays: bool = True,
) -> None:
    """
    Append a snapshot to a list.

    Args:
        epoch: epoch number
        Wc, Wnce: current matrices
        store: list to append into
        copy_arrays: if True, store copies (safe). If False, stores references (unsafe).
    """
    if copy_arrays:
        store.append(TrainingSnapshot(epoch=epoch, Wc=Wc.copy(), Wnce=Wnce.copy()))
    else:
        store.append(TrainingSnapshot(epoch=epoch, Wc=Wc, Wnce=Wnce))


def save_training_artifacts(
    filepath: str,
    history: List[object],
    snapshots: Optional[List[TrainingSnapshot]] = None,
) -> None:
    """
    Save training history and optional snapshots to a .npz file.

    - history is assumed to be a list of TrainStats-like objects with attributes:
        epoch, mean_loss, mean_s_pos, mean_s_neg

    - snapshots stores Wc/Wnce per epoch (optional)

    This makes it easy to load later and re-plot without retraining.
    """
    epochs = np.array([h.epoch for h in history], dtype=int)
    mean_loss = np.array([h.mean_loss for h in history], dtype=float)
    mean_s_pos = np.array([h.mean_s_pos for h in history], dtype=float)
    mean_s_neg = np.array([h.mean_s_neg for h in history], dtype=float)

    payload: Dict[str, np.ndarray] = {
        "epochs": epochs,
        "mean_loss": mean_loss,
        "mean_s_pos": mean_s_pos,
        "mean_s_neg": mean_s_neg,
    }

    if snapshots is not None and len(snapshots) > 0:
        snap_epochs = np.array([s.epoch for s in snapshots], dtype=int)
        Wc_stack = np.stack([s.Wc for s in snapshots], axis=0)      # (E, V, d)
        Wnce_stack = np.stack([s.Wnce for s in snapshots], axis=0)  # (E, V, d)

        payload["snap_epochs"] = snap_epochs
        payload["Wc_stack"] = Wc_stack
        payload["Wnce_stack"] = Wnce_stack

    np.savez_compressed(filepath, **payload)
