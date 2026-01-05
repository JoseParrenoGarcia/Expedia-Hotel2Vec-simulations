# data.py
from __future__ import annotations
from typing import List, Tuple


def generate_skipgram_pairs(sessions: List[List[int]], window_size: int) -> List[Tuple[int, int]]:
    """
    Convert click sessions into skip-gram (target, context) pairs.

    Example:
        session = [0, 1, 2], window_size = 1
        pairs = (0,1), (1,0), (1,2), (2,1)

    Args:
        sessions: list of sessions, each a list of hotel IDs (ints)
        window_size: how many positions left/right to treat as context

    Returns:
        List of (target_id, context_id) pairs.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    pairs: List[Tuple[int, int]] = []
    for session in sessions:
        n = len(session)
        for i in range(n):
            target = session[i]
            left = max(0, i - window_size)
            right = min(n - 1, i + window_size)

            for j in range(left, right + 1):
                if j == i:
                    continue
                context = session[j]
                pairs.append((target, context))

    return pairs

if __name__ == "__main__":
    # Simple test
    sessions = [
        [0, 1, 2],
        [3, 4, 5, 6]
    ]
    window_size = 1
    pairs = generate_skipgram_pairs(sessions, window_size)
    print(pairs)

    # Expected output:
    # [(0, 1), (1, 0), (1, 2), (2, 1), (3, 4), (4, 3), (4, 5), (5, 4), (5, 6), (6, 5)]