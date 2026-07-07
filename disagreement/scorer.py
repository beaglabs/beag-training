import numpy as np


def compute_kl_divergence(
    student_logprobs: list[float],
    frontier_logprobs: list[float],
) -> float:
    """Per-token reverse KL: KL(student || frontier).

    Args:
        student_logprobs: Log probabilities from the student model [seq_len, vocab]
        frontier_logprobs: Log probabilities from the frontier model [seq_len, vocab]

    Returns:
        Average KL divergence across all tokens.
    """
    student_probs = np.exp(student_logprobs)
    kl = student_probs * (np.array(student_logprobs) - np.array(frontier_logprobs))
    return float(kl.sum())


def select_contested(
    examples: list[dict],
    max_count: int = 500,
    percentile: float = 0.05,
) -> list[dict]:
    """Return the top-N examples by KL divergence.

    These are the examples where the student model most disagreed
    with the frontier label — likely candidates for mislabeling or edge cases.

    Args:
        examples: List of examples with 'kl_divergence' key
        max_count: Hard cap on contested examples (default 500)
        percentile: Fraction of dataset to flag (default 5%)

    Returns:
        Subset of examples, sorted by most contested first.
    """
    sorted_examples = sorted(
        examples,
        key=lambda x: x.get("kl_divergence", 0),
        reverse=True,
    )
    count = min(max_count, max(1, int(len(sorted_examples) * percentile)))
    return sorted_examples[:count]
