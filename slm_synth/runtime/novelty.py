from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def canonical_exact(text: str) -> str:
    """Whitespace-only canonicalization used where exact surface equality matters."""
    return " ".join(text.split())


def shingles(text: str, n: int = 3) -> set[str]:
    tokens = normalize(text).split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {
        " ".join(tokens[index:index + n])
        for index in range(len(tokens) - n + 1)
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NoveltyFilter:
    """Scalable exact / near-duplicate filter from the finalized one-offs.

    Exact matches are O(1). Near-duplicate candidates come from an inverted
    3-token-shingle index. SequenceMatcher runs only against a bounded set of
    rows already sharing meaningful shingle content.
    """

    def __init__(
        self,
        *,
        jaccard_threshold: float,
        sequence_threshold: float,
    ):
        self.jaccard_threshold = jaccard_threshold
        self.sequence_threshold = sequence_threshold

        self._exact: dict[str, str] = {}
        self._norm_by_key: dict[str, str] = {}
        self._shingles_by_key: dict[str, set[str]] = {}
        self._shingle_index: dict[str, set[str]] = {}

        self.candidate_comparisons = 0
        self.sequence_comparisons = 0

    def add(self, key: str, text: str) -> None:
        norm = normalize(text)
        sh = shingles(text)

        self._exact.setdefault(norm, key)
        self._norm_by_key[key] = norm
        self._shingles_by_key[key] = sh

        for item in sh:
            self._shingle_index.setdefault(item, set()).add(key)

    def check(self, text: str) -> tuple[bool, str | None, dict[str, Any]]:
        norm = normalize(text)
        if not norm:
            return False, "empty", {}

        exact_key = self._exact.get(norm)
        if exact_key is not None:
            return False, "exact_duplicate", {"match": exact_key}

        sh = shingles(text)
        overlap_counts: dict[str, int] = {}
        for item in sh:
            for key in self._shingle_index.get(item, ()):
                overlap_counts[key] = overlap_counts.get(key, 0) + 1

        plausible: list[str] = []
        for key, overlap in overlap_counts.items():
            other_sh = self._shingles_by_key[key]
            min_overlap = (
                self.jaccard_threshold
                * (len(sh) + len(other_sh))
                / (1.0 + self.jaccard_threshold)
            )
            if overlap >= min_overlap:
                plausible.append(key)

        best_key = None
        best_j = 0.0
        best_s = 0.0

        for key in plausible:
            self.candidate_comparisons += 1
            jac = jaccard(sh, self._shingles_by_key[key])
            if jac > best_j:
                best_key = key
                best_j = jac

            if jac >= self.jaccard_threshold:
                return False, "near_duplicate", {
                    "match": key,
                    "jaccard": round(jac, 4),
                    "sequence_ratio": None,
                }

        if overlap_counts:
            ranked = sorted(
                overlap_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:32]
            min_shared = max(2, int(len(sh) * 0.20))

            for key, overlap in ranked:
                if overlap < min_shared:
                    break

                self.sequence_comparisons += 1
                seq = SequenceMatcher(
                    None,
                    norm,
                    self._norm_by_key[key],
                    autojunk=False,
                ).ratio()

                if seq > best_s:
                    best_key = key
                    best_s = seq

                if seq >= self.sequence_threshold:
                    return False, "near_duplicate", {
                        "match": key,
                        "jaccard": round(
                            jaccard(sh, self._shingles_by_key[key]),
                            4,
                        ),
                        "sequence_ratio": round(seq, 4),
                    }

        return True, None, {
            "nearest": best_key,
            "nearest_jaccard": round(best_j, 4),
            "nearest_sequence_ratio": round(best_s, 4),
        }
