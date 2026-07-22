"""Deterministic within-episode preference pairs.

Why pairs at all: the human label is a 1-5 naturalness score, but the product
question is "which of *these* breakpoints should we cut at". Regressing on the
absolute score would spend the model's capacity on annotator calibration -- one
person's 4 is another's 3 -- while the ordering within an episode is the part
annotators actually agree on. A pairwise objective learns exactly that ordering
and is invariant to the per-annotator offset.

Why never across episodes: a 4 in a dense news podcast and a 4 in a rambling
interview are not the same physical break, and an episode with twenty good
candidates would otherwise generate pairs against every weak candidate in every
other episode. Cross-episode pairs are both meaningless and numerically
dominant, so they are structurally impossible here: pairs are only ever formed
inside one episode's candidate list, and :func:`generate_pairs` refuses a mixed
split outright.

Determinism: candidates are sorted by id, enumerated in order, and any sampling
draws from a generator seeded with ``(seed, episode_id)``. The same corpus and
the same config produce byte-identical pairs, on any machine, in any order the
episodes happen to arrive.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from slotify_rank.config.versions import PAIR_GENERATION_VERSION
from slotify_rank.datasets.schema import TrainingExample

__all__ = [
    "PairConfig",
    "RankingPair",
    "PairSet",
    "generate_pairs",
    "index_pairs",
]

WeightScheme = Literal["uniform", "score_difference"]
SamplingStrategy = Literal["all", "capped", "balanced"]


@dataclass(frozen=True)
class PairConfig:
    """Everything that determines which pairs exist and how much each counts."""

    #: Minimum human-score gap for a pair to be formed. At the default of 1.0 an
    #: exact tie is excluded (a tie is not a preference), and so is any gap the
    #: rubric cannot really distinguish.
    minimum_score_difference: float = 1.0
    #: Hard cap per episode. Without it a 60-candidate episode contributes 1770
    #: pairs and a 6-candidate episode contributes 15, so the objective becomes
    #: "rank the longest episode well".
    max_pairs_per_episode: int | None = 64
    sampling_strategy: SamplingStrategy = "balanced"
    weight_scheme: WeightScheme = "uniform"
    #: >0 up-weights near-ties, which are the pairs the model is actually
    #: uncertain about. Applied as a multiplier, not as a resampling, so the
    #: pair *set* stays independent of the emphasis.
    hard_pair_emphasis: float = 0.0
    #: Swap which slot holds the preferred candidate, deterministically. Guards
    #: against a model or a bug that learns "the first input wins".
    shuffle_direction: bool = True
    seed: int = 42
    version: str = PAIR_GENERATION_VERSION

    def __post_init__(self) -> None:
        if self.minimum_score_difference <= 0:
            raise ValueError(
                "minimum_score_difference must be positive; a threshold of 0 would "
                "turn every tie into a pair with an arbitrary preferred side"
            )
        if self.max_pairs_per_episode is not None and self.max_pairs_per_episode < 1:
            raise ValueError("max_pairs_per_episode must be at least 1 when set")
        if self.hard_pair_emphasis < 0:
            raise ValueError("hard_pair_emphasis must be non-negative")
        if self.sampling_strategy not in ("all", "capped", "balanced"):
            raise ValueError(f"Unknown sampling_strategy {self.sampling_strategy!r}")
        if self.weight_scheme not in ("uniform", "score_difference"):
            raise ValueError(f"Unknown weight_scheme {self.weight_scheme!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_score_difference": self.minimum_score_difference,
            "max_pairs_per_episode": self.max_pairs_per_episode,
            "sampling_strategy": self.sampling_strategy,
            "weight_scheme": self.weight_scheme,
            "hard_pair_emphasis": self.hard_pair_emphasis,
            "shuffle_direction": self.shuffle_direction,
            "seed": self.seed,
            "version": self.version,
        }


@dataclass(frozen=True)
class RankingPair:
    """One preference. ``preferred`` always names the higher-scored candidate.

    ``left``/``right``/``target`` are the *model-facing* view: with direction
    shuffling on, the preferred candidate is sometimes the right-hand input and
    ``target`` is ``-1``. The semantic fields never move, so a report can always
    say which candidate the annotators preferred regardless of slot order.
    """

    pair_id: str
    episode_id: str
    preferred_candidate_id: str
    nonpreferred_candidate_id: str
    score_difference: float
    pair_weight: float
    left_candidate_id: str
    right_candidate_id: str
    target: int

    def __post_init__(self) -> None:
        if self.target not in (1, -1):
            raise ValueError(f"target must be +1 or -1, got {self.target}")
        if self.score_difference <= 0:
            raise ValueError(
                f"{self.pair_id}: score_difference must be positive; the preferred "
                "candidate is by definition the higher-scored one"
            )
        expected_left = (
            self.preferred_candidate_id if self.target == 1 else self.nonpreferred_candidate_id
        )
        if self.left_candidate_id != expected_left:
            raise ValueError(
                f"{self.pair_id}: target {self.target} says the left slot holds "
                f"{expected_left!r} but it holds {self.left_candidate_id!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "episode_id": self.episode_id,
            "preferred_candidate_id": self.preferred_candidate_id,
            "nonpreferred_candidate_id": self.nonpreferred_candidate_id,
            "score_difference": self.score_difference,
            "pair_weight": self.pair_weight,
            "left_candidate_id": self.left_candidate_id,
            "right_candidate_id": self.right_candidate_id,
            "target": self.target,
        }


@dataclass
class PairSet:
    """Generated pairs plus the statistics needed to judge them."""

    pairs: list[RankingPair] = field(default_factory=list)
    config: PairConfig = field(default_factory=PairConfig)
    split: str = ""
    episodes_without_pairs: list[str] = field(default_factory=list)
    pairs_per_episode: dict[str, int] = field(default_factory=dict)
    eligible_before_capping: int = 0

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def contributing_episode_count(self) -> int:
        return len({pair.episode_id for pair in self.pairs})

    def score_difference_histogram(self) -> dict[str, int]:
        """Pair counts bucketed by rounded score difference."""
        buckets = Counter(
            f"{np.floor(pair.score_difference * 2) / 2:.1f}" for pair in self.pairs
        )
        return dict(sorted(buckets.items(), key=lambda item: float(item[0])))

    def summary(self) -> dict[str, Any]:
        counts = list(self.pairs_per_episode.values())
        return {
            "split": self.split,
            "pair_count": len(self.pairs),
            "eligible_pair_count_before_capping": self.eligible_before_capping,
            "contributing_episode_count": self.contributing_episode_count,
            "episodes_without_pairs": list(self.episodes_without_pairs),
            "episodes_without_pairs_count": len(self.episodes_without_pairs),
            "pairs_per_episode": dict(sorted(self.pairs_per_episode.items())),
            "pairs_per_episode_min": min(counts) if counts else 0,
            "pairs_per_episode_max": max(counts) if counts else 0,
            "pairs_per_episode_mean": (sum(counts) / len(counts)) if counts else 0.0,
            "score_difference_histogram": self.score_difference_histogram(),
            "config": self.config.to_dict(),
        }

    def to_rows(self) -> list[dict[str, Any]]:
        return [pair.to_dict() for pair in self.pairs]


def _pair_id(episode_id: str, preferred: str, nonpreferred: str) -> str:
    """Content-addressed, so the same preference always has the same id."""
    # A delimiter that cannot occur in an id, so ("ab", "c") and ("a", "bc")
    # can never hash to the same pair.
    digest = hashlib.sha256(
        "\x1f".join((episode_id, preferred, nonpreferred)).encode("utf-8")
    ).hexdigest()
    return f"pair-{digest[:16]}"


def _episode_generator(seed: int, episode_id: str) -> np.random.Generator:
    """Per-episode RNG, so an episode's sample never depends on episode order."""
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def generate_pairs(
    examples: Sequence[TrainingExample],
    config: PairConfig | None = None,
    split: str = "",
) -> PairSet:
    """Build every within-episode preference pair permitted by ``config``.

    ``examples`` must all belong to one split. That is checked rather than
    assumed: a pair spanning train and validation is a leak that no downstream
    metric can detect.
    """
    config = config or PairConfig()
    result = PairSet(config=config, split=split)
    if not examples:
        return result

    splits = sorted({example.split for example in examples})
    if len(splits) > 1:
        raise ValueError(
            f"Refusing to generate pairs across splits {splits}. Pairs must be "
            "generated per split, or a preference will straddle the train/"
            "validation boundary."
        )
    result.split = result.split or splits[0]

    grouped: dict[str, list[TrainingExample]] = {}
    for example in examples:
        grouped.setdefault(example.episode_id, []).append(example)

    for episode_id in sorted(grouped):
        members = sorted(grouped[episode_id], key=lambda e: e.candidate_id)
        candidates = _eligible_pairs(episode_id, members, config)
        result.eligible_before_capping += len(candidates)
        selected = _select(candidates, config, episode_id)
        if not selected:
            result.episodes_without_pairs.append(episode_id)
            continue
        result.pairs_per_episode[episode_id] = len(selected)
        result.pairs.extend(selected)

    return result


def _eligible_pairs(
    episode_id: str, members: Sequence[TrainingExample], config: PairConfig
) -> list[RankingPair]:
    """Every unordered candidate pair whose score gap clears the threshold."""
    pairs: list[RankingPair] = []
    seen: set[tuple[str, str]] = set()
    for index, left in enumerate(members):
        for right in members[index + 1 :]:
            difference = left.quality_score - right.quality_score
            magnitude = abs(difference)
            if magnitude < config.minimum_score_difference:
                continue
            if difference > 0:
                preferred, nonpreferred = left, right
            else:
                preferred, nonpreferred = right, left
            key = (preferred.candidate_id, nonpreferred.candidate_id)
            if key in seen:
                # Unreachable for a well-formed input -- the loader rejects
                # duplicate candidate ids -- but a silent duplicate pair would
                # double one preference's weight, so it is refused here too.
                continue
            seen.add(key)
            pairs.append(
                _build_pair(episode_id, preferred, nonpreferred, magnitude, config)
            )
    return pairs


def _build_pair(
    episode_id: str,
    preferred: TrainingExample,
    nonpreferred: TrainingExample,
    magnitude: float,
    config: PairConfig,
) -> RankingPair:
    pair_id = _pair_id(episode_id, preferred.candidate_id, nonpreferred.candidate_id)

    weight = 1.0 if config.weight_scheme == "uniform" else float(magnitude)
    if config.hard_pair_emphasis:
        excess = magnitude - config.minimum_score_difference
        weight *= float(np.exp(-config.hard_pair_emphasis * excess))

    if config.shuffle_direction:
        # Derived from the pair id, so the slot assignment is a property of the
        # pair itself: stable across runs, across shuffles and across resumes.
        flip = int(pair_id[-1], 16) % 2 == 1
    else:
        flip = False

    if flip:
        left, right, target = nonpreferred.candidate_id, preferred.candidate_id, -1
    else:
        left, right, target = preferred.candidate_id, nonpreferred.candidate_id, 1

    return RankingPair(
        pair_id=pair_id,
        episode_id=episode_id,
        preferred_candidate_id=preferred.candidate_id,
        nonpreferred_candidate_id=nonpreferred.candidate_id,
        score_difference=float(magnitude),
        pair_weight=float(weight),
        left_candidate_id=left,
        right_candidate_id=right,
        target=target,
    )


def _select(
    pairs: Sequence[RankingPair], config: PairConfig, episode_id: str
) -> list[RankingPair]:
    """Apply the per-episode cap using the configured strategy."""
    ordered = sorted(pairs, key=lambda pair: pair.pair_id)
    cap = config.max_pairs_per_episode
    if config.sampling_strategy == "all" or cap is None or len(ordered) <= cap:
        return ordered

    rng = _episode_generator(config.seed, episode_id)

    if config.sampling_strategy == "capped":
        chosen = rng.choice(len(ordered), size=cap, replace=False)
        return sorted((ordered[index] for index in chosen), key=lambda p: p.pair_id)

    # Balanced: fill round-robin across score-difference buckets so a cap does
    # not silently keep only the easy, widely-separated pairs.
    buckets: dict[str, list[RankingPair]] = {}
    for pair in ordered:
        key = f"{np.floor(pair.score_difference * 2) / 2:.1f}"
        buckets.setdefault(key, []).append(pair)
    for key in buckets:
        order = rng.permutation(len(buckets[key]))
        buckets[key] = [buckets[key][index] for index in order]

    selected: list[RankingPair] = []
    bucket_keys = sorted(buckets, key=float)
    position = 0
    while len(selected) < cap:
        progressed = False
        for key in bucket_keys:
            if position < len(buckets[key]):
                selected.append(buckets[key][position])
                progressed = True
                if len(selected) == cap:
                    break
        if not progressed:
            break
        position += 1
    return sorted(selected, key=lambda pair: pair.pair_id)


def index_pairs(pair_set: PairSet) -> list[dict[str, Any]]:
    """Render pairs as the mappings :class:`PairDataset` consumes."""
    return [
        {
            "left_candidate_id": pair.left_candidate_id,
            "right_candidate_id": pair.right_candidate_id,
            "target": pair.target,
            "pair_weight": pair.pair_weight,
            "episode_id": pair.episode_id,
        }
        for pair in pair_set.pairs
    ]
