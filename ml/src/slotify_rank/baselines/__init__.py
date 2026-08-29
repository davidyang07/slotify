"""Comparison systems the learned ranker is measured against.

Two of them, and they answer different questions:

``heuristic_offline_v1``
    (:mod:`slotify_rank.candidates.heuristic`) The deterministic signal scorer
    that shipped in the hackathon build, frozen. It is the *canonical* baseline:
    the denominator of the published improvement, credential-free and
    reproducible by anyone.

``classical_handcrafted_v1``
    (:mod:`slotify_rank.baselines.classical`) A scikit-learn gradient-boosted
    model over the handcrafted scalars alone, trained on the same human labels.
    It is not the published denominator; it exists to answer the question a
    reader should ask about a multimodal model -- "would a good tabular model on
    the cheap features have done just as well?" -- and it is reported alongside.
"""

from __future__ import annotations

__all__: list[str] = []
