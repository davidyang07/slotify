"""Phase 4 training dataset: eligibility, normalization, batching.

Nothing in this package computes a feature. Every number it hands to a model was
produced by the Phase 3 pipeline and cached on disk; the job here is to decide
*which* cached candidates may be trained on, project them onto a fixed column
layout, and group them by episode.
"""
