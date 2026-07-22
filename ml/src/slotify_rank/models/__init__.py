"""Ranking model variants behind one shared interface.

Every variant takes the same inputs and returns the same
:class:`~slotify_rank.models.base.RankerOutput`, so the trainer, the checkpoint
format and the evaluation path never branch on which model is being run. What
differs between them is only which modalities they are allowed to look at --
which is exactly what the Phase 5 ablation needs to compare.
"""
