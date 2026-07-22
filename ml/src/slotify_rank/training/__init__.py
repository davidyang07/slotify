"""Phase 4 training: configuration, preparation, the loop, checkpoints, reports.

Deliberately a plain PyTorch training loop. No Lightning, no Hydra, no
experiment-tracking server: this trains a sub-million-parameter MLP on a few
thousand cached vectors, on a laptop CPU, and every one of those would add a
dependency whose failure modes are harder to debug than the fifty lines it
replaces.
"""
