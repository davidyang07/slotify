# Training run artifacts

Each `<run_id>/` directory is one training run, content-addressed over its
config, its data and the code that produced it. Everything here is generated;
nothing is hand-edited.

Read `training_summary.json` before quoting any number from a run. Its
`evidence_class`, `data_provenance` and `label_source` fields say what the run
actually measured, and there are three very different kinds of run in this
directory.

## `gated-d8ed976101aa4c3b` — the committed bootstrap checkpoint

This is the only checkpoint whose `.pt` file is committed (see the exception in
`.gitignore`). It exists so the learned inference path works from a fresh clone
without a training run, which is what makes the demo reproducible.

**It is a weakly supervised distillation of `heuristic_offline_v1`.**

- `label_source`: `weak_heuristic`
- `data_provenance`: `weak_supervision`
- Targets: `heuristic_offline_v1`'s own score, binned into the 1–5 rubric within
  each episode by `slotify_rank.labelling.weak`.
- Training data: the corpus as it stood when it was produced -- 13 episodes,
  0.84 h, 260 train candidates over 11 episodes, one held-out validation
  episode. The corpus has since grown; this checkpoint was not retrained,
  because retraining a distillation of the baseline on more data would still be
  a distillation of the baseline.

What it proves: a 489,477-parameter multimodal PyTorch ranker consuming Whisper
speech representations, MiniLM transcript embeddings and 110 handcrafted
acoustic/structural features loads, scores real uploaded audio, and returns a
deterministic ranking through the product API.

What it does **not** prove: anything about ranking quality. Its teacher is the
baseline, so its validation NDCG@3 of 1.0 measures how faithfully it reproduces
that teacher — nothing more. `evaluation compare` refuses to publish a headline
improvement computed from it, and `/api/insert-sections` attaches a warning
saying so to every response it produces.

Regenerate it with:

```bash
cd ml
python -m slotify_rank.cli label weak
python -m slotify_rank.cli training run \
  --model-config configs/models/gated_v1.yaml \
  --labels ../data/labels/weak_labels_v1.jsonl \
  --allow-label-source weak_heuristic \
  --split-version v4
```

The committed checkpoint itself was produced against split v2, over the corpus
as it stood in August; v4 is the current split and is what a rerun would use.
Either way the result is a distillation of the baseline, which is why the
checkpoint was not regenerated when the split moved.

## `concat-*`, `handcrafted-*`, `gated-58e27d4507da3401` — synthetic smoke runs

Trained on `training synthesize` fixtures. `data_provenance` is
`synthetic_fixture` and `label_source` is `synthetic`. They demonstrate that the
training system runs end to end on every variant; their validation NDCG of ~1.0
is a property of the fixture, not a result. **Superseded** as anything but a
smoke record: they predate corpus v2 and split v4, and no current claim reads
them.

## What replaces all of this

A run whose `label_source` is `human`, from the labelling round the readiness
gate is waiting for:

```bash
cd ml
python -m slotify_rank.cli experiment train \
  --labels ../data/labels/labels_full-v2.jsonl \
  --split-version v4
```

That writes one directory per (variant, seed) cell and a matrix summary next to
them, and it refuses to nominate a run whose label source is anything but
`human`.

One thing then changes automatically. `scripts/lib/select-checkpoint.mjs` starts
serving the human-trained checkpoint instead of the bootstrap -- `npm run
demo` and `npm run preflight` both print which one they picked and its label
source.

Until then, the report says so, which is accurate.
