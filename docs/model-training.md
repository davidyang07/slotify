# Model training (Phase 4)

This is the CPU-first PyTorch learning-to-rank system that consumes the Phase 3
multimodal feature artifacts and trains ranking models over candidates grouped
by episode. It is the implementation behind the ranking capability:

> A multimodal PyTorch learning-to-rank model combines waveform features, speech
> representations and transcript embeddings to identify natural podcast ad
> breaks.

**What Phase 4 is.** The training machinery: dataset loading, eligibility
checking, train-only normalization, within-episode pair generation, five model
variants behind one interface, a pairwise ranking objective plus an auxiliary
acceptability head, episode-grouped validation, checkpointing, and resume.

**What Phase 4 is not.** It is not the final model comparison, the ablation
matrix, the human-preference study, or the editing-time benchmark — those are
Phase 5, and they require real human labels. **No Phase 4 metric is evidence of
model quality.** The smoke runs train on deterministic synthetic fixtures (see
[Smoke training](#smoke-training)); a validation NDCG of 1.0 on that data means
the loop learns a signal that was planted to be learnable, nothing more.

---

## Why these design choices

**Pairwise ranking, not score regression.** The human label is a 1–5 naturalness
score, but the product question is "which of *these* breakpoints should we cut
at". Regressing the absolute score spends model capacity on annotator
calibration — one person's 4 is another's 3 — while the *ordering within an
episode* is what annotators actually agree on. A pairwise margin objective learns
that ordering and is invariant to each annotator's offset.

**Everything is within-episode.** A "4" in a dense news podcast and a "4" in a
rambling interview are not the same physical break. Pairs are only ever formed
inside one episode of one split, and validation NDCG is computed per episode and
macro-averaged — never over one pooled global list, which would mostly measure
how episodes differ in average label.

**Train-only normalization.** Scalar features are standardized using statistics
fitted on the training split alone. Fitting over the whole corpus is the most
common way a small-dataset result turns out optimistic: the validation episodes'
means and variances leak into every training input, invisibly.

**No model recomputation.** Training reads the cached `.npy` embedding arrays the
Phase 3 pipeline already wrote. It imports neither Whisper nor MiniLM. This is
both a speed property (preparation takes seconds) and a correctness one — the
vectors a checkpoint's schema describes are exactly the ones on disk.

---

## Training data requirements

A candidate becomes a training example only if it satisfies **every** integrity
condition. The loader counts every candidate under exactly one reason, so a run
that reports "48 examples" against 80 candidates can account for the other 32:

```
eligible               usable as a training example
missing_label          no human label (supervised training requires one)
missing_features       an available modality's array was unreadable
stale_features         feature/spec version disagreement, or wrong dimension
invalid_features       NaN/inf, a failed Phase 3 record, or an all-zero "available" vector
synthetic_excluded     synthetic product padding
ineligible_candidate   eligible_for_labelling / eligible_for_evaluation is false
split_mismatch         unassigned split, or disagreement with the split manifest
unsupported_version    feature pipeline/spec version the run does not accept
```

Minimum for a real supervised run: at least two training episodes, one
validation episode, and enough non-tied within-episode pairs to learn from.

### Label requirements

Human labels are the **only** default supervised source. Weak or heuristic
labels are never loaded — the loader raises if a row carries
`label_source != "human"`. Multiple annotators per candidate are aggregated
deterministically: quality score by mean, acceptability by majority vote (ties
broken by the mean against the exported threshold), and any `is_unusable` vote
removes the candidate.

The label export's `.meta.json` sidecar is required: it carries the
acceptability threshold in force when the labels were exported, so the auxiliary
targets are interpreted the same way they were labelled.

### Feature compatibility

Embedding dimensions are read from the artifacts, never hard-coded. Whisper
`tiny.en` and MiniLM both produce 384-dimensional vectors; every loader verifies
the stored dimension against what it expects and refuses a mismatch. The
constructed model inputs are:

| Modality | Dimension | Construction |
|---|---|---|
| handcrafted | 110 (from the manifest header) | normalized scalars + missing masks |
| audio | 4 × 384 = 1536 | stored `before`, `after`, `context`, `difference` |
| text | 4 × 384 = 1536 | stored `before`, `after`, plus their `difference` and elementwise `product` (arithmetic on cached MiniLM output — **not** a model re-run) |

---

## Scalar normalization

Fitted on the training split, applied to all splits, saved as `normalizer.json`.
Three cases are handled explicitly:

- **Missing values** are excluded from the fit (a masked feature carries a
  sentinel, not a measurement) and set to exactly zero after transform.
- **Binary indicators** (`sentence_end`, the `source_*` one-hots) pass through
  unscaled — rescaling a 0/1 flag by a sample standard deviation turns it into
  two arbitrary reals and destroys the "absence is zero" property the masks rely
  on.
- **Constant features** are centered but never scaled (divisor forced to 1.0), so
  a feature that carries no information cannot be amplified by `1/ε` into
  dominating the first layer. An epsilon floor of `1e-6` decides "constant".

The artifact records the feature names, means, standard deviations, the constant
and binary masks, the fit episode/candidate counts, the training-split hash, and
the feature-pipeline/spec versions.

---

## Pair generation

Deterministic, within-episode, per-split. For candidates *a* and *b* a pair is
formed when their human quality scores differ by at least
`minimum_score_difference` (default 1.0 — an exact tie is not a preference). The
higher-scored candidate is `preferred`. Each pair carries a content-addressed id,
the score difference, and a weight.

- **No cross-episode pairs, no cross-split pairs** — the latter is refused
  outright, since a preference straddling the train/validation boundary is a leak
  no metric can detect.
- **Per-episode cap** (`max_pairs_per_episode`, default 64) with balanced
  sampling across score-difference buckets, so a 60-candidate episode does not
  dominate a 6-candidate one.
- **Direction shuffling** places the preferred candidate in the left or right
  slot deterministically (derived from the pair id), guarding against a model
  that learns "the first input wins". The semantic fields never move, so a report
  always knows which candidate was preferred.

---

## Model variants

All five share one interface (`RankerOutput`: a ranking score and an optional
acceptability logit) and one registry, so the trainer, the checkpoint format and
the validation loop never branch on which model they hold. They form a ladder so
an ablation can attribute a gain to a specific modality:

| Variant | Sees | Parameters | Purpose |
|---|---|---|---|
| `handcrafted` | scalars + masks | 45,570 | learned non-embedding baseline |
| `text_only` | transcript representation | 214,146 | semantic signal in isolation |
| `audio_only` | Whisper representation | 214,146 | `audio_embedding_only` experiment |
| `audio_plus_acoustic` | Whisper + scalars | 259,074 | the other named audio experiment |
| `concat` | all three, concatenated | 472,578 | multimodal fusion by concatenation |
| `gated` | all three, gated | 489,477 | **primary** — fusion by learned gates |

Every projection uses LayerNorm (not BatchNorm — a masked modality would
otherwise let one candidate's *availability* shift another's activations), GELU,
and dropout. A masked modality is multiplied by its availability flag after the
non-linearity, so an unavailable modality contributes exactly zero rather than
the bias term's image.

### Gated fusion

The primary architecture. Each modality is projected to a common width and the
three are combined as a weighted sum whose weights the network computes from the
projected states **and** the availability flags. An unavailable modality's gate
logit is set to `-inf` before the softmax, so it receives exactly zero weight
while the remaining gates still sum to one. That renormalization matters: without
it, a candidate missing its transcript would get a systematically smaller fused
representation than a complete one, and the model would learn to rank on
transcript availability rather than on content. Gate values are retained on the
output for inspection.

This is deliberately **not** attention — with three modalities and a few hundred
labelled candidates, a query/key/value block adds parameters and instability to
express what three scalars already express.

---

## Ranking loss and auxiliary head

**Ranking loss** is `torch.nn.MarginRankingLoss` with a default margin of 0.2.
`target = +1` means "the left candidate should score higher than the right one",
so the loss falls to zero once `left ≥ right + margin`. A margin of 0 is
satisfied by an infinitesimal difference and lets the scores collapse together;
0.2 asks for a visible separation. Pair weights are supported.

**Auxiliary acceptability** is binary cross-entropy with logits over the batch's
**unique** candidates — not its pair slots. A strong candidate appears in many
pairs; running the auxiliary loss over the batch as-drawn would weight it by how
many pairs it happens to be in, which correlates with its label and distorts the
class distribution the head trains on. The collate function isolates the unique
candidates for exactly this reason.

Combined objective:

```
total_loss = ranking_loss + auxiliary_weight * acceptability_loss   # default auxiliary_weight = 0.25
```

The auxiliary head can be disabled in configuration; the interface stays stable.
Optional class weighting is derived from the training split only.

---

## Checkpoints

Saved as `state_dict`s and JSON primitives — **no arbitrary Python objects** —
and always loaded with `weights_only=True`. A checkpoint that has to be *trusted*
before it can be read is a remote-code-execution primitive with a `.pt`
extension, and model files get shared around.

Each checkpoint records the model variant, the input schema, the feature
ordering, the feature-pipeline versions, the normalizer identity, the split
hash, the seed, the resolved device/dtype, the dependency versions, the git
commit, and the run id. Loading **rejects** an incompatible variant, input
dimension, feature ordering, feature-pipeline version, normalizer version, or
split hash (the last overridable for inference-only use on a new corpus). Writes
go through the Phase 2 Windows/OneDrive-safe atomic rename, so a crash leaves the
previous file or the new one, never half of either.

All `.pt` files are git-ignored.

## Resume behaviour

Resuming restores the model, optimizer, epoch, global step, early-stopping state,
best metric, and the torch and DataLoader RNG states — so a resumed run draws the
same batches and applies the same dropout as an uninterrupted one from that point
on. `test_interrupted_then_resumed_matches_an_uninterrupted_run` asserts the two
agree within `1e-4`.

Bit-identity is **not** claimed. Floating-point reduction order in BLAS kernels
is not seed-controlled, so two mathematically equal sequences can differ in the
last few bits and compound over epochs. `1e-4` is far below any difference that
would change a ranking decision. Results are reproducible on the same machine and
torch build; they are not guaranteed across torch versions, BLAS thread counts,
or a CUDA device.

---

## Compute policy

CPU-first. `device: auto` selects CUDA only if torch actually reports a usable
one and falls back to CPU otherwise; an explicit `cuda` request on a machine
without CUDA is an error, not a silent eight-hour CPU run. Mixed precision is
*offered* but never required, and is refused on CPU (float16 has no accelerated
CPU kernels and would be slower than float32). Thread count, worker count and a
small-memory mode are configurable. The device and resolved dtype are recorded in
`environment.json`, and approximate throughput (pairs/second) is reported per
epoch.

The primary gated ranker is ~490k parameters — small enough to train in minutes
on a modest CPU dataset.

---

## CPU training commands

From `ml/`, with the `features` extra installed (`pip install -e .[features,dev]`
or the project's `.venv`):

```bash
# Inspect the model registry.
python -m slotify_rank.cli models list
python -m slotify_rank.cli models describe --model gated

# Prepare a dataset and see the eligibility accounting (no training).
python -m slotify_rank.cli training prepare --labels data/labels/labels_v1.jsonl

# Generate and inspect the within-episode pairs.
python -m slotify_rank.cli training pairs --labels data/labels/labels_v1.jsonl --pairs-output pairs.jsonl

# Train the gated multimodal ranker on CPU.
python -m slotify_rank.cli training run \
    --labels data/labels/labels_v1.jsonl \
    --model-config configs/models/gated_v1.yaml

# Validate, inspect and resume.
python -m slotify_rank.cli training validate --labels data/labels/labels_v1.jsonl --checkpoint artifacts/training/<run_id>/best_checkpoint.pt
python -m slotify_rank.cli training inspect  --checkpoint artifacts/training/<run_id>/best_checkpoint.pt
python -m slotify_rank.cli training resume   --labels data/labels/labels_v1.jsonl --model-config configs/models/gated_v1.yaml --run-dir artifacts/training/<run_id> --checkpoint artifacts/training/<run_id>/last_checkpoint.pt --epochs 30
```

---

## Smoke training

Until enough real human labels exist, the training path is exercised on
deterministic **synthetic fixtures**. These are generated numbers, not real audio
and not human labels. Every synthetic artifact carries `"synthetic": true` and a
`label_source` of `synthetic`, and the training summaries say so in their first
field and their title. A model *can* fit them because the latent target is a
known function of a few features plus seeded noise — which is exactly what makes
the smoke run able to fail (a loop that cannot learn a planted signal is broken).

```bash
# Write a synthetic corpus INSIDE the checkout (manifest paths are repo-relative).
python -m slotify_rank.cli training synthesize --data-root ../data-synthetic --episodes 10 --candidates-per-episode 8

# Train the gated model on it. --smoke is a recorded shorthand for small caps.
python -m slotify_rank.cli training run \
    --data-root ../data-synthetic \
    --labels ../data-synthetic/labels/labels_synthetic.jsonl \
    --model-config configs/models/gated_v1.yaml --smoke
```

`--smoke` sets a small epoch budget and small data caps. It is **not** a hidden
mode: every value it changes is written to the run's `resolved_config.json` under
`overrides` and reproduced in the Markdown summary.

The committed synthetic smoke evidence lives at
`artifacts/training/{handcrafted,concat,gated}-*/` (reports committed;
checkpoints git-ignored). See also the
[evidence matrix](evaluation-evidence.md) Phase 4 section.

---

## Real training (once 200–300 human labels exist)

The commands are identical; only the label source changes. Once
`data/labels/labels_v1.jsonl` holds 200–300+ human labels spanning at least a few
series:

```bash
# 1. Export the human labels (Phase 2 labelling loop).
python -m slotify_rank.cli label export

# 2. Regenerate the split now that there are enough series to split (Phase 2).
python -m slotify_rank.cli dataset split --config configs/splits_v1.yaml
python -m slotify_rank.cli dataset validate --deep

# 3. Confirm the features are current (Phase 3), recomputing nothing already cached.
python -m slotify_rank.cli pipeline features

# 4. Train each variant. Real labels, real splits, no --smoke, full epoch budget.
for m in handcrafted text_only audio_only concat gated; do
  python -m slotify_rank.cli training run \
      --labels data/labels/labels_v1.jsonl \
      --model-config configs/models/${m}_v1.yaml
done

# 5. Validate the best gated checkpoint on the held-out split.
python -m slotify_rank.cli training validate \
    --labels data/labels/labels_v1.jsonl \
    --checkpoint artifacts/training/<gated_run_id>/best_checkpoint.pt \
    --output artifacts/training/<gated_run_id>/validation_real.json
```

The resulting `training_summary.json` files carry `data_provenance: real` and
`label_source: human`, and their metrics are the first that may be quoted — but
the **comparison against `heuristic_offline_v1`, the ablation matrix, and the
headline NDCG improvement all belong to Phase 5**, not here.

## How Phase 4 differs from Phase 5

| | Phase 4 (this system) | Phase 5 (not built) |
|---|---|---|
| Goal | prove the training machinery works | measure whether the model is good |
| Data | synthetic fixtures (+ real forward pass) | real human labels only |
| Output | trainable checkpoints, reports | baseline-vs-model comparison, ablations |
| Metrics | smoke NDCG — **not** quality evidence | headline NDCG@3 improvement with CIs |
| Supports | "a trainable multimodal ranker exists" | "it outperforms the heuristic by X %" |

A synthetic smoke NDCG of 1.0 is not a result. It says the optimizer, the loss,
the metric and the checkpointing are wired correctly on data where the answer is
known. Quoting it as model quality would be exactly the error the whole
evidence matrix exists to prevent.
