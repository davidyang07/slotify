# The resume experiment

One experiment, defined in advance, to answer one question:

> Does the learned multimodal ranker place podcast ad breaks better than
> `heuristic_offline_v1`, measured as NDCG@3 on a held-out set of human
> judgements — and by how much?

This document is the operator's guide to running it. The claim it exists to
support or refute is written into
[`ml/configs/experiment_resume_v1.yaml`](../ml/configs/experiment_resume_v1.yaml),
and the verdict is written by
[`artifacts/reports/resume_evidence.md`](../artifacts/reports/resume_evidence.md),
which is generated, never edited.

---

## The four quantities that are never added together

Most of the honesty machinery in this repository exists to keep these apart.

| Quantity | What it is | Where it comes from |
| --- | --- | --- |
| **Generated candidates** | Timestamps the deterministic signal rules proposed. Not labels, not predictions, not evidence of anything but recall. | `candidates generate` → `artifacts/dataset/candidate_statistics.json` |
| **Weak labels** | Targets derived from `heuristic_offline_v1`'s own score. Useful for proving the training and serving paths work; **circular** as evidence, because the baseline is the teacher. | `label weak` → `data/labels/weak_labels_v1.jsonl` |
| **Human labels** | A person listened to a clip and rated it 1–5. The only supervised source this experiment accepts. | `label resume-experiment` → `data/labels/labels.sqlite3` |
| **Validation metrics** | Measured on the validation split during development. Model selection may use them. They are **not** the result. | `artifacts/training/*/training_summary.json` |
| **Held-out test metrics** | Measured once, at the end, on a split no development decision touched. This is the result. | `evaluation compare` → `artifacts/evaluation/*/comparison.json` |

A number quoted without saying which of these it is means nothing. The evidence
report labels every one.

---

## The protocol, fixed in advance

`ml/configs/experiment_resume_v1.yaml` is committed *before* the test split is
read, and `slotify-rank experiment manifest` resolves it into a manifest that
hashes the bytes it will run against — the split manifest, the candidate
manifest, the label snapshot, the feature schema, the model and baseline
configs.

Frozen once written:

- the split (`splits_v3.yaml`, grouped by **series**, seed 42, 70/15/15);
- the canonical baseline (`heuristic_offline_v1`);
- the primary metric (NDCG@3, relevance threshold 4.0, gain offset 1.0);
- checkpoint selection (validation NDCG@3) and seed selection (the **median**
  seed by validation NDCG@3, not the best — reporting the best of several seeds
  is seed cherry-picking with extra steps);
- the label gate (2,400 unique human labels) and the claim threshold (18 %).

Free to change during development, using train and validation only: model
architecture, features, and training hyperparameters. That is what a validation
split is for.

### Why the split groups on series

Two episodes of one show share hosts, room, mic chain, editing rhythm and
vocabulary. A model that saw one in training and the other in test would be
graded partly on memorisation. The unit of assignment is therefore the **show**,
never the episode and never the candidate — and series grouping implies episode
grouping, so the weaker guarantee comes free.

`experiment manifest` refuses to resolve when the split manifest on disk
disagrees with the declared `group_by` or seed, when the split is degraded
(everything in one `development` partition), or when the test groups appear in
another partition.

---

## Running it

### 1. Build the corpus — one command

```bash
cd ml
python -m slotify_rank.cli dataset prepare-resume-experiment
```

Acquire → probe → normalize → generate candidates → split → transcribe →
acoustic features → audio embeddings → text embeddings → assemble → validate →
statistics → labelling queue, then a **measured** readiness summary written to
`artifacts/experiments/prepare_report.json`.

Every stage is cache-aware, so re-running after an interruption costs only the
work that was actually lost. The feature stages are the long pole: Whisper
`tiny.en` transcribes at roughly 3.7× real time on a CPU, so a ten-hour corpus
is a couple of hours of transcription.

The corpus itself comes from a committed plan
([`ml/configs/corpus_resume_v1.yaml`](../ml/configs/corpus_resume_v1.yaml)) that
names shows, licences and per-show episode caps; `dataset discover` resolves it
into the committed source registry by reading the Internet Archive's public
metadata API. See [`docs/dataset-card.md`](dataset-card.md) for what is in it and
under what terms.

### 2. Label — the one step no command can do

```bash
cd ml
python -m slotify_rank.cli label resume-experiment
```

This pre-cuts every clip, reports how many labels remain, and opens the UI at
<http://127.0.0.1:8000/>. After it prints its readiness banner, the only
remaining work is human judgement.

The session is built for throughput:

- **one keystroke per item** — `1`–`5` rates *and* advances; there is no confirm
  step, because a confirm step doubles the keystrokes for a whole session;
- **the next items are already loaded** — the client holds a small window of
  upcoming items and their audio, so advancing is a repaint;
- **`S` skips**, which defers an item without judging it. A skip never reaches
  the export, the readiness gate or a model;
- **`U` marks a clip broken**, which *is* a judgement and is recorded as one;
- **progress reads `1234 / 2400`** against the experiment's target, not against
  everything ever generated;
- **the session resumes** — nothing is buffered client-side, so a closed tab
  loses at most the item on screen.

What the annotator never sees: the heuristic's score, the candidate's source
flags, or any marker distinguishing a first showing from a blind repeat.

Read [`docs/labelling-guide.md`](labelling-guide.md) before starting. The rubric
is a graded relevance scale: NDCG consumes `quality_score` directly, and the
equivalent 0–4 grade is `quality_score - 1`.

### 3. Freeze, train, evaluate

```bash
# Gate: does what exists support the experiment?
python -m slotify_rank.cli experiment readiness --split-version v3 --require-ready
python -m slotify_rank.cli label export --dataset-version resume-v1
python -m slotify_rank.cli experiment freeze --snapshot-version resume-v1 --split-version v3
python -m slotify_rank.cli experiment manifest --require-ready

# Five ablations x three seeds, on human labels only. Reports the MEDIAN seed
# by validation NDCG@3 and writes the whole matrix, so the spread is visible.
python -m slotify_rank.cli experiment train \
    --labels ../data/labels/labels_resume-v1.jsonl --split-version v3

# Once, at the end, on the frozen test split.
python -m slotify_rank.cli evaluation compare \
    --labels ../data/labels/labels_resume-v1.jsonl \
    --model ../artifacts/training/<run>/best_checkpoint.pt \
    --split test --split-version v3 \
    --experiment-config configs/experiment_resume_v1.yaml \
    --require-publishable
```

`evaluation compare` refuses to publish a headline unless the ground truth is
human, the model was trained on human labels, the baseline is the canonical one,
the split is `test`, no evaluation episode was trained on, and this project's
NDCG agrees with `sklearn.metrics.ndcg_score`. Each refusal is written into the
artifact with its reason.

### 4. Publish the evidence

```bash
npm run resume-evidence            # regenerate
npm run resume-evidence -- --check # CI: fail if the report has drifted
```

---

## What "supported" means here

The report has three states per claim, and the difference between the last two
is the point:

- **PASS** — an artifact establishes it.
- **FAIL** — an artifact was read and it does not.
- **NOT MEASURED** — nothing produced that number. This is *not* a pass.

The improvement threshold is read from the committed experiment definition. If
the measured improvement is below it, the report says FAIL and prints the
measured number. There is no code path in this repository that can turn a
measured 6 % into a claimed 18 %, and
`tests/test_resume_evidence.py::test_the_report_contains_no_hard_coded_resume_numbers`
asserts the threshold does not appear as a literal in the checker.

### The library claims

"The stack includes X" is not satisfied by a line in `pyproject.toml`. Each
library needs three independent things:

1. declared as a dependency (parsed from `pyproject.toml`, not grepped);
2. imported by at least one committed module (found by parsing the AST, so a
   mention in a docstring does not count);
3. recorded as having actually run, by a named field in a generated artifact.

For scikit-learn that third requirement is deliberately strict: a published
comparison must record **both** an independent NDCG cross-check that agreed
*and* a fitted classical baseline. Adding scikit-learn and never using it fails
this check, which is the entire reason the check is shaped this way.

---

## Where scikit-learn earns its place

Two jobs, both load-bearing, neither decorative:

**An independent implementation of the headline metric.** The result is a ratio
of two NDCG values from one implementation, so a bug in that implementation
moves numerator and denominator together and stays invisible to every test that
checks it against itself. `sklearn.metrics.ndcg_score` recomputes the
per-episode NDCG in `slotify_rank.evaluation.crosscheck`, and a disagreement
*blocks publication*. `tests/test_sklearn_integration.py` deliberately breaks
the project's NDCG and asserts the cross-check notices — a checker that agrees
with everything checks nothing.

**A classical comparison point.** `HistGradientBoostingRegressor` over the 110
handcrafted scalars and their missing mask, tuned by `GroupKFold` inside the
training split with the episode as the group. It answers the question a
sceptical reader should ask about any multimodal model: *would a good tabular
model on the cheap features have done just as well?* It is reported alongside
the headline and is never the denominator — that stays `heuristic_offline_v1`,
which needs no labels, no training and no dependencies, so anyone can reproduce
it.

---

## Uncertainty

NDCG@3 is macro-averaged over episodes and a test split holds a few dozen of
them, so the point estimate has real spread. `evaluation compare` bootstraps
over **episodes** — the unit the average is taken over — resampling both systems
on the same draw, because the two scores are paired observations of one episode.
The interval on the *improvement* is the one that matters: an improvement whose
interval spans zero is not a result, and the comparison says so in its caveats.

---

## Related documents

| Document | What it covers |
| --- | --- |
| [`dataset-card.md`](dataset-card.md) | The corpus, its licences and its limits |
| [`labelling-guide.md`](labelling-guide.md) | The 1–5 rubric an annotator applies |
| [`human-labelling-workflow.md`](human-labelling-workflow.md) | The labelling loop end to end |
| [`evaluation-evidence.md`](evaluation-evidence.md) | Every capability mapped to its artifact |
| [`model-training.md`](model-training.md) | The training system |
