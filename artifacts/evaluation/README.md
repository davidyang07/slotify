# Evaluation artifacts

Each `eval-<id>/` directory is one run of `slotify-rank evaluation compare`,
content-addressed over its inputs. Everything here is generated; nothing is
hand-edited.

**Read `comparison.json` before quoting any number.** Its `headline_publishable`
flag and `blocking_reasons` list say whether the run measured anything that may
be reported, and a run may be present here and still be worth nothing as
evidence — that is the point of keeping it. A comparison that ran, produced real
numbers, and was refused publication is more auditable than one that was never
run.

## `eval-5fdb9c01c499f43f` — SUPERSEDED, never publishable

The only comparison in this repository. It is **not a result**, and two
independent things disqualify it:

- **circular ground truth.** `label_source` is `weak_heuristic`: the labels were
  derived from `heuristic_offline_v1`'s own score, and the baseline it is
  compared against is that same scorer. The baseline is the teacher.
- **a distilled model.** The evaluated checkpoint was trained on those same weak
  labels, so its ordering reflects its teacher rather than human judgement.

Both reasons are written into `comparison.json`; `--require-publishable` exits
non-zero on it, and the evidence reports list it under "comparisons that ran but
may not be published" rather than as a measurement.

It is also **superseded by the corpus it was run against**. It used split `v2`
over 1 episode and 268 candidates, before corpus v2 and split v4 existed. Its
NDCG@3 of 1.0 for both systems is what a single-episode comparison against its
own teacher produces; it says nothing about ranking quality.

It is kept because deleting it would remove the record that the evaluator has
actually been run end to end on real artifacts, and because the refusal it
triggered is itself the evidence that the publication gate works.

## What replaces it

One run of `evaluation compare` whose `label_source` is `human`, against split
`v4`, using a checkpoint from the `experiment train` matrix, with
`--require-publishable`. Until that exists, every report in this repository says
the held-out improvement is **NOT MEASURED**, which is accurate.
