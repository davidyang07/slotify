"""Model variants: shapes, parameter counts, gating and failure modes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from slotify_rank.config.settings import find_repo_root
from slotify_rank.datasets.schema import DatasetSchema
from slotify_rank.models.registry import (
    MODEL_REGISTRY,
    build_model,
    describe_variant,
    list_models,
)
from slotify_rank.models.schema import ModelConfig, load_model_config

HANDCRAFTED_DIMENSION = 110
NATIVE = 384


def make_schema(handcrafted: int = HANDCRAFTED_DIMENSION) -> DatasetSchema:
    return DatasetSchema(
        handcrafted_feature_names=tuple(f"f{index}" for index in range(handcrafted)),
        handcrafted_dimension=handcrafted,
        audio_dimension=NATIVE * 4,
        text_dimension=NATIVE * 4,
        native_embedding_dimension=NATIVE,
    )


def make_batch(batch_size: int = 5, schema: DatasetSchema | None = None):
    schema = schema or make_schema()
    generator = torch.Generator().manual_seed(0)
    return {
        "handcrafted": torch.randn(
            batch_size, schema.handcrafted_dimension, generator=generator
        ),
        "handcrafted_missing_mask": torch.zeros(
            batch_size, schema.handcrafted_dimension
        ),
        "audio": torch.randn(batch_size, schema.audio_dimension, generator=generator),
        "audio_available": torch.ones(batch_size),
        "text": torch.randn(batch_size, schema.text_dimension, generator=generator),
        "text_available": torch.ones(batch_size),
    }


def config_for(variant: str, **overrides) -> ModelConfig:
    return ModelConfig(variant=variant, **overrides)


@pytest.mark.parametrize("variant", sorted(MODEL_REGISTRY))
def test_every_variant_forward_pass_has_the_right_shape(variant):
    schema = make_schema()
    model = build_model(config_for(variant), schema)
    output = model(**make_batch(5, schema))
    assert output.ranking_score.shape == (5,)
    assert output.acceptability_logit.shape == (5,)
    assert torch.isfinite(output.ranking_score).all()
    assert torch.isfinite(output.acceptability_logit).all()


@pytest.mark.parametrize("variant", sorted(MODEL_REGISTRY))
def test_every_variant_shares_the_interface(variant):
    model = build_model(config_for(variant), make_schema())
    output = model(**make_batch())
    assert hasattr(output, "ranking_score")
    assert hasattr(output, "acceptability_logit")
    assert hasattr(output, "gates")
    assert model.variant == variant


@pytest.mark.parametrize("variant", sorted(MODEL_REGISTRY))
def test_auxiliary_head_can_be_disabled(variant):
    model = build_model(config_for(variant, auxiliary_head=False), make_schema())
    output = model(**make_batch())
    assert output.acceptability_logit is None
    assert output.ranking_score.shape == (5,)


@pytest.mark.parametrize("variant", sorted(MODEL_REGISTRY))
def test_gradients_reach_every_parameter(variant):
    model = build_model(config_for(variant), make_schema())
    output = model(**make_batch())
    # Both heads, since the acceptability head is deliberately unreachable from
    # the ranking score alone.
    (output.ranking_score.sum() + output.acceptability_logit.sum()).backward()
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
    ]
    assert not unused, f"parameters with no finite gradient: {unused[:5]}"


@pytest.mark.parametrize("variant", sorted(MODEL_REGISTRY))
def test_parameter_counts_stay_under_one_million(variant):
    model = build_model(config_for(variant), make_schema())
    count = model.parameter_count()
    assert 0 < count < 1_000_000, f"{variant} has {count} parameters"


def test_wrong_input_dimension_fails_loudly():
    model = build_model(config_for("concat"), make_schema())
    batch = make_batch()
    batch["audio"] = torch.randn(5, 999)
    with pytest.raises(ValueError, match="Expected 1536 input features"):
        model(**batch)


def test_a_config_for_another_variant_is_rejected():
    from slotify_rank.models.variants import GatedRanker

    with pytest.raises(ValueError, match="variant"):
        GatedRanker(config_for("concat"), make_schema())


def test_unknown_variant_is_rejected():
    with pytest.raises(KeyError, match="Unknown model variant"):
        build_model(config_for("transformer_xl"), make_schema())


# ---------------------------------------------------------------------------
# Missing modalities
# ---------------------------------------------------------------------------


def test_unavailable_modality_receives_exactly_zero_gate_weight():
    model = build_model(config_for("gated"), make_schema()).eval()
    batch = make_batch(4)
    batch["text_available"] = torch.zeros(4)
    with torch.no_grad():
        output = model(**batch)
    gates = output.gates
    assert gates.shape == (4, 3)
    assert torch.all(gates[:, 2] == 0.0)


def test_gates_are_finite_and_normalized():
    model = build_model(config_for("gated"), make_schema()).eval()
    batch = make_batch(6)
    batch["text_available"] = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    batch["audio_available"] = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    with torch.no_grad():
        output = model(**batch)
    assert torch.isfinite(output.gates).all()
    assert torch.allclose(output.gates.sum(dim=-1), torch.ones(6), atol=1e-5)
    assert (output.gates >= 0).all()


def test_a_masked_modality_cannot_change_the_score():
    """Perturbing an unavailable modality's vector must not move the output."""
    model = build_model(config_for("gated"), make_schema()).eval()
    batch = make_batch(3)
    batch["text_available"] = torch.zeros(3)
    with torch.no_grad():
        first = model(**batch).ranking_score
        batch["text"] = batch["text"] * 100.0 + 7.0
        second = model(**batch).ranking_score
    assert torch.allclose(first, second, atol=1e-6)


def test_concat_also_ignores_a_masked_modality():
    model = build_model(config_for("concat"), make_schema()).eval()
    batch = make_batch(3)
    batch["audio_available"] = torch.zeros(3)
    with torch.no_grad():
        first = model(**batch).ranking_score
        batch["audio"] = batch["audio"] * -50.0
        second = model(**batch).ranking_score
    assert torch.allclose(first, second, atol=1e-6)


def test_models_run_with_every_learned_modality_missing():
    for variant in sorted(MODEL_REGISTRY):
        model = build_model(config_for(variant), make_schema()).eval()
        batch = make_batch(2)
        batch["audio_available"] = torch.zeros(2)
        batch["text_available"] = torch.zeros(2)
        with torch.no_grad():
            output = model(**batch)
        assert torch.isfinite(output.ranking_score).all(), variant


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_gated_requires_equal_projection_widths():
    with pytest.raises(ValueError, match="must be equal"):
        build_model(config_for("gated", text_projection=64), make_schema())


def test_audio_only_experiment_names_are_distinct():
    embedding_only = config_for("audio_only", include_handcrafted_features=False)
    plus_acoustic = config_for("audio_only", include_handcrafted_features=True)
    assert embedding_only.experiment_name == "audio_embedding_only"
    assert plus_acoustic.experiment_name == "audio_plus_acoustic"
    schema = make_schema()
    assert (
        build_model(plus_acoustic, schema).parameter_count()
        > build_model(embedding_only, schema).parameter_count()
    )


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="unknown setting"):
        ModelConfig.from_mapping({"variant": "gated", "hidden_size": 64})


def test_dimension_policy_has_no_coerce_mode():
    with pytest.raises(ValueError, match="Unknown dimension_policy"):
        ModelConfig(variant="gated", dimension_policy="coerce")


@pytest.mark.parametrize(
    "filename",
    [
        "handcrafted_v1.yaml",
        "text_only_v1.yaml",
        "audio_only_v1.yaml",
        "audio_plus_acoustic_v1.yaml",
        "concat_v1.yaml",
        "gated_v1.yaml",
    ],
)
def test_shipped_configs_build_a_working_model(filename):
    path = Path(find_repo_root()) / "ml" / "configs" / "models" / filename
    config = load_model_config(path)
    model = build_model(config, make_schema())
    output = model(**make_batch())
    assert output.ranking_score.shape == (5,)


def test_registry_lists_every_variant():
    listed = {entry["variant"] for entry in list_models()}
    assert listed == set(MODEL_REGISTRY)
    assert all(entry["description"] for entry in list_models())


def test_describe_reports_the_real_parameter_count():
    described = describe_variant("gated", config_for("gated"), make_schema())
    model = build_model(config_for("gated"), make_schema())
    assert described["trainable_parameters"] == model.parameter_count()
    assert described["modalities"] == ["handcrafted", "audio", "text"]
