"""Fast synthetic tests for AML screening utilities."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from config import (
    BOND_FEATURE_DIM,
    DRUG_HIDDEN_DIM,
    FUSION_HIDDEN_DIM,
    NODE_FEATURE_DIM,
    PROTEIN_EMBEDDING_DIM,
    PROTEIN_HIDDEN_DIM,
)
from src.models import create_advanced_dta_model, create_dta_model
from src.preprocessing import (
    ResidueEmbeddingCache,
    build_drug_graph_cache,
    build_protein_tensor_cache,
)
import src.screening as screening

from src.screening import (
    SCREENING_RESULT_DESCRIPTION,
    build_screening_pairs,
    compare_screening_models,
    load_screening_checkpoint,
    predict_advanced_checkpoint,
    predict_baseline_checkpoint,
    predict_screening_pairs,
    prepare_aml_targets,
    prepare_approved_drugs,
    rank_screening_predictions,
    run_dual_model_aml_screening,
    validate_screening_checkpoint_compatibility,
)


def _approved_drug_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "drug_id": "D1",
                "drug_name": "ethanol",
                "smiles": "CCO",
                "approval_source": "synthetic-test-source",
            },
            {
                "drug_id": "D2",
                "drug_name": "duplicate ethanol",
                "smiles": "OCC",
                "approval_source": "synthetic-test-source",
            },
            {
                "drug_id": "D3",
                "drug_name": "ethylamine",
                "smiles": "CCN",
                "approval_source": "synthetic-test-source",
            },
            {
                "drug_id": "D4",
                "drug_name": "sodium ethanol mixture",
                "smiles": "CCO.[Na+]",
                "approval_source": "synthetic-test-source",
            },
            {
                "drug_id": "D5",
                "drug_name": "invalid",
                "smiles": "not a smiles",
                "approval_source": "synthetic-test-source",
            },
        ]
    )


def _target_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_id": "P00001",
                "gene_symbol": "GENE1",
                "protein_sequence": "ACDEFGHIK",
            },
            {
                "target_id": "P00002",
                "gene_symbol": "GENE2",
                "protein_sequence": "LMNPQRSTV",
            },
        ]
    )


def test_approved_drugs_are_canonical_deduplicated_and_auditable() -> None:
    prepared, rejected = prepare_approved_drugs(
        _approved_drug_fixture(),
        return_rejections=True,
    )

    assert prepared["drug_id"].tolist() == ["D1", "D3"]
    assert prepared["canonical_smiles"].is_unique
    assert prepared["inchikey"].str.len().eq(27).all()
    assert prepared["approval_source"].eq("synthetic-test-source").all()
    assert set(rejected["rejection_reason"]) == {
        "duplicate_structure",
        "multicomponent_structure",
        "invalid_smiles",
    }
    duplicate = rejected.loc[
        rejected["rejection_reason"] == "duplicate_structure"
    ].iloc[0]
    assert duplicate["duplicate_of_drug_id"] == "D1"


def test_source_id_formulation_conflicts_are_row_level_rejections() -> None:
    source = pd.DataFrame(
        [
            {
                "drug_id": "DB1",
                "drug_name": "example parent",
                "smiles": "CCN",
                "approval_source": "source-a",
            },
            {
                "drug_id": "DB1",
                "drug_name": "example hydrochloride",
                "smiles": "CC[NH3+].[Cl-]",
                "approval_source": "source-b",
            },
            {
                "drug_id": "DB1",
                "drug_name": "conflicting formulation",
                "smiles": "CCO",
                "approval_source": "source-c",
            },
        ]
    )

    prepared, rejected = prepare_approved_drugs(source, return_rejections=True)

    assert prepared[["drug_id", "drug_name"]].to_dict("records") == [
        {"drug_id": "DB1", "drug_name": "example parent"}
    ]
    assert set(rejected["rejection_reason"]) == {
        "multicomponent_structure",
        "conflicting_drug_id",
    }
    assert set(rejected["approval_source"]) == {"source-b", "source-c"}


def test_csv_scalar_ids_are_normalised_and_blank_rows_are_audited() -> None:
    source = pd.DataFrame(
        [
            {"drug_id": 101, "drug_name": 202, "smiles": "CCO"},
            {"drug_id": 102, "drug_name": "", "smiles": "CCN"},
            {"drug_id": 103, "drug_name": "Drug 103", "smiles": None},
            {"drug_id": None, "drug_name": "Drug 104", "smiles": "CCC"},
        ]
    )

    prepared, rejected = prepare_approved_drugs(source, return_rejections=True)

    assert prepared.loc[0, "drug_id"] == "101"
    assert prepared.loc[0, "drug_name"] == "202"
    assert prepared.loc[0, "source_drug_id"] == 101
    assert prepared.loc[0, "source_drug_name"] == 202
    assert set(rejected["rejection_reason"]) == {
        "missing_or_invalid_drug_id",
        "missing_or_invalid_drug_name",
        "missing_or_invalid_smiles",
    }


def test_graph_incompatible_structure_is_rejected() -> None:
    drugs = _approved_drug_fixture().iloc[[0]].copy()

    def incompatible_graph_builder(smiles: str) -> None:
        raise RuntimeError(f"unsupported graph: {smiles}")

    with pytest.raises(ValueError, match="No approved-drug records"):
        prepare_approved_drugs(drugs, graph_builder=incompatible_graph_builder)


def test_targets_are_preserved_and_pair_ids_are_stable() -> None:
    prepared_drugs = prepare_approved_drugs(_approved_drug_fixture())
    source_targets = _target_fixture()
    original_ids = source_targets["target_id"].copy()
    original_sequences = source_targets["protein_sequence"].copy()
    prepared_targets = prepare_aml_targets(source_targets)

    pd.testing.assert_series_equal(
        prepared_targets["target_id"],
        original_ids,
    )
    pd.testing.assert_series_equal(
        prepared_targets["protein_sequence"],
        original_sequences,
    )
    pairs = build_screening_pairs(prepared_drugs, prepared_targets)
    reversed_pairs = build_screening_pairs(
        prepared_drugs.iloc[::-1].reset_index(drop=True),
        prepared_targets.iloc[::-1].reset_index(drop=True),
    )

    assert len(pairs) == len(prepared_drugs) * len(prepared_targets)
    assert pairs["pair_id"].is_unique
    mapping = pairs.set_index(["drug_id", "target_id"])["pair_id"].to_dict()
    reversed_mapping = reversed_pairs.set_index(
        ["drug_id", "target_id"]
    )["pair_id"].to_dict()
    assert mapping == reversed_mapping


def test_numeric_csv_target_ids_are_normalised_with_source_audit() -> None:
    source_targets = _target_fixture().copy()
    source_targets["target_id"] = [101, 102]

    prepared = prepare_aml_targets(source_targets)

    assert prepared["target_id"].tolist() == ["101", "102"]
    assert prepared["source_target_id"].tolist() == [101, 102]
    assert prepared["protein_sequence"].tolist() == source_targets[
        "protein_sequence"
    ].tolist()


def test_inference_callback_ranking_and_safe_wording() -> None:
    drug_rows = [
        {"drug_id": f"D{index:02d}", "drug_name": f"Drug {index}"}
        for index in range(1, 9)
    ]
    target_rows = [
        {"target_id": target_id, "protein_sequence": "ACDE"}
        for target_id in ["T1", "T2", "T3"]
    ]
    pairs = build_screening_pairs(
        pd.DataFrame(drug_rows),
        pd.DataFrame(target_rows),
    )
    score_by_pair = {
        pair_id: float(index)
        for index, pair_id in enumerate(pairs["pair_id"], start=1)
    }

    def predictor(batch: pd.DataFrame) -> pd.DataFrame:
        # Reverse callback output to verify pair-ID based realignment.
        output = pd.DataFrame(
            {
                "pair_id": batch["pair_id"],
                "predicted_binding_affinity_score": batch["pair_id"].map(
                    score_by_pair
                ),
            }
        )
        return output.iloc[::-1].reset_index(drop=True)

    predictions = predict_screening_pairs(pairs, predictor, batch_size=7)
    rankings = rank_screening_predictions(predictions, model_name="fixture")

    assert predictions["predicted_binding_affinity_score"].tolist() == list(
        score_by_pair.values()
    )
    assert len(rankings.top_10_overall) == 10
    assert len(rankings.top_20_overall) == 20
    assert rankings.top_5_per_target.groupby("target_id").size().eq(5).all()
    assert rankings.ranked.iloc[0]["overall_rank"] == 1
    assert (
        rankings.metadata["result_description"]
        == SCREENING_RESULT_DESCRIPTION
    )
    assert "not experimentally or clinically validated" in rankings.metadata[
        "interpretation"
    ]


def test_baseline_advanced_comparison_reports_overlap_and_target_agreement() -> None:
    pairs = build_screening_pairs(
        pd.DataFrame(
            [
                {"drug_id": "D1", "drug_name": "Drug 1"},
                {"drug_id": "D2", "drug_name": "Drug 2"},
                {"drug_id": "D3", "drug_name": "Drug 3"},
            ]
        ),
        _target_fixture(),
    )
    baseline = pairs.copy()
    advanced = pairs.copy()
    baseline["predicted_binding_affinity_score"] = [9, 8, 7, 6, 5, 4]
    advanced["predicted_binding_affinity_score"] = [9, 7, 8, 6, 4, 5]

    comparison = compare_screening_models(
        baseline,
        advanced,
        top_n=4,
        per_target_n=2,
    )

    assert len(comparison.pair_comparison) == len(pairs)
    assert {
        "overall_rank_difference",
        "target_rank_difference",
        "score_difference",
    }.issubset(comparison.pair_comparison.columns)
    assert not comparison.shared_top_drugs.empty
    assert not comparison.top_pair_overlap.empty
    assert set(comparison.target_agreement["target_id"]) == {"P00001", "P00002"}
    assert comparison.target_agreement["top_drug_jaccard"].between(0, 1).all()
    assert (
        comparison.summary["result_description"]
        == SCREENING_RESULT_DESCRIPTION
    )


def test_target_validation_rejects_duplicate_ids_without_changing_them() -> None:
    targets = _target_fixture()
    targets.loc[1, "target_id"] = targets.loc[0, "target_id"]
    with pytest.raises(ValueError, match="must be unique"):
        prepare_aml_targets(targets)


def _checkpoint_pairs() -> pd.DataFrame:
    drugs = prepare_approved_drugs(
        pd.DataFrame(
            [
                {"drug_id": "D1", "drug_name": "Drug 1", "smiles": "CCO"},
                {"drug_id": "D2", "drug_name": "Drug 2", "smiles": "CCN"},
            ]
        )
    )
    return build_screening_pairs(drugs, prepare_aml_targets(_target_fixture()))


def _residue_cache(
    pairs: pd.DataFrame,
    *,
    model_revision: str = "fixture-revision-001",
    verified: bool = True,
    max_length: int = 12,
    embedding_dimension: int = 8,
) -> ResidueEmbeddingCache:
    sequences = (
        pairs[["target_id", "protein_sequence"]]
        .drop_duplicates("target_id")
        .set_index("target_id")["protein_sequence"]
        .to_dict()
    )
    generator = torch.Generator().manual_seed(123)
    tensors = {
        str(target_id): torch.randn(
            len(sequence),
            embedding_dimension,
            generator=generator,
        )
        for target_id, sequence in sequences.items()
    }
    hashes = {
        str(target_id): hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        for target_id, sequence in sequences.items()
    }
    metadata = {
        "schema_version": 2,
        "model_name": "fixture/esm2",
        "model_revision": model_revision,
        "resolved_model_revision": model_revision,
        "model_provenance_verified": verified,
        "provenance_verified": verified,
        "artifact_integrity_verified": verified,
        "model_parameters_frozen": True,
        "max_length": max_length,
        "embedding_dimension": embedding_dimension,
        "target_sequence_sha256": hashes,
        "target_residue_lengths": {
            str(target_id): len(sequence)
            for target_id, sequence in sequences.items()
        },
        "cached_target_ids": sorted(tensors),
        "tensor_file": "fixture_residue_cache.pt",
        "tensor_file_bytes": 1,
        "tensor_file_sha256": hashlib.sha256(b"fixture-cache").hexdigest(),
    }
    return ResidueEmbeddingCache(tensors, metadata=metadata)


def _checkpoint_configuration(
    *,
    model_family: str,
    dataset: str = "davis",
    residue_cache: ResidueEmbeddingCache | None = None,
    seed: int = 42,
) -> dict:
    affinity_label = "pKd" if dataset == "davis" else "KIBA score"
    data_fingerprint = hashlib.sha256(
        f"{dataset}-synthetic-training-data".encode("utf-8")
    ).hexdigest()
    common = {
        "dataset": dataset,
        "split": "random",
        "model": "gcn" if model_family == "baseline" else "advanced",
        "affinity_label": affinity_label,
        "model_family": model_family,
        "seed": seed,
        "custom_model_supplied": False,
        "max_protein_length": 12,
        "atom_feature_dim": NODE_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "dropout": 0.0,
        "checkpoint_selection_metric": "validation_mse",
        "test_evaluation_policy": "after_best_validation_checkpoint",
        "data_provenance": {
            "mode": "synthetic_fixture",
            "data_sha256": data_fingerprint,
        },
        "runtime_provenance": {
            "python": "synthetic-test-runtime",
            "git_commit": "synthetic-test-commit",
        },
    }
    if model_family == "baseline":
        common["architecture"] = {
            "drug_hidden_dim": DRUG_HIDDEN_DIM,
            "protein_embedding_dim": PROTEIN_EMBEDDING_DIM,
            "protein_hidden_dim": PROTEIN_HIDDEN_DIM,
            "fusion_hidden_dim": FUSION_HIDDEN_DIM,
            "protein_encoder": "v1 token embedding + 2-layer 1D CNN",
        }
    else:
        assert residue_cache is not None
        cache_metadata = dict(residue_cache.metadata)
        common["architecture"] = {
            "hidden_dim": 16,
            "gine_layers": 4,
            "protein_cnn_layers": 3,
            "attention_heads": 4,
            "attention_feedforward_dim": 32,
            "regression_hidden_dim": 32,
            "regression_bottleneck_dim": 16,
            "esm2_model_name": cache_metadata["model_name"],
            "residue_embedding_dim_used": 8,
            "esm2_execution_during_training": False,
            "residue_cache_provenance": cache_metadata,
        }
    return common


def _write_checkpoint(
    path: Path,
    *,
    model_family: str,
    dataset: str = "davis",
    residue_cache: ResidueEmbeddingCache | None = None,
    seed: int = 42,
    split_fingerprint_source: bytes = b"fixture-split",
) -> Path:
    torch.manual_seed(42)
    configuration = _checkpoint_configuration(
        model_family=model_family,
        dataset=dataset,
        residue_cache=residue_cache,
        seed=seed,
    )
    if model_family == "baseline":
        model = create_dta_model("gcn", dropout=0.0)
    else:
        model = create_advanced_dta_model(
            atom_input_dim=6,
            bond_feature_dim=6,
            esm_embedding_dim=8,
            embedding_dim=16,
            attention_heads=4,
            attention_feedforward_dim=32,
            regression_hidden_dim=32,
            regression_bottleneck_dim=16,
            dropout=0.0,
        )
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "validation_metrics": {"mse": 0.25},
            "metadata": {
                "configuration_used": configuration,
                "affinity_label": configuration["affinity_label"],
                "data_fingerprint": configuration["data_provenance"][
                    "data_sha256"
                ],
                "data_provenance": configuration["data_provenance"],
                "runtime_provenance": configuration["runtime_provenance"],
                "split_fingerprint": hashlib.sha256(
                    split_fingerprint_source
                ).hexdigest(),
                "split_manifest": {
                    "mode": "deterministically_reconstructed",
                    "path": None,
                    "metadata_path": None,
                    "sha256": None,
                },
            },
        },
        path,
    )
    return path


def test_baseline_checkpoint_prediction_and_cache_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _checkpoint_pairs()
    checkpoint_path = _write_checkpoint(
        tmp_path / "baseline.pt",
        model_family="baseline",
    )
    original_torch_load = torch.load
    observed_map_locations: list[str] = []

    def record_restricted_load(*args, **kwargs):
        observed_map_locations.append(kwargs["map_location"])
        assert kwargs["weights_only"] is True
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(screening.torch, "load", record_restricted_load)
    loaded = load_screening_checkpoint(
        checkpoint_path,
        expected_model_family="baseline",
        device="cpu",
    )
    assert observed_map_locations == ["cpu"]
    graph_cache = build_drug_graph_cache(pairs)
    token_cache = build_protein_tensor_cache(pairs, max_length=12)

    def unexpected_build(*args, **kwargs):
        raise AssertionError("provided screening caches should be reused")

    monkeypatch.setattr(screening, "build_drug_graph_cache", unexpected_build)
    monkeypatch.setattr(screening, "build_protein_tensor_cache", unexpected_build)
    predictions = predict_baseline_checkpoint(
        pairs,
        loaded,
        drug_graph_cache=graph_cache,
        protein_tensor_cache=token_cache,
        batch_size=2,
    )

    assert len(predictions) == len(pairs)
    assert torch.isfinite(
        torch.tensor(predictions["predicted_binding_affinity_score"].to_numpy())
    ).all()
    assert predictions["training_dataset"].eq("davis").all()
    assert predictions["affinity_label"].eq("pKd").all()
    assert predictions["checkpoint_sha256"].str.len().eq(64).all()
    assert predictions["checkpoint_model_family"].eq("baseline").all()
    assert "checkpoint_configuration" not in predictions.columns
    assert "checkpoint_split_manifest" not in predictions.columns
    assert "checkpoint_configuration" in predictions.attrs[
        "checkpoint_provenance"
    ]


def test_advanced_checkpoint_prediction_requires_matching_verified_cache(
    tmp_path: Path,
) -> None:
    pairs = _checkpoint_pairs()
    cache = _residue_cache(pairs)
    checkpoint_path = _write_checkpoint(
        tmp_path / "advanced.pt",
        model_family="advanced",
        residue_cache=cache,
    )
    predictions = predict_advanced_checkpoint(
        pairs,
        checkpoint_path,
        cache,
        batch_size=2,
        device="cpu",
    )

    assert len(predictions) == len(pairs)
    assert predictions["checkpoint_model_family"].eq("advanced").all()
    checkpoint_configuration = predictions.attrs["checkpoint_provenance"][
        "checkpoint_configuration"
    ]
    assert (
        checkpoint_configuration["architecture"]["residue_cache_provenance"][
            "resolved_model_revision"
        ]
        == "fixture-revision-001"
    )

    unverified = _residue_cache(pairs, verified=False)
    with pytest.raises(ValueError, match="provenance-verified"):
        predict_advanced_checkpoint(
            pairs,
            checkpoint_path,
            unverified,
            device="cpu",
        )
    wrong_revision = _residue_cache(pairs, model_revision="different-revision")
    with pytest.raises(ValueError, match="revision"):
        predict_advanced_checkpoint(
            pairs,
            checkpoint_path,
            wrong_revision,
            device="cpu",
        )


def test_dual_checkpoint_runner_and_mismatched_scale_rejection(
    tmp_path: Path,
) -> None:
    pairs = _checkpoint_pairs()
    cache = _residue_cache(pairs)
    baseline_path = _write_checkpoint(
        tmp_path / "baseline_davis.pt",
        model_family="baseline",
    )
    advanced_path = _write_checkpoint(
        tmp_path / "advanced_davis.pt",
        model_family="advanced",
        residue_cache=cache,
    )
    result = run_dual_model_aml_screening(
        pairs,
        baseline_path,
        advanced_path,
        cache,
        batch_size=2,
        device="cpu",
    )

    assert result.provenance["training_dataset"] == "davis"
    assert len(result.provenance["screening_input"]["screening_pairs_sha256"]) == 64
    assert result.provenance["screening_input"]["screening_pair_count"] == len(
        pairs
    )
    assert len(result.baseline_predictions) == len(pairs)
    assert len(result.advanced_predictions) == len(pairs)
    assert len(result.comparison.pair_comparison) == len(pairs)
    baseline_csv = tmp_path / "baseline_predictions.csv"
    advanced_csv = tmp_path / "advanced_predictions.csv"
    result.baseline_predictions.to_csv(baseline_csv, index=False)
    result.advanced_predictions.to_csv(advanced_csv, index=False)
    csv_comparison = compare_screening_models(
        pd.read_csv(baseline_csv),
        pd.read_csv(advanced_csv),
    )
    assert len(csv_comparison.pair_comparison) == len(pairs)

    advanced_kiba_path = _write_checkpoint(
        tmp_path / "advanced_kiba.pt",
        model_family="advanced",
        dataset="kiba",
        residue_cache=cache,
    )
    baseline_loaded = load_screening_checkpoint(baseline_path, device="cpu")
    advanced_loaded = load_screening_checkpoint(advanced_path, device="cpu")
    different_data = replace(
        advanced_loaded,
        data_fingerprint=hashlib.sha256(b"different-training-data").hexdigest(),
    )
    with pytest.raises(ValueError, match="training-data.*fingerprints"):
        validate_screening_checkpoint_compatibility(
            baseline_loaded,
            different_data,
        )
    different_split = replace(
        advanced_loaded,
        split_fingerprint=hashlib.sha256(b"different-split").hexdigest(),
    )
    with pytest.raises(ValueError, match="different split fingerprints"):
        validate_screening_checkpoint_compatibility(
            baseline_loaded,
            different_split,
        )
    advanced_kiba = load_screening_checkpoint(advanced_kiba_path, device="cpu")
    with pytest.raises(ValueError, match="different datasets"):
        validate_screening_checkpoint_compatibility(
            baseline_loaded,
            advanced_kiba,
        )

    advanced_other_evidence_path = _write_checkpoint(
        tmp_path / "advanced_other_evidence.pt",
        model_family="advanced",
        residue_cache=cache,
        seed=123,
        split_fingerprint_source=b"different-split-membership",
    )
    advanced_other_evidence = load_screening_checkpoint(
        advanced_other_evidence_path,
        device="cpu",
    )
    with pytest.raises(ValueError, match="'seed'"):
        validate_screening_checkpoint_compatibility(
            baseline_loaded,
            advanced_other_evidence,
        )


def test_comparison_rejects_checkpoint_provenance_on_different_scales(
    tmp_path: Path,
) -> None:
    pairs = _checkpoint_pairs()
    graph_cache = build_drug_graph_cache(pairs)
    token_cache = build_protein_tensor_cache(pairs, max_length=12)
    baseline_path = _write_checkpoint(
        tmp_path / "baseline.pt",
        model_family="baseline",
    )
    baseline = predict_baseline_checkpoint(
        pairs,
        baseline_path,
        drug_graph_cache=graph_cache,
        protein_tensor_cache=token_cache,
        device="cpu",
    )
    advanced = baseline.copy()
    advanced["checkpoint_model_family"] = "advanced"
    advanced["training_dataset"] = "kiba"
    advanced["affinity_label"] = "KIBA score"

    with pytest.raises(ValueError, match="different training datasets"):
        compare_screening_models(baseline, advanced)

    advanced_same_scale = baseline.copy()
    advanced_same_scale["checkpoint_model_family"] = "advanced"
    advanced_same_scale["checkpoint_data_fingerprint"] = hashlib.sha256(
        b"different-training-data"
    ).hexdigest()
    with pytest.raises(ValueError, match="checkpoint_data_fingerprint"):
        compare_screening_models(baseline, advanced_same_scale)
