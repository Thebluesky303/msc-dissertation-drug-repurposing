"""Regression tests for verified benchmark preprocessing behaviour."""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.preprocessing import (
    ensure_pair_ids,
    build_esm2_residue_embedding_cache,
    kd_nm_to_pkd,
    load_residue_embedding_cache,
    make_pair_id,
    prepare_dataset,
    protein_sequence_to_tensor,
    validate_residue_embedding_cache,
)


def test_davis_kd_conversion_is_unchanged() -> None:
    np.testing.assert_allclose(
        kd_nm_to_pkd(np.array([1.0, 10.0, 1000.0])),
        np.array([9.0, 8.0, 6.0]),
    )
    with pytest.raises(ValueError, match="greater than zero"):
        kd_nm_to_pkd(0.0)


def test_protein_encoding_preserves_v1_truncation_and_unknown_token() -> None:
    encoded = protein_sequence_to_tensor("AC?DE", max_length=4)
    assert encoded.tolist() == [1, 2, 0, 3]
    assert protein_sequence_to_tensor("AAAAAA", max_length=3).shape == (3,)


def test_pair_ids_are_stable_unique_and_checked() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["davis", "davis"],
            "drug_id": ["D1", "D2"],
            "target_id": ["T1", "T1"],
        }
    )
    result = ensure_pair_ids(frame)
    assert result["pair_id"].tolist() == [
        make_pair_id("davis", "D1", "T1"),
        make_pair_id("davis", "D2", "T1"),
    ]
    bad = result.copy()
    bad.loc[0, "pair_id"] = "changed"
    with pytest.raises(ValueError, match="do not match"):
        ensure_pair_ids(bad)


def test_prepare_davis_dataset_preserves_graphdta_order_and_affinity(tmp_path) -> None:
    raw = tmp_path / "davis"
    raw.mkdir()
    (raw / "ligands_can.txt").write_text(
        json.dumps({"D1": "CC", "D2": "O"}), encoding="utf-8"
    )
    (raw / "proteins.txt").write_text(
        json.dumps({"T1": "ACD", "T2": "EFG"}), encoding="utf-8"
    )
    with (raw / "Y").open("wb") as file:
        pickle.dump(np.array([[1.0, np.nan], [10.0, 100.0]]), file)

    output = tmp_path / "davis_pairs.csv"
    pairs, summary = prepare_dataset("davis", raw, output)
    assert list(zip(pairs["drug_id"], pairs["target_id"])) == [
        ("D1", "T1"),
        ("D2", "T1"),
        ("D2", "T2"),
    ]
    np.testing.assert_allclose(pairs["affinity"], [9.0, 8.0, 7.0])
    assert pairs["affinity_type"].unique().tolist() == ["pKd"]
    assert summary["rows"] == 3
    assert summary["affinity_target"] == "pKd"
    assert summary["missing_affinity_cells"] == 1
    assert len(summary["raw_source_files"]["Y"]["sha256"]) == 64
    assert summary["processed_csv_file"] == output.name
    assert summary["processed_csv_bytes"] == output.stat().st_size
    assert len(summary["processed_csv_sha256"]) == 64
    assert summary["processed_csv_rows"] == len(pairs)
    assert len(summary["scientific_data_sha256"]) == 64
    assert output.exists()
    assert output.with_suffix(".metadata.json").exists()


def test_residue_cache_loader_reports_missing_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_residue_embedding_cache(tmp_path / "missing.pt")


class _FakeTokenizer:
    def __call__(self, sequences, **kwargs):
        del kwargs
        lengths = [len(sequence) + 2 for sequence in sequences]
        width = max(lengths)
        input_ids = torch.zeros((len(sequences), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        special_tokens_mask = torch.ones_like(input_ids)
        for row, sequence in enumerate(sequences):
            input_ids[row, : lengths[row]] = torch.tensor(
                [1, *range(2, len(sequence) + 2), 9]
            )
            attention_mask[row, : lengths[row]] = 1
            special_tokens_mask[row, 1 : len(sequence) + 1] = 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
        }


class _FakeEsmModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = input_ids.unsqueeze(-1).repeat(1, 1, 4).float() * self.scale
        return SimpleNamespace(last_hidden_state=hidden)


def test_esm_builder_caches_residues_and_freezes_supplied_model(tmp_path) -> None:
    pairs = pd.DataFrame(
        {
            "target_id": ["T1", "T2"],
            "protein_sequence": ["ACDE", "FG"],
        }
    )
    model = _FakeEsmModel()
    cache_path = tmp_path / "residues.pt"
    cache = build_esm2_residue_embedding_cache(
        dataframe=pairs,
        output_path=cache_path,
        max_length=3,
        batch_size=2,
        tokenizer=_FakeTokenizer(),
        model=model,
        device="cpu",
    )
    assert cache["T1"].shape == (3, 4)
    assert cache["T2"].shape == (2, 4)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    loaded = load_residue_embedding_cache(cache_path)
    torch.testing.assert_close(loaded["T1"], cache["T1"])
    assert loaded.metadata["residue_level"] is True
    assert loaded.metadata["max_length"] == 3
    assert loaded.metadata["cached_target_count"] == 2
    assert loaded.metadata["schema_version"] == 2
    assert loaded.metadata["tensor_file"] == cache_path.name
    assert loaded.metadata["tensor_file_bytes"] == cache_path.stat().st_size
    assert len(loaded.metadata["tensor_file_sha256"]) == 64
    assert loaded.metadata["target_residue_lengths"] == {"T1": 3, "T2": 2}
    assert cache_path.with_suffix(".metadata.json").exists()


def test_esm_resume_fails_if_a_target_sequence_changes(tmp_path) -> None:
    cache_path = tmp_path / "residues.pt"
    original = pd.DataFrame(
        {"target_id": ["T1"], "protein_sequence": ["ACD"]}
    )
    changed = pd.DataFrame(
        {"target_id": ["T1"], "protein_sequence": ["ACE"]}
    )
    build_esm2_residue_embedding_cache(
        original,
        cache_path,
        tokenizer=_FakeTokenizer(),
        model=_FakeEsmModel(),
        device="cpu",
    )
    with pytest.raises(ValueError, match="sequences changed"):
        build_esm2_residue_embedding_cache(
            changed,
            cache_path,
            tokenizer=_FakeTokenizer(),
            model=_FakeEsmModel(),
            device="cpu",
            resume=True,
        )


def test_loader_requires_metadata_by_default(tmp_path) -> None:
    cache_path = tmp_path / "legacy.pt"
    torch.save({"T1": torch.ones(2, 4)}, cache_path)
    with pytest.raises(FileNotFoundError, match="metadata"):
        load_residue_embedding_cache(cache_path)
    unverified = load_residue_embedding_cache(
        cache_path, require_metadata=False
    )
    assert unverified.metadata["provenance_verified"] is False


def test_residue_loader_rejects_tampered_tensor_before_deserializing(
    tmp_path,
) -> None:
    cache_path = tmp_path / "residues.pt"
    pairs = pd.DataFrame(
        {"target_id": ["T1"], "protein_sequence": ["ACD"]}
    )
    build_esm2_residue_embedding_cache(
        pairs,
        cache_path,
        tokenizer=_FakeTokenizer(),
        model=_FakeEsmModel(),
        device="cpu",
    )
    payload = bytearray(cache_path.read_bytes())
    payload[-1] ^= 1
    cache_path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256"):
        load_residue_embedding_cache(cache_path)


def test_residue_validator_rejects_tensor_sequence_length_mismatch() -> None:
    pairs = pd.DataFrame(
        {"target_id": ["T1"], "protein_sequence": ["ACD"]}
    )
    cache = {"T1": torch.ones(2, 4)}
    with pytest.raises(ValueError, match="length does not match"):
        validate_residue_embedding_cache(pairs, cache, max_length=1000)
