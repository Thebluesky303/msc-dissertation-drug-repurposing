"""Fast regression tests for v1-compatible splits and v2 manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.splits import (
    apply_split_manifest,
    build_split_manifest,
    create_split,
    load_split_manifest,
    materialize_split_manifests,
    save_split_manifest,
    structural_overlap_diagnostics,
    validate_split_integrity,
)
from src.preprocessing import (
    ensure_pair_ids,
    load_graphdta_pairs,
    pair_id_sha256,
    scientific_pairs_sha256,
)


def _synthetic_pairs(row_count: int = 20) -> pd.DataFrame:
    """Create enough unique drugs/targets for every three-way split."""

    return pd.DataFrame(
        {
            "dataset": "demo",
            "drug_id": [f"d{index}" for index in range(row_count)],
            "target_id": [f"t{index}" for index in range(row_count)],
            "canonical_smiles": [
                "CCO" if index % 2 == 0 else "CCN"
                for index in range(row_count)
            ],
            "smiles": [
                "CCO" if index % 2 == 0 else "CCN"
                for index in range(row_count)
            ],
            "protein_sequence": [
                "MKT" if index % 3 == 0 else "AAA"
                for index in range(row_count)
            ],
            "affinity": [float(index) for index in range(row_count)],
            "affinity_type": ["demo_score"] * row_count,
        }
    )


def _v1_reference_membership(
    dataframe: pd.DataFrame,
    split_type: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce the frozen v1 algorithm independently for regression QA."""

    if split_type == "random":
        train, temporary = train_test_split(
            dataframe,
            train_size=0.70,
            random_state=seed,
            shuffle=True,
        )
        validation, test = train_test_split(
            temporary,
            train_size=0.50,
            random_state=seed,
            shuffle=True,
        )
    else:
        entity_column = (
            "drug_id" if split_type == "cold_drug" else "target_id"
        )
        entities = sorted(
            dataframe[entity_column].astype(str).unique().tolist()
        )
        train_entities, temporary_entities = train_test_split(
            entities,
            train_size=0.70,
            random_state=seed,
            shuffle=True,
        )
        validation_entities, test_entities = train_test_split(
            temporary_entities,
            train_size=0.50,
            random_state=seed,
            shuffle=True,
        )
        entity_values = dataframe[entity_column].astype(str)
        train = dataframe[entity_values.isin(set(train_entities))]
        validation = dataframe[
            entity_values.isin(set(validation_entities))
        ]
        test = dataframe[entity_values.isin(set(test_entities))]

    return tuple(
        partition.reset_index(drop=True)
        for partition in (train, validation, test)
    )


def _pair_list(dataframe: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(dataframe["drug_id"], dataframe["target_id"]))


@pytest.mark.parametrize("split_type", ["random", "cold_drug", "cold_target"])
def test_dispatcher_exactly_matches_frozen_v1_algorithm(
    split_type: str,
) -> None:
    dataframe = _synthetic_pairs()
    expected = _v1_reference_membership(dataframe, split_type, seed=42)

    actual = create_split(dataframe, split_type, random_seed=42)

    assert [_pair_list(partition) for partition in actual] == [
        _pair_list(partition) for partition in expected
    ]


@pytest.mark.parametrize("split_type", ["random", "cold_drug", "cold_target"])
def test_splits_are_deterministic_and_conserve_every_pair(
    split_type: str,
) -> None:
    dataframe = _synthetic_pairs()

    first = create_split(dataframe, split_type, random_seed=123)
    second = create_split(dataframe, split_type, random_seed=123)

    assert [_pair_list(partition) for partition in first] == [
        _pair_list(partition) for partition in second
    ]
    assigned = [pair for partition in first for pair in _pair_list(partition)]
    assert len(assigned) == len(dataframe)
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(_pair_list(dataframe))

    report = validate_split_integrity(*first, split_type=split_type)
    assert report["pair_overlap"] is False
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        assert report[f"pair_{left}_{right}_overlap_count"] == 0
        if split_type == "cold_drug":
            assert report[f"drug_{left}_{right}_overlap_count"] == 0
        if split_type == "cold_target":
            assert report[f"target_{left}_{right}_overlap_count"] == 0


def test_integrity_checks_validation_test_cold_drug_leakage() -> None:
    train = pd.DataFrame({"drug_id": ["d1"], "target_id": ["t1"]})
    validation = pd.DataFrame({"drug_id": ["d2"], "target_id": ["t2"]})
    test = pd.DataFrame({"drug_id": ["d2"], "target_id": ["t3"]})

    with pytest.raises(ValueError, match="Drug identifiers overlap"):
        validate_split_integrity(
            train,
            validation,
            test,
            split_type="cold_drug",
        )


def test_integrity_checks_validation_test_cold_target_leakage() -> None:
    train = pd.DataFrame({"drug_id": ["d1"], "target_id": ["t1"]})
    validation = pd.DataFrame({"drug_id": ["d2"], "target_id": ["t2"]})
    test = pd.DataFrame({"drug_id": ["d3"], "target_id": ["t2"]})

    with pytest.raises(ValueError, match="Target identifiers overlap"):
        validate_split_integrity(
            train,
            validation,
            test,
            split_type="cold_target",
        )


def test_structural_diagnostics_are_separate_from_id_membership() -> None:
    train = pd.DataFrame(
        {
            "drug_id": ["d1"],
            "target_id": ["t1"],
            "smiles": ["CCO"],
            "protein_sequence": ["mk t"],
        }
    )
    validation = pd.DataFrame(
        {
            "drug_id": ["d2"],
            "target_id": ["t2"],
            "smiles": ["OCC"],
            "protein_sequence": ["MKT"],
        }
    )
    test = pd.DataFrame(
        {
            "drug_id": ["d3"],
            "target_id": ["t3"],
            "smiles": ["CCN"],
            "protein_sequence": ["AAA"],
        }
    )

    report = structural_overlap_diagnostics(
        train,
        validation,
        test,
        canonical_smiles_column="smiles",
        protein_sequence_column="protein_sequence",
        canonicalize_smiles_values=True,
    )

    assert report["canonical_smiles_train_validation_overlap_count"] == 1
    assert report["canonical_smiles_overlap"] is True
    assert report["protein_sequence_train_validation_overlap_count"] == 1
    assert report["protein_sequence_overlap"] is True

    integrity = validate_split_integrity(
        train,
        validation,
        test,
        split_type="cold_drug",
        canonical_smiles_column="smiles",
        protein_sequence_column="protein_sequence",
        canonicalize_smiles_values=True,
    )
    assert integrity["drug_overlap"] is False
    assert integrity["canonical_smiles_overlap"] is True


def test_manifest_round_trip_reconstructs_recorded_partition_order(
    tmp_path: Path,
) -> None:
    dataframe = _synthetic_pairs()
    splits = create_split(dataframe, "random", random_seed=2026)
    manifest = build_split_manifest(*splits, dataset_name="demo")
    path = tmp_path / "demo_random_manifest.csv"

    saved_path = save_split_manifest(
        manifest,
        path,
        metadata={"seed": 2026},
    )
    loaded = load_split_manifest(saved_path)
    reconstructed = apply_split_manifest(
        dataframe.sample(frac=1.0, random_state=7),
        loaded,
    )

    assert saved_path == path
    assert loaded.equals(manifest)
    assert [_pair_list(partition) for partition in reconstructed] == [
        _pair_list(partition) for partition in splits
    ]
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    assert metadata["schema_version"] == 1
    assert metadata["manifest_rows"] == len(dataframe)
    assert len(metadata["manifest_sha256"]) == 64
    assert metadata["partition_rows"] == {
        "train": len(splits[0]),
        "validation": len(splits[1]),
        "test": len(splits[2]),
    }


def test_manifest_application_fails_closed_for_stale_source() -> None:
    dataframe = _synthetic_pairs()
    splits = create_split(dataframe, "random", random_seed=42)
    manifest = build_split_manifest(*splits, dataset_name="demo")

    with pytest.raises(ValueError, match="Manifest/source pair IDs differ"):
        apply_split_manifest(dataframe.iloc[:-1], manifest)


def test_materialization_writes_all_deterministic_manifests_and_metadata(
    tmp_path: Path,
) -> None:
    dataframe = _synthetic_pairs()
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"

    first_paths = materialize_split_manifests(
        dataframe,
        dataset_name="demo",
        output_root=first_directory,
        seed=42,
    )
    second_paths = materialize_split_manifests(
        dataframe,
        dataset_name="demo",
        output_root=second_directory,
        seed=42,
    )
    alternate_seed_paths = materialize_split_manifests(
        dataframe,
        dataset_name="demo",
        output_root=first_directory,
        seed=123,
    )

    assert set(first_paths) == {"random", "cold_drug", "cold_target"}
    assert all("seed42" in path.name for path in first_paths.values())
    assert all("seed123" in path.name for path in alternate_seed_paths.values())
    assert set(first_paths.values()).isdisjoint(alternate_seed_paths.values())
    assert all(path.exists() for path in first_paths.values())
    assert all(path.exists() for path in alternate_seed_paths.values())
    for split_type in first_paths:
        assert first_paths[split_type].read_bytes() == second_paths[
            split_type
        ].read_bytes()
        manifest = load_split_manifest(first_paths[split_type])
        assert len(manifest) == len(dataframe)

    cold_drug_metadata = json.loads(
        first_paths["cold_drug"].with_suffix(".metadata.json").read_text()
    )
    assert cold_drug_metadata["membership_policy"].startswith("ID-based")
    assert cold_drug_metadata["structural_diagnostics_change_membership"] is False
    assert cold_drug_metadata["integrity"]["drug_overlap"] is False
    assert cold_drug_metadata["integrity"]["canonical_smiles_overlap"] is True
    assert len(cold_drug_metadata["source_data_sha256"]) == 64
    assert cold_drug_metadata["source_data_sha256"] == scientific_pairs_sha256(
        ensure_pair_ids(dataframe)
    )
    assert cold_drug_metadata["source_data_fingerprint_algorithm"] == (
        "scientific_pairs_sha256_v1"
    )
    assert cold_drug_metadata["source_pair_id_fingerprint_algorithm"] == (
        "ordered_pair_ids_sha256_v1"
    )


@pytest.mark.parametrize(
    ("dataset", "rows", "pair_digest", "data_digest", "split_expectations"),
    [
        (
            "davis",
            30_056,
            "dac2247c98d11522dad590e9d50dd7e381fb4c2e32b502d7a1453a521f256380",
            "36563adcc1d090266eecec63561366b3bb639e79a91aa798a3ca6ae3cfeec9a6",
            {
                "random": ([21_039, 4_508, 4_509], "74acb4aac1f932c32ea686a434e01ee116c130e27624f0bf679a52b88876a992"),
                "cold_drug": ([20_774, 4_420, 4_862], "63b0182075de3fa7a4cb28cd7f1762c5dc20d56f29b54c226672740b40285cd0"),
                "cold_target": ([21_012, 4_488, 4_556], "fa7256abb0434b939814f4c9a268c037982f712717d33dd66344464ddb5b6a13"),
            },
        ),
        (
            "kiba",
            118_254,
            "5fc01ba99180ebb88f196c76306a0f4e75544d4b99f03a440006f976b576fed4",
            "fdb98bc878ce45ee8472625ddea46f246daa597b40a0b420e122efb293d01e9d",
            {
                "random": ([82_777, 17_738, 17_739], "e284e48327477ce9826c12dfdaa51e0a977580b7aa204ec596df8a49315b93e0"),
                "cold_drug": ([83_966, 17_544, 16_744], "9bf2bc71343a60e3dd608f455457a762cee0fa2c8e49881a3e5646594c40a43f"),
                "cold_target": ([84_964, 16_513, 16_777], "860b1b354a38d140f76191782cc67201f1efe8d0c4152e75faae5f31c240071b"),
            },
        ),
    ],
)
def test_optional_local_graphdta_regression(
    dataset: str,
    rows: int,
    pair_digest: str,
    data_digest: str,
    split_expectations: dict[str, tuple[list[int], str]],
) -> None:
    raw_directory = Path("data/raw") / dataset
    if not all(
        (raw_directory / name).exists()
        for name in ("ligands_can.txt", "proteins.txt", "Y")
    ):
        pytest.skip("Optional local GraphDTA raw files are not staged.")
    pairs = load_graphdta_pairs(dataset, raw_directory)
    assert len(pairs) == rows
    assert pair_id_sha256(pairs) == pair_digest
    assert scientific_pairs_sha256(pairs) == data_digest
    for split_type, (counts, expected_digest) in split_expectations.items():
        partitions = create_split(pairs, split_type, random_seed=42)
        assert [len(partition) for partition in partitions] == counts
        payload = {
            name: partition["pair_id"].astype(str).tolist()
            for name, partition in zip(
                ("train", "validation", "test"), partitions
            )
        }
        observed_digest = hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert observed_digest == expected_digest
