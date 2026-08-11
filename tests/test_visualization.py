"""Fast visual-contract tests for molecular graph figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from rdkit import Chem

from src.visualization import (
    atom_graph_label,
    bond_style_name,
    plot_molecule_and_graph,
)


def test_atom_labels_include_index_and_formal_charge() -> None:
    molecule = Chem.MolFromSmiles("C[NH3+]")
    assert molecule is not None
    assert atom_graph_label(molecule.GetAtomWithIdx(0)) == "C0"
    assert atom_graph_label(molecule.GetAtomWithIdx(1)) == "N1+"


def test_bond_styles_match_rdkit_bond_types() -> None:
    expected = {
        "CC": "single",
        "C=C": "double",
        "C#N": "triple",
        "c1ccccc1": "aromatic",
    }
    for smiles, style in expected.items():
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        assert bond_style_name(molecule.GetBondWithIdx(0)) == style


def test_side_by_side_figure_contains_matching_graph_content() -> None:
    molecule = Chem.MolFromSmiles("c1ccccc1C#N")
    assert molecule is not None
    figure, axes = plot_molecule_and_graph("c1ccccc1C#N")

    assert len(axes) == 2
    assert axes[0].get_title() == "RDKit 2D molecular structure"
    assert axes[1].get_title() == "Molecular graph (atoms and bonds)"
    assert len(axes[1].texts) == molecule.GetNumAtoms()
    assert {text.get_text() for text in axes[1].texts} == {
        atom_graph_label(atom) for atom in molecule.GetAtoms()
    }
    # Six dashed aromatic lines + one single line + three triple-bond lines.
    assert sum(line.get_linestyle() == "--" for line in axes[1].lines) == 6
    assert len(axes[1].lines) == 10
    plt.close(figure)


@pytest.mark.parametrize("smiles", ["", "not a smiles"])
def test_invalid_smiles_raise_clear_errors(smiles: str) -> None:
    with pytest.raises(ValueError, match="SMILES"):
        plot_molecule_and_graph(smiles)
