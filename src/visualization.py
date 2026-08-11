"""Dissertation-ready molecular structure and graph visualisation utilities."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Draw, rdDepictor


ATOM_COLOURS: dict[str, str] = {
    "C": "#64748B",
    "N": "#2563EB",
    "O": "#DC2626",
    "F": "#16A34A",
    "Cl": "#16A34A",
    "Br": "#A16207",
    "I": "#7E22CE",
    "P": "#EA580C",
    "S": "#EAB308",
    "B": "#D97706",
    "H": "#CBD5E1",
}

BOND_STYLE_METADATA: dict[str, dict[str, Any]] = {
    "single": {"line_count": 1, "linestyle": "-"},
    "double": {"line_count": 2, "linestyle": "-"},
    "triple": {"line_count": 3, "linestyle": "-"},
    "aromatic": {"line_count": 1, "linestyle": "--"},
    "other": {"line_count": 1, "linestyle": ":"},
}

__all__ = [
    "ATOM_COLOURS",
    "BOND_STYLE_METADATA",
    "atom_graph_label",
    "bond_style_name",
    "plot_molecule_and_graph",
    "visualize_molecular_graph",
]


def atom_graph_label(atom: Chem.Atom, *, include_index: bool = True) -> str:
    """Return a readable element/index/formal-charge label for a graph node."""

    isotope = str(atom.GetIsotope()) if atom.GetIsotope() else ""
    symbol = atom.GetSymbol()
    index = str(atom.GetIdx()) if include_index else ""
    charge = atom.GetFormalCharge()
    if charge > 0:
        charge_label = "+" if charge == 1 else f"+{charge}"
    elif charge < 0:
        charge_label = "-" if charge == -1 else str(charge)
    else:
        charge_label = ""
    return f"{isotope}{symbol}{index}{charge_label}"


def bond_style_name(bond: Chem.Bond) -> str:
    """Map an RDKit bond to the visual style used in the graph panel."""

    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        return "single"
    if bond_type == Chem.rdchem.BondType.DOUBLE:
        return "double"
    if bond_type == Chem.rdchem.BondType.TRIPLE:
        return "triple"
    if bond_type == Chem.rdchem.BondType.AROMATIC:
        return "aromatic"
    return "other"


def _draw_graph_bond(
    axis: Axes,
    start: np.ndarray,
    end: np.ndarray,
    *,
    style_name: str,
    offset: float,
) -> None:
    """Draw single/double/triple/aromatic bonds with chemical conventions."""

    style = BOND_STYLE_METADATA[style_name]
    direction = end - start
    length = float(np.linalg.norm(direction))
    perpendicular = (
        np.array([-direction[1], direction[0]], dtype=float) / length
        if length > 0
        else np.zeros(2, dtype=float)
    )
    line_count = int(style["line_count"])
    if line_count == 1:
        offsets = [0.0]
    elif line_count == 2:
        offsets = [-offset / 2.0, offset / 2.0]
    else:
        offsets = [-offset, 0.0, offset]

    for displacement in offsets:
        shift = perpendicular * displacement
        axis.plot(
            [start[0] + shift[0], end[0] + shift[0]],
            [start[1] + shift[1], end[1] + shift[1]],
            color="#334155",
            linewidth=2.0,
            linestyle=str(style["linestyle"]),
            zorder=1,
            solid_capstyle="round",
        )


def plot_molecule_and_graph(
    smiles: str,
    *,
    molecule_title: str = "RDKit 2D molecular structure",
    graph_title: str = "Molecular graph (atoms and bonds)",
    include_atom_indices: bool = True,
    show_bond_legend: bool = True,
    figsize: tuple[float, float] = (13.0, 6.0),
    dpi: int = 140,
) -> tuple[Figure, np.ndarray]:
    """Plot an RDKit structure and its scientifically matched graph side by side.

    Graph nodes use element/index/formal-charge labels. Single, double and
    triple bonds contain one, two and three solid lines respectively; aromatic
    bonds are dashed. The function is independent of all training code and
    returns the Matplotlib objects for notebook display or lossless export.
    """

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    molecule = Chem.Mol(molecule)
    rdDepictor.Compute2DCoords(molecule)
    conformer = molecule.GetConformer()
    coordinates = np.asarray(
        [
            [conformer.GetAtomPosition(index).x, conformer.GetAtomPosition(index).y]
            for index in range(molecule.GetNumAtoms())
        ],
        dtype=float,
    )

    figure, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    structure_axis, graph_axis = axes

    draw_options = Draw.MolDrawOptions()
    draw_options.clearBackground = False
    structure_image = Draw.MolToImage(
        molecule,
        size=(900, 700),
        kekulize=False,
        options=draw_options,
    )
    structure_axis.imshow(structure_image)
    structure_axis.set_title(molecule_title, fontsize=13, fontweight="bold")
    structure_axis.axis("off")

    span = np.ptp(coordinates, axis=0)
    coordinate_scale = max(float(span.max()), 1.0)
    bond_offset = 0.055 * coordinate_scale
    for bond in molecule.GetBonds():
        _draw_graph_bond(
            graph_axis,
            coordinates[bond.GetBeginAtomIdx()],
            coordinates[bond.GetEndAtomIdx()],
            style_name=bond_style_name(bond),
            offset=bond_offset,
        )

    atom_colours = [
        ATOM_COLOURS.get(atom.GetSymbol(), "#94A3B8")
        for atom in molecule.GetAtoms()
    ]
    graph_axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=760,
        c=atom_colours,
        edgecolors="#0F172A",
        linewidths=1.4,
        zorder=2,
    )
    for atom, (x_coordinate, y_coordinate) in zip(
        molecule.GetAtoms(),
        coordinates,
        strict=True,
    ):
        graph_axis.text(
            x_coordinate,
            y_coordinate,
            atom_graph_label(atom, include_index=include_atom_indices),
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white" if atom.GetSymbol() != "S" else "#0F172A",
            zorder=3,
        )

    graph_axis.set_title(graph_title, fontsize=13, fontweight="bold")
    graph_axis.set_aspect("equal")
    margin = 0.18 * coordinate_scale
    graph_axis.set_xlim(
        coordinates[:, 0].min() - margin,
        coordinates[:, 0].max() + margin,
    )
    graph_axis.set_ylim(
        coordinates[:, 1].min() - margin,
        coordinates[:, 1].max() + margin,
    )
    graph_axis.axis("off")

    if show_bond_legend:
        handles = [
            Line2D(
                [0],
                [0],
                color="#334155",
                linewidth=(1.7 if name != "triple" else 3.2),
                linestyle=str(BOND_STYLE_METADATA[name]["linestyle"]),
                label=name.capitalize(),
            )
            for name in ("single", "double", "triple", "aromatic")
        ]
        graph_axis.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=4,
            frameon=False,
            fontsize=8.5,
        )

    figure.tight_layout(w_pad=2.5)
    return figure, axes


# Short alias for notebook use.
visualize_molecular_graph = plot_molecule_and_graph
