# This file is OBH: OBMol Helpers. Contains the methods
# for making the OBMOL transformations easier
from pathlib import Path

from openbabel import openbabel
from mepd.helper_functions import symbol_to_atomic_number


def load_obmol_from_fp(fp: Path) -> openbabel.OBMol:
    """
    takes in a pathlib file path as input and reads it in as an openbabel molecule
    """
    assert isinstance(fp, Path), "fp must be a Path object"
    file_type = fp.suffix[1:]  # get what type of file this is

    obmol = openbabel.OBMol()
    obconversion = openbabel.OBConversion()
    print("using:", fp.resolve())
    obconversion.SetInFormat(file_type)

    obconversion.ReadFile(obmol, str(fp.resolve()))

    return make_copy(obmol)


def from_xyz(coords, symbols):
    obmol = openbabel.OBMol()
    for i in range(len(coords)):
        x, y, z = coords[i]

        symbol = symbols[i]
        atomic_num = symbol_to_atomic_number(symbol)
        atom = openbabel.OBAtom()
        atom.SetVector(x, y, z)
        atom.SetAtomicNum(atomic_num)
        obmol.AddAtom(atom)

    return make_copy(obmol)


def make_copy(obmol):
    copy_obmol = openbabel.OBMol()
    for atom in openbabel.OBMolAtomIter(obmol):
        copy_obmol.AddAtom(atom)

    for bond in openbabel.OBMolBondIter(obmol):
        copy_obmol.AddBond(bond)

    copy_obmol.SetTotalCharge(obmol.GetTotalCharge())
    copy_obmol.SetTotalSpinMultiplicity(obmol.GetTotalSpinMultiplicity())

    return copy_obmol


