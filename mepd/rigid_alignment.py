"""Kabsch rigid alignment -- shared by `mepd.conformers` (realigning
RDKit's independently-embedded fragments back onto the input complex) and
`mepd.atom_mapping_selection` (scoring candidate mappings by endpoint RMSD).
Depends on nothing but numpy, so importing it never pulls in the heavier
optional chemistry backends (openbabel, RDKit) the way `mepd.helper_functions`
or `mepd.qcdata_structure_helpers` would.
"""
from __future__ import annotations

import numpy as np


def kabsch_align(mobile, target):
    """`mobile` (n x 3) rigidly rotated and translated to best overlay
    `target` (n x 3), without reflection."""
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    if len(mobile) < 2:
        return mobile - mc + tc
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return (mobile - mc) @ rotation + tc
