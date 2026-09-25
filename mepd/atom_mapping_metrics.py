"""The names of the atom-mapping selection metrics.

Split out from `mepd.atom_mapping_selection` (which implements them, and
re-exports `METRICS` so existing imports keep working) so that light
consumers can name the choices without paying for the scoring code: the web
UI's parameter registry only needs the list to build a dropdown, and it
shells out to the CLI for the actual work, so it deliberately never imports
rdkit or openbabel.

Anything offering these as a choice should read them from here rather than
spelling them out, so a new metric shows up everywhere at once.
"""

from __future__ import annotations

METRICS = ("gi-energy", "geodesic-distance", "path-rmsd", "endpoint-rmsd")
