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

METRICS = ("gi-energy", "geodesic-distance", "path-rmsd", "endpoint-rmsd", "rmsd-geodesic", "snap-gi-xtb", "snap")

# The ones offered to users (CLI help, web forms, profile form). The others
# stay accepted so older profiles and scripts still run ("snap" is also the
# cheap first stage of snap-gi-xtb).
OFFERED = ("snap-gi-xtb", "endpoint-rmsd", "geodesic-distance", "rmsd-geodesic")
LABELS = {
    "snap-gi-xtb": "Snap + GI + xtb",
    "endpoint-rmsd": "Endpoint RMSD",
    "geodesic-distance": "GI path",
    "rmsd-geodesic": "Filtered GI path",
}
HELP = (
    "How atom mappings (and, in channels, conformer pairs) are compared. Snap + GI + xtb (snap-gi-xtb, the "
    "channels default): conformer pairs ranked by the endpoint RMSD of snap's pick (the reactant's symmetric "
    "groups permuted one at a time, as snap-RMSD does), then for the pairs kept, snap's few picks interpolated "
    "(GI) and the one whose GFN2-xTB energy profile peaks lowest taken; without xtb the shortest interpolation, "
    "without a working interpolation the lowest RMSD. Endpoint RMSD (endpoint-rmsd): aligned "
    "RMSD of the two ends, no path (fast, weakest). GI path (geodesic-distance): the length of each candidate's "
    "geodesic interpolation (slower, best). Filtered GI path (rmsd-geodesic): channels rank conformer pairs by "
    "endpoint RMSD first and compute GI paths only for those within --atom-mapping-rmsd-window standard "
    "deviations of the lowest (default 1); a pair's own mapping candidates are always compared by GI path.")
