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

METRICS = ("gi-energy", "geodesic-distance", "path-rmsd", "endpoint-rmsd", "rmsd-geodesic")

# The three offered to users (CLI help, web forms, profile form). The other
# two stay accepted so older profiles and scripts still run.
OFFERED = ("endpoint-rmsd", "geodesic-distance", "rmsd-geodesic")
LABELS = {
    "endpoint-rmsd": "Endpoint RMSD",
    "geodesic-distance": "GI path",
    "rmsd-geodesic": "Filtered GI path",
}
HELP = (
    "How atom mappings (and, in channels, conformer pairs) are compared. Endpoint RMSD (endpoint-rmsd): aligned "
    "RMSD of the two ends, no path (fast, weakest). GI path (geodesic-distance): the length of each candidate's "
    "geodesic interpolation (slower, best). Filtered GI path (rmsd-geodesic): channels rank conformer pairs by "
    "endpoint RMSD first and compute GI paths only for those within --atom-mapping-rmsd-window standard "
    "deviations of the lowest (default 1); a pair's own mapping candidates are always compared by GI path.")
