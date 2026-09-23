Third-party browser libraries, vendored so `mepd web` works without internet
access (compute nodes, air-gapped lab servers). Unmodified upstream builds:

| file | package | version | license |
|---|---|---|---|
| preact-htm.module.js | htm/preact standalone (Preact + hooks + htm) | htm 3.1.1 | MIT |
| cytoscape.esm.min.js | cytoscape | 3.30.2 | MIT |
| 3Dmol-min.js | 3dmol | 2.5.5 | BSD-3-Clause |

To update, re-download from https://cdn.jsdelivr.net/npm/<package>@<version>/...
