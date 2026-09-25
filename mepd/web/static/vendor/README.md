Third-party browser libraries, vendored so `mepd web` works without internet
access (compute nodes, air-gapped lab servers). Unmodified upstream builds:

| file | package | version | license |
|---|---|---|---|
| preact-htm.module.js | htm/preact standalone (Preact + hooks + htm) | htm 3.1.1 | MIT |
| cytoscape.esm.min.js | cytoscape | 3.30.2 | MIT |
| 3Dmol-min.js | 3dmol | 2.5.5 | BSD-3-Clause |
| fonts/source-serif-4-*.woff2 | @fontsource-variable/source-serif-4 (latin, greek) | 5.3.0 | SIL OFL 1.1 (fonts/LICENSE-source-serif-4.txt) |
| fonts/source-sans-3-*.woff2 | @fontsource-variable/source-sans-3 (latin, greek, latin italic) | 5.3.0 | SIL OFL 1.1 (fonts/LICENSE-source-sans-3.txt) |
| fonts/geist-mono-*.woff2 | @fontsource-variable/geist-mono (latin) | 5.3.0 | SIL OFL 1.1 (fonts/LICENSE-geist-mono.txt) |

To update, re-download from https://cdn.jsdelivr.net/npm/<package>@<version>/...

The fonts match atomsforhumanity.org's typography (Source Serif 4 titles, Source Sans 3 text, Geist Mono code).
