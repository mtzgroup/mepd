# Minimal runtime for the public mepd web demo.
#
# Nothing of mepd is baked in: run.sh mounts this checkout, its .venv, the
# uv-managed Python and the g-xTB / CREST / GSM installs read-only at their
# host paths, so the container runs exactly the code you run -- but can only
# write to the demo data directory. Same Ubuntu release as the host so the
# host-built GSM binary finds a matching libstdc++. The X libraries are for
# the Cairo bundled in openbabel-wheel (its plugins fail to load without them).
FROM ubuntu:26.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libgfortran5 libstdc++6 ca-certificates \
      libxrender1 libxext6 libx11-6 libexpat1 \
 && rm -rf /var/lib/apt/lists/*
