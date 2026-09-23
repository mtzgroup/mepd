"""mepd web: a browser front end over the mepd CLI.

The server keeps a *workspace* (a directory holding a structure library, a
user-drawn reaction graph over those structures, RunInputs profiles, and
job folders) and runs every calculation as a `mepd ...` subprocess inside
that workspace. Calculations therefore behave exactly like their CLI
counterparts (same resume semantics, same output layout), crash in
isolation, and can be cancelled by killing their process group.

Start it with `mepd web [WORKSPACE]` (needs the `web` extra).
"""
