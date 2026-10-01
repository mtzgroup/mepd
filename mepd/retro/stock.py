"""Stock: the molecules a route may start from (bought, or simply at hand).

A stock is a set of InChIKeys (first block only with `stereo=False`, so a
racemate in stock covers either enantiomer), filled from any of:

    paroutes-n1, paroutes-n5   PaRoutes' building-block sets (CC-BY 4.0,
                               ~13k each, downloaded on first use)
    zinc                       ZINC in-stock, as AiZynthFinder ships it
                               (`mepd retro setup --aizynthfinder` converts it)
    a file                     .smi/.txt (one SMILES or InChIKey per line,
                               first column), .csv (a smiles column)

plus a size rule: `max_heavy` treats any molecule with at most that many
heavy atoms as available (common solvents, reagents and small blocks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from mepd.retro import chem
from mepd.retro.data import data_dir, path

BUILTIN = {"paroutes-n1": "stock_n1.txt", "paroutes-n5": "stock_n5.txt"}
ZINC_KEYS = "zinc_stock_inchikeys.txt"


class Stock:
    def __init__(self, keys: Iterable[str] = (), *, max_heavy: int = 0, names: Iterable[str] = (),
                 stereo: bool = False):
        self.stereo = stereo
        self._keys = {self._norm(k) for k in keys}
        self._packed: list = []    # sorted uint64 arrays of first blocks (large stocks)
        self.max_heavy = int(max_heavy)
        self.names = list(names)

    def _norm(self, key: str) -> str:
        return key if self.stereo else key.split("-")[0]

    def __len__(self) -> int:
        return len(self._keys) + sum(len(a) for a in self._packed)

    def __contains__(self, smiles: str) -> bool:
        return self.why(smiles) is not None

    def why(self, smiles: str) -> Optional[str]:
        """Why the molecule counts as available ('stock' or 'small'), or None."""
        key = chem.inchikey(smiles)
        if key is not None and self._norm(key) in self._keys:
            return "stock"
        if key is not None and self._packed:
            import numpy as np

            h = _pack(np.array([key[:14].encode()], dtype="S14"))[0]
            for arr in self._packed:
                i = np.searchsorted(arr, h)
                if i < len(arr) and arr[i] == h:
                    return "stock"
        if self.max_heavy and chem.heavy_atoms(smiles) <= self.max_heavy:
            return "small"
        return None

    def add(self, smiles_or_keys: Iterable[str]) -> None:
        for s in smiles_or_keys:
            k = s if _is_inchikey(s) else chem.inchikey(s)
            if k:
                self._keys.add(self._norm(k))

    def describe(self) -> str:
        parts = [f"{', '.join(self.names)} ({len(self):,} molecules)"] if self.names else []
        if self.max_heavy:
            parts.append(f"anything with at most {self.max_heavy} heavy atoms")
        return " + ".join(parts) or "empty"


def _pack(blocks):
    """InChIKey first blocks (14 letters, as bytes) -> uint64 (base 26, wrapping):
    17M keys in 140 MB instead of a 1 GB set."""
    import numpy as np

    b = np.frombuffer(np.ascontiguousarray(blocks, dtype="S14").tobytes(), dtype=np.uint8).reshape(-1, 14)
    h = np.zeros(len(b), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for c in range(14):
            h = h * np.uint64(26) + (b[:, c].astype(np.uint64) - np.uint64(65))
    return h


def _packed_keys(fp: Path):
    """A big InChIKey list as a sorted uint64 array, cached next to it."""
    import numpy as np

    cache = fp.with_suffix(".u64.npy")
    if cache.exists() and cache.stat().st_mtime >= fp.stat().st_mtime:
        return np.load(cache)
    raw = np.fromfile(fp, dtype=np.uint8)
    if len(raw) % 28 == 0 and np.all(raw.reshape(-1, 28)[:, 27] == 10):
        blocks = raw.reshape(-1, 28)[:, :14].copy().view("S14").ravel()
    else:
        blocks = np.array([line[:14].encode() for line in _read_lines(fp) if _is_inchikey(line)], dtype="S14")
    arr = np.unique(_pack(blocks))
    np.save(cache, arr)
    return arr


def _is_inchikey(s: str) -> bool:
    return len(s) == 27 and s[14] == "-" and s[25] == "-" and s.replace("-", "").isalpha()


def _read_lines(fp: Path) -> list[str]:
    out = []
    if fp.suffix == ".csv":
        import csv

        with open(fp, newline="") as fh:
            rows = csv.DictReader(fh)
            col = next((c for c in rows.fieldnames or [] if c.lower() in ("smiles", "inchikey", "inchi_key")), None)
            if col is None:
                raise ValueError(f"{fp}: needs a 'smiles' or 'inchikey' column")
            return [r[col].strip() for r in rows if r.get(col)]
    for line in fp.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.split()[0].split(",")[0])
    return out


def _keys_file(fp: Path) -> Path:
    """A SMILES list's InChIKeys, computed once next to it."""
    out = fp.with_suffix(".inchikeys")
    if not out.exists() or out.stat().st_mtime < fp.stat().st_mtime:
        keys = [chem.inchikey(s) for s in _read_lines(fp)]
        out.write_text("\n".join(k for k in keys if k) + "\n")
    return out


def load(sources: Iterable[str] = ("paroutes-n5",), *, max_heavy: int = 0, say=None) -> Stock:
    """A stock from built-in names and/or files."""
    stock = Stock(max_heavy=max_heavy)
    for src in sources:
        src = str(src).strip()
        if not src:
            continue
        if src in BUILTIN:
            fp = _keys_file(path(BUILTIN[src], say=say))
        elif src == "zinc":
            fp = data_dir() / ZINC_KEYS
            if not fp.exists():
                raise FileNotFoundError("The ZINC stock is set up with `mepd retro setup --aizynthfinder` "
                                        "(downloads AiZynthFinder's public data, ~790 MB).")
            if stock.stereo:
                raise ValueError("the ZINC stock is matched without stereo (first InChIKey block) only")
            stock._packed.append(_packed_keys(fp))
            stock.names.append(src)
            continue
        else:
            fp = Path(src).expanduser()
            if not fp.exists():
                raise FileNotFoundError(f"stock {src!r}: not a built-in stock ({', '.join([*BUILTIN, 'zinc'])}) "
                                        "and no such file")
        stock.add(_read_lines(fp))
        stock.names.append(src if src in BUILTIN or src == "zinc" else fp.name)
    return stock
