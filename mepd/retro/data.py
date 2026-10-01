"""Downloaded data for retrosynthesis: where it lives, where it comes from.

Everything is fetched on first use into one cache folder (MEPD_RETRO_DATA,
default ~/.cache/mepd/retro) and checked against the publisher's MD5. Each
file records its source and license here, so `mepd retro setup` can say
what it downloads and under which terms.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

_PAROUTES = "https://zenodo.org/api/records/7341155/files/{}/content"
_ONNX = "https://zenodo.org/api/records/7797465/files/{}/content"


@dataclass(frozen=True)
class DataFile:
    name: str
    url: str
    size_mb: float
    license: str
    what: str


FILES = {f.name: f for f in [
    DataFile("uspto_unique_templates.csv.gz", _PAROUTES.format("uspto_unique_templates.csv.gz"), 3.3,
             "CC-BY-4.0 (PaRoutes, Zenodo 7341155)", "42,554 USPTO retro templates"),
    DataFile("uspto_model.onnx", _ONNX.format("uspto_model.onnx"), 91.5,
             "CC-BY-4.0 (AiZynthFinder models, Zenodo 7797465)", "template policy network"),
    DataFile("uspto_filter_model.onnx", _ONNX.format("uspto_filter_model.onnx"), 16.8,
             "CC-BY-4.0 (AiZynthFinder models, Zenodo 7797465)", "reaction feasibility filter network"),
    DataFile("stock_n1.txt", _PAROUTES.format("stock_n1.txt"), 0.4,
             "CC-BY-4.0 (PaRoutes, Zenodo 7341155)", "PaRoutes n1 stock (13k building blocks)"),
    DataFile("stock_n5.txt", _PAROUTES.format("stock_n5.txt"), 0.4,
             "CC-BY-4.0 (PaRoutes, Zenodo 7341155)", "PaRoutes n5 stock (13k building blocks)"),
]}


def data_dir() -> Path:
    d = Path(os.environ.get("MEPD_RETRO_DATA") or Path.home() / ".cache" / "mepd" / "retro")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _md5(fp: Path) -> str:
    h = hashlib.md5()
    with open(fp, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _zenodo_md5(f: DataFile) -> str:
    """The checksum Zenodo publishes for the file ("" if it can't be read)."""
    import json

    record = f.url.split("/records/")[1].split("/")[0]
    try:
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=30) as r:
            for item in json.load(r)["files"]:
                if item["key"] == f.name:
                    return str(item.get("checksum", "")).removeprefix("md5:")
    except Exception:
        return ""
    return ""


def path(name: str, *, download: bool = True, say: Optional[Callable[[str], None]] = None) -> Optional[Path]:
    """The local copy of a data file, downloaded (and checksum-verified) if
    missing and `download`; None if it is missing and not downloaded."""
    f = FILES[name]
    fp = data_dir() / name
    if fp.exists():
        return fp
    if not download:
        return None
    if say:
        say(f"Downloading {f.what} ({f.size_mb:g} MB, {f.license}) to {fp} ...")
    fd, tmp = tempfile.mkstemp(dir=fp.parent, prefix=f".{name}.")
    os.close(fd)
    try:
        with urllib.request.urlopen(f.url, timeout=60) as r, open(tmp, "wb") as out:
            shutil.copyfileobj(r, out, 1 << 20)
        want = _zenodo_md5(f)
        if want and _md5(Path(tmp)) != want:
            raise RuntimeError(f"{name}: checksum mismatch after download (expected md5 {want})")
        os.replace(tmp, fp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return fp
