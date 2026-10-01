"""External programs mepd runs, found or (g-xTB) fetched on first use.

g-xTB is not a Python package, so `pip install mepd[web]` cannot bring it;
yet a new workspace's default level is g-xTB. So the first time mepd needs
it and none is installed, it downloads the official release build
(github.com/grimme-lab/g-xtb, GPL-3.0) for this platform, checks it against
the release's SHA-256, and keeps it under the user's data directory. The
g-xTB build is an `xtb` that also runs GFN1/GFN2-xTB (and ALPB/GBSA
solvation), so it stands in for a plain `xtb` too.

Lookup order for g-xTB: $GXTB_EXECUTABLE, `gxtb` / `gxtb-xtb` on PATH, an
earlier manual install under ~/.local/opt/gxtb-*/bin/xtb, mepd's own copy,
then (if allowed) the download. Set MEPD_NO_DOWNLOAD=1 to never download.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

GXTB_VERSION = "2.0.1"
_GXTB_RELEASE = f"https://github.com/grimme-lab/g-xtb/releases/download/v{GXTB_VERSION}"
_GXTB_BUILD = "xtb-6.7.1-gxtb-140526"
# (system, machine) -> release asset
_GXTB_ASSETS = {
    ("Linux", "x86_64"): f"{_GXTB_BUILD}-linux-x86_64.tar.xz",
    ("Darwin", "arm64"): f"{_GXTB_BUILD}-macos-arm64.tar.gz",
    ("Darwin", "x86_64"): f"{_GXTB_BUILD}-macos-x86_64.tar.gz",
    ("Windows", "AMD64"): f"{_GXTB_BUILD}-windows-x86_64.zip",
}


def data_dir() -> Path:
    """Where mepd keeps the programs it fetched ($MEPD_DATA_DIR overrides)."""
    if os.getenv("MEPD_DATA_DIR"):
        return Path(os.environ["MEPD_DATA_DIR"])
    if sys.platform == "win32":
        return Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "mepd"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "mepd"
    return Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "mepd"


def _own_gxtb() -> Path:
    return data_dir() / "programs" / f"gxtb-{GXTB_VERSION}" / "bin" / ("xtb.exe" if sys.platform == "win32" else "xtb")


def find_gxtb() -> Optional[str]:
    """An installed g-xTB build, or None (nothing is downloaded)."""
    if os.getenv("GXTB_EXECUTABLE"):
        return os.environ["GXTB_EXECUTABLE"]
    for name in ("gxtb", "gxtb-xtb"):
        if shutil.which(name):
            return shutil.which(name)
    for manual in sorted((Path.home() / ".local" / "opt").glob("gxtb-*/bin/xtb"), reverse=True):
        if os.access(manual, os.X_OK):
            return str(manual)
    own = _own_gxtb()
    return str(own) if own.exists() else None


def gxtb_executable(download: bool = True) -> Optional[str]:
    """g-xTB, downloading it the first time if it is not installed (unless
    `download` is False or MEPD_NO_DOWNLOAD is set). None if unavailable."""
    found = find_gxtb()
    if found or not download or os.getenv("MEPD_NO_DOWNLOAD"):
        return found
    return str(install_gxtb())


def xtb_executable(download: bool = True) -> Optional[str]:
    """A program that runs GFN-xTB: $XTB_EXECUTABLE, `xtb` on PATH, else the
    g-xTB build (it runs GFN1/GFN2-xTB too)."""
    return os.getenv("XTB_EXECUTABLE") or shutil.which("xtb") or gxtb_executable(download)


def install_gxtb() -> Path:
    """Download, verify and unpack the g-xTB release for this platform into
    data_dir(); returns its executable. Safe to call from several processes
    at once (each unpacks into a temporary folder, the first rename wins)."""
    target = _own_gxtb()
    if target.exists():
        return target
    key = (platform.system(), platform.machine())
    asset = _GXTB_ASSETS.get(key)
    if asset is None:
        raise RuntimeError(f"g-xTB has no release build for {key[0]} {key[1]}: build it from "
                           f"github.com/grimme-lab/g-xtb and set GXTB_EXECUTABLE.")
    home = target.parents[1]
    print(f"mepd: g-xTB is not installed; downloading g-xTB {GXTB_VERSION} for {key[0]} "
          f"(about 40 MB, first use only) into {home} ...", file=sys.stderr, flush=True)
    try:
        home.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=home.parent) as tmp:
            archive = Path(tmp) / asset
            with urllib.request.urlopen(f"{_GXTB_RELEASE}/{asset}", timeout=600) as r, archive.open("wb") as fh:
                shutil.copyfileobj(r, fh)
            with urllib.request.urlopen(f"{_GXTB_RELEASE}/{asset}.sha256", timeout=60) as r:
                want = r.read().decode().split()[0].lower()
            got = hashlib.sha256(archive.read_bytes()).hexdigest()
            if got != want:
                raise RuntimeError(f"the g-xTB download is corrupt (SHA-256 {got}, expected {want})")
            unpacked = Path(tmp) / "unpacked"
            if asset.endswith(".zip"):
                with zipfile.ZipFile(archive) as z:
                    z.extractall(unpacked)
            else:
                with tarfile.open(archive) as t:
                    t.extractall(unpacked, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(unpacked)
            (top,) = [p for p in unpacked.iterdir() if p.is_dir()]   # xtb-6.7.1/{bin,share,...}
            exe = top / "bin" / target.name
            exe.chmod(exe.stat().st_mode | 0o111)
            try:
                top.rename(home)
            except OSError:
                if not target.exists():   # not another process that finished first
                    raise
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"could not download g-xTB ({type(exc).__name__}: {exc}). Install it by hand from "
                           f"github.com/grimme-lab/g-xtb/releases and set GXTB_EXECUTABLE, or connect to the "
                           f"internet and run again.") from exc
    print(f"mepd: g-xTB {GXTB_VERSION} installed: {target}", file=sys.stderr, flush=True)
    return target
