"""mepd.programs: g-xTB is found where it is installed, else fetched once
from its release (checked against the release's SHA-256). Offline: the
download is faked."""

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

import mepd.programs as programs


def _fake_release(monkeypatch, tmp_path, *, corrupt=False):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as t:
        data = b"#!/bin/sh\necho g-xtb\n"
        info = tarfile.TarInfo("xtb-6.7.1/bin/xtb")
        info.size, info.mode = len(data), 0o644
        t.addfile(info, io.BytesIO(data))
    archive = buf.getvalue()
    digest = hashlib.sha256(archive if not corrupt else b"other").hexdigest()
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        return io.BytesIO(f"{digest}  x.tar.xz\n".encode() if url.endswith(".sha256") else archive)

    monkeypatch.setattr(programs.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(programs.platform, "system", lambda: "Linux")
    monkeypatch.setattr(programs.platform, "machine", lambda: "x86_64")
    for var in ("GXTB_EXECUTABLE", "MEPD_NO_DOWNLOAD", "XTB_EXECUTABLE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MEPD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(programs.shutil, "which", lambda name: None)
    return calls


def test_g_xtb_is_downloaded_once_and_then_found(monkeypatch, tmp_path):
    calls = _fake_release(monkeypatch, tmp_path)
    assert programs.find_gxtb() is None
    exe = programs.gxtb_executable()
    assert exe.endswith("gxtb-2.0.1/bin/xtb") and (tmp_path / "data" / "programs" / "gxtb-2.0.1" / "bin" / "xtb").exists()
    assert len(calls) == 2                       # the archive and its checksum
    assert programs.gxtb_executable() == exe and len(calls) == 2   # found, not fetched again
    assert programs.xtb_executable() == exe      # it stands in for a plain xtb


def test_a_corrupt_download_is_refused(monkeypatch, tmp_path):
    _fake_release(monkeypatch, tmp_path, corrupt=True)
    with pytest.raises(RuntimeError, match="corrupt"):
        programs.gxtb_executable()
    assert programs.find_gxtb() is None


def test_no_download_when_asked_not_to_or_when_one_is_set(monkeypatch, tmp_path):
    calls = _fake_release(monkeypatch, tmp_path)
    monkeypatch.setenv("MEPD_NO_DOWNLOAD", "1")
    assert programs.gxtb_executable() is None and not calls
    monkeypatch.setenv("GXTB_EXECUTABLE", "/opt/my/xtb")
    assert programs.gxtb_executable() == "/opt/my/xtb"


def test_an_unwritable_data_dir_is_a_clear_error(monkeypatch, tmp_path):
    _fake_release(monkeypatch, tmp_path)
    monkeypatch.setenv("MEPD_DATA_DIR", "/proc/no-such-dir/mepd")
    with pytest.raises(RuntimeError, match="could not download g-xTB"):
        programs.gxtb_executable()
