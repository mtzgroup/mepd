from types import SimpleNamespace

import numpy as np
import pytest

from mepd.helper_functions import parse_nma_freq_data


def _fake_hessian_result(files):
    return SimpleNamespace(
        input_data=SimpleNamespace(
            structure=SimpleNamespace(geometry=np.zeros((2, 3), dtype=float))
        ),
        results=SimpleNamespace(files=files, trajectory=[]),
    )


def test_parse_nma_freq_data_accepts_absolute_scr_geometry_key():
    nma_text = "\n".join(
        [
            "=== mode 0 -123.4 ===",
            "1.0",
            "0.0",
            "0.0",
            "0.0",
            "1.0",
            "0.0",
            "=== mode 1 55.6 ===",
            "0.0",
            "0.0",
            "1.0",
            "1.0",
            "0.0",
            "0.0",
        ]
    )
    hessres = _fake_hessian_result(
        {
            "/tmp/tmpu2_wxfam/scr.geometry/Mass.weighted.modes.dat": nma_text,
        }
    )

    modes, freqs = parse_nma_freq_data(hessres)

    assert len(modes) == 2
    assert modes[0].shape == (2, 3)
    assert modes[1].shape == (2, 3)
    assert freqs == pytest.approx([-123.4, 55.6])


def test_parse_nma_freq_data_reports_available_keys_when_modes_file_missing():
    hessres = _fake_hessian_result({"not_the_modes_file.txt": "content"})

    with pytest.raises(KeyError, match="Mass\\.weighted\\.modes\\.dat"):
        parse_nma_freq_data(hessres)


def test_parse_nma_freq_data_falls_back_to_hessian_matrix_when_modes_file_missing():
    hessian = np.diag([-4.0, -1.0, 0.25, 1.0, 4.0, 9.0])
    hessres = _fake_hessian_result(
        {
            "geometry.xyz": "ignored",
            "tc.in": "ignored",
        }
    )
    hessres.results.hessian = hessian

    modes, freqs = parse_nma_freq_data(hessres)

    assert len(modes) == 6
    assert all(mode.shape == (2, 3) for mode in modes)
    assert freqs == pytest.approx([-2.0, -1.0, 0.5, 1.0, 2.0, 3.0])
