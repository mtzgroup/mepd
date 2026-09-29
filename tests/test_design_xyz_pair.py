"""Design from XYZ: one structure, or two opened as a reaction (reactant,
then product) like a reaction SMILES; more than two belong in Explore."""

import pytest
from fastapi.testclient import TestClient

from mepd.web.app import create_app

HCN = "3\nHCN\nC 0.000 0.000 0.000\nN 0.000 0.000 1.160\nH 0.000 0.000 -1.070\n"
HNC = "3\nHNC\nC 0.000 0.000 0.000\nN 0.000 0.000 1.170\nH 0.000 0.000 2.160\n"
HNC_REORDERED = "3\nHNC\nH 0.000 0.000 2.160\nC 0.000 0.000 0.000\nN 0.000 0.000 1.170\n"


@pytest.fixture
def client(tmp_path):
    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        yield c


@pytest.mark.parametrize("product, matched_by", [(HNC, "xyz-order"), (HNC_REORDERED, "SLAPMapper")])
def test_two_xyz_structures_open_as_a_reaction(client, product, matched_by):
    d = client.post("/api/design/new", json={"xyz": HCN + product}).json()
    assert d["smiles"] == "C#N"
    assert d["reaction"]["product"]["smiles"] == "[C-]#[NH+]"
    assert d["reaction"]["mapping"]["source"] == matched_by
    assert d["reaction"]["amap"] == [0, 1, 2]


def test_one_structure_is_a_plain_design(client):
    d = client.post("/api/design/new", json={"xyz": HCN}).json()
    assert d["smiles"] == "C#N" and not d.get("reaction")


def test_more_than_two_or_mismatched_atoms_are_refused(client):
    r = client.post("/api/design/new", json={"xyz": HCN + HNC + HCN})
    assert r.status_code == 400 and "Explore" in r.json()["detail"]
    r = client.post("/api/design/new", json={"xyz": HCN + "2\n\nH 0 0 0\nH 0 0 0.74\n"})
    assert r.status_code == 400 and "same atoms" in r.json()["detail"]
