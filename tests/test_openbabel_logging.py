import importlib

from openbabel import openbabel


def test_qcio_structure_helpers_silences_openbabel_warnings():
    openbabel.obErrorLog.SetOutputLevel(1)
    module = importlib.import_module("mepd.qcio_structure_helpers")
    importlib.reload(module)
    assert openbabel.obErrorLog.GetOutputLevel() == 0
