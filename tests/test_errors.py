from mepd.errors import (
    ElectronicStructureError,
    EnergiesNotComputedError,
    extract_electronic_structure_error_details,
    format_exception_message,
)


def test_dataclass_exception_messages_are_stringified():
    exc = EnergiesNotComputedError(msg="Energies have not been computed.")
    assert str(exc) == "Energies have not been computed."


def test_format_exception_message_prefers_msg_and_falls_back_to_type():
    assert format_exception_message(EnergiesNotComputedError(msg="missing energies")) == "missing energies"
    assert format_exception_message(Exception()) == "Exception"


def test_extract_electronic_structure_error_details_from_nested_results():
    class FakeResultPayload:
        stderr = "Line 1\nfatal SCF failed to converge after 500 iterations\n"

    class FakeOutput:
        message = ""
        error = ""
        results = FakeResultPayload()

    exc = ElectronicStructureError(msg="Gradient calculation failed.", obj=[FakeOutput(), FakeOutput()])
    details = extract_electronic_structure_error_details(exc.obj)
    assert details == ["fatal SCF failed to converge after 500 iterations"]
