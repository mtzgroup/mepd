from dataclasses import dataclass
from typing import Any


@dataclass
class NoneConvergedException(Exception):

    trajectory: list
    msg: str
    obj: Any = None

    def __post_init__(self) -> None:
        super().__init__(self.msg)


@dataclass
class EnergiesNotComputedError(Exception):
    msg: str = "Energies not computed."

    def __post_init__(self) -> None:
        super().__init__(self.msg)


@dataclass
class GradientsNotComputedError(Exception):
    msg: str = "Gradients not computed."

    def __post_init__(self) -> None:
        super().__init__(self.msg)


@dataclass
class ElectronicStructureError(Exception):

    msg: str
    obj: Any = None

    def __post_init__(self) -> None:
        # Keep Exception args in sync so CLI/error handlers show the message text.
        super().__init__(self.msg)


@dataclass
class CriticalNEBError(Exception):

    msg: str
    obj: Any = None

    def __post_init__(self) -> None:
        super().__init__(self.msg)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate(text: str, *, limit: int = 280) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)] + "..."


def _find_meaningful_error_line(text: str) -> str | None:
    candidate = text[-12000:] if len(text) > 12000 else text
    lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    if not lines:
        return None
    markers = (
        "error",
        "exception",
        "failed",
        "failure",
        "fatal",
        "die called",
        "cannot",
        "not found",
        "invalid",
        "too many electrons",
        "traceback",
    )
    for line in reversed(lines):
        lowered = line.lower()
        if lowered.startswith("traceback (most recent call last):"):
            continue
        if any(marker in lowered for marker in markers):
            return _truncate(line)
    return _truncate(lines[-1])


def _extract_error_line_from_object(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, dict):
        for field in ("message", "error", "stderr", "stdout", "logs", "traceback"):
            line = _find_meaningful_error_line(_coerce_text(value.get(field)))
            if line:
                return line
        for field in ("results", "return_result"):
            if field in value:
                nested_line = _extract_error_line_from_object(value.get(field))
                if nested_line:
                    return nested_line
        return None

    for field in ("message", "error", "stderr", "stdout", "logs", "traceback"):
        line = _find_meaningful_error_line(_coerce_text(getattr(value, field, None)))
        if line:
            return line

    for field in ("results", "return_result"):
        nested = getattr(value, field, None)
        nested_line = _extract_error_line_from_object(nested)
        if nested_line:
            return nested_line

    return None


def format_exception_message(exc: Exception) -> str:
    if getattr(exc, "msg", None):
        return str(getattr(exc, "msg")).strip()
    text = str(exc).strip()
    if text:
        return text
    return type(exc).__name__


def extract_electronic_structure_error_details(obj: Any, *, limit: int = 8) -> list[str]:
    if obj is None:
        return []

    items = list(obj) if isinstance(obj, (list, tuple)) else [obj]
    details: list[str] = []
    seen: set[str] = set()
    for item in items:
        line = _extract_error_line_from_object(item)
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        details.append(line)
        if len(details) >= limit:
            break

    return details
