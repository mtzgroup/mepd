from .engine import Engine

_ENGINE_EXPORTS = {
    "QCOPEngine": ("mepd.engines.qcop", "QCOPEngine"),
    "ASEEngine": ("mepd.engines.ase", "ASEEngine"),
    "GXTBCalculator": ("mepd.engines.gxtb", "GXTBCalculator"),
    "ThreeWellPotential": ("mepd.engines.threewell", "ThreeWellPotential"),
    "FlowerPotential": ("mepd.engines.flower", "FlowerPotential"),
}

__all__ = sorted(["Engine", *_ENGINE_EXPORTS])


def __getattr__(name: str):
    if name in _ENGINE_EXPORTS:
        import importlib

        module_name, attr_name = _ENGINE_EXPORTS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
