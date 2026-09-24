from dataclasses import dataclass
from abc import ABC, abstractmethod
from mepd.elementarystep import ElemStepResults
from pathlib import Path
import shutil


@dataclass
class PathMinimizer(ABC):
    @abstractmethod
    def optimize_chain(self) -> ElemStepResults: ...

    def write_to_disk(self, fp: Path, write_history=True, write_qcio: bool = False):
        # write output chain
        self.chain_trajectory[-1].write_to_disk(fp, write_qcio=write_qcio)
        validated_ts = getattr(self, "validated_ts", None)
        if validated_ts is not None:
            from mepd.chain import Chain

            folder = Path(fp).with_name(Path(fp).stem + "_validated")
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir()
            Chain.model_validate(
                {"nodes": [validated_ts], "parameters": self.chain_trajectory[-1].parameters}
            ).write_to_disk(folder / "ts.xyz")
            self.validated_irc.write_to_disk(folder / "irc.xyz")

        if write_history:
            out_folder = fp.resolve().parent / (fp.stem + "_history")

            if out_folder.exists():
                shutil.rmtree(out_folder)

            if not out_folder.exists():
                out_folder.mkdir()

            for i, chain in enumerate(self.chain_trajectory):
                fp = out_folder / f"traj_{i}.xyz"
                chain.write_to_disk(fp, write_qcio=write_qcio)
