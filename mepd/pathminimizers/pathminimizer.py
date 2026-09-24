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

        if write_history:
            out_folder = fp.resolve().parent / (fp.stem + "_history")

            if out_folder.exists():
                shutil.rmtree(out_folder)

            if not out_folder.exists():
                out_folder.mkdir()

            for i, chain in enumerate(self.chain_trajectory):
                fp = out_folder / f"traj_{i}.xyz"
                chain.write_to_disk(fp, write_qcio=write_qcio)
