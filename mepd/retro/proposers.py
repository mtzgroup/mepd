"""Single-step retrosynthesis: for a molecule, the precursor sets it can be
made from, each with a probability-like score.

A proposer is built once per run (`make(name, **options)`) and called as
`proposer(smiles, k)` -> list[Step], best first. Every proposal goes
through `_accept` (RDKit-valid, canonical, not the target itself), so a
proposer that hallucinates (an LLM) can't put a bad molecule in a route.

To add a method: a factory returning such a callable, and a `Proposer`
entry in PROPOSERS (see the docstring of mepd.retro for the built-in ones).
"""
from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from mepd.retro import chem


@dataclass
class Step:
    """product <= reactants: one retrosynthetic step."""
    product: str
    reactants: tuple[str, ...]
    score: float                     # in (0, 1]; the search's cost is -log(score)
    method: str
    info: dict = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return -math.log(max(self.score, 1e-6))

    def to_json(self) -> dict:
        return {"product": self.product, "reactants": list(self.reactants), "score": self.score,
                "method": self.method, **({"info": self.info} if self.info else {})}


@dataclass(frozen=True)
class Proposer:
    name: str
    label: str
    summary: str
    factory: Callable[..., Callable[[str, int], list[Step]]]
    packages: tuple = ()
    install: str = ""
    options: dict = field(default_factory=dict)
    references: tuple = ()
    multistep: bool = False          # plans whole routes itself (aizynthfinder): not used inside the search

    def available(self) -> bool:
        return all(importlib.util.find_spec(p) is not None for p in self.packages) and self._extra_ok()

    def _extra_ok(self) -> bool:
        if self.name == "aizynthfinder":
            from mepd.retro.aizynth import installed

            return installed()
        return True

    def problem(self) -> Optional[str]:
        missing = [p for p in self.packages if importlib.util.find_spec(p) is None]
        if missing:
            return f"{self.label} needs {', '.join(missing)}: {self.install}"
        if not self._extra_ok():
            return f"{self.label} is not set up: {self.install}"
        return None


def _accept(product: str, reactants, score: float, method: str, info: Optional[dict] = None) -> Optional[Step]:
    """A checked step, or None: every precursor valid, none the product."""
    canon = []
    for r in reactants:
        c = chem.canonical(r)
        if c is None:
            return None
        canon.extend(c.split("."))
    target = chem.canonical(product)
    if not canon or target in canon:
        return None
    return Step(target, tuple(sorted(canon)), float(score), method, info or {})


def _dedupe(steps: list[Step], k: int) -> list[Step]:
    seen, out = set(), []
    for s in sorted(steps, key=lambda s: -s.score):
        if s.reactants not in seen:
            seen.add(s.reactants)
            out.append(s)
    return out[:k]


# ----------------------------------------------------------------- templates
class _Templates:
    """rdchiral retro templates; ranked by the ONNX policy (AiZynthFinder's
    public USPTO model) when it is there, else by how often each template
    occurs in USPTO; the feasibility filter screens the outcomes."""

    def __init__(self, *, policy: bool = True, filter: bool = True, top_templates: int = 50,
                 cumulative: float = 0.995, filter_cutoff: float = 0.05, min_occurrence: int = 1,
                 download: bool = True, say=None):
        import gzip

        import pandas as pd

        from mepd.retro.data import path

        with gzip.open(path("uspto_unique_templates.csv.gz", download=download, say=say), "rt") as fh:
            df = pd.read_csv(fh, sep="\t", index_col=0)
        self.smarts = df["retro_template"].tolist()
        self.occurrence = df["library_occurence"].to_numpy(dtype=float)
        self.top, self.cumulative, self.cutoff = int(top_templates), float(cumulative), float(filter_cutoff)
        self.min_occurrence = int(min_occurrence)
        self.policy = self.filter = None
        if policy or filter:
            if importlib.util.find_spec("onnxruntime") is None:
                if say:
                    say("onnxruntime is not installed: templates are ranked by how often they occur "
                        "(pip install 'mepd[retro]' for the policy network).")
            else:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                if policy:
                    self.policy = ort.InferenceSession(str(path("uspto_model.onnx", download=download, say=say)),
                                                       opts, providers=["CPUExecutionProvider"])
                if filter:
                    self.filter = ort.InferenceSession(
                        str(path("uspto_filter_model.onnx", download=download, say=say)), opts,
                        providers=["CPUExecutionProvider"])
        self._rxn: dict = {}
        self._product_patterns: dict = {}
        if self.policy is None:
            self._prior = self.occurrence / self.occurrence.sum()

    def _reaction(self, idx: int):
        from rdchiral.main import rdchiralReaction

        if idx not in self._rxn:
            try:
                self._rxn[idx] = rdchiralReaction(self.smarts[idx])
            except Exception:
                self._rxn[idx] = None
        return self._rxn[idx]

    def _matches(self, idx: int, m) -> bool:
        """Quick screen: does the template's product side occur in the molecule at all?"""
        from rdkit import Chem

        if idx not in self._product_patterns:
            self._product_patterns[idx] = Chem.MolFromSmarts(self.smarts[idx].split(">>")[0])
        p = self._product_patterns[idx]
        return p is not None and m.HasSubstructMatch(p)

    def ranked(self, smiles: str) -> list[tuple[int, float]]:
        """(template index, prior) to try, best first."""
        if self.policy is not None:
            fp = chem.fingerprint(smiles)[None, :]
            probs = self.policy.run(None, {self.policy.get_inputs()[0].name: fp})[0][0]
            order = np.argsort(probs)[::-1]
            out, total = [], 0.0
            for idx in order[:self.top]:
                out.append((int(idx), float(probs[idx])))
                total += float(probs[idx])
                if total >= self.cumulative:
                    break
            return out
        m = chem.mol(smiles)
        out = []
        for idx in np.argsort(self.occurrence)[::-1]:
            if self.occurrence[idx] < self.min_occurrence:
                break
            if self._matches(int(idx), m):
                out.append((int(idx), float(self._prior[idx])))
                if len(out) >= self.top:
                    break
        return out

    def feasibility(self, product: str, reactants) -> float:
        if self.filter is None:
            return 1.0
        prod = chem.fingerprint(product)
        rxn = prod - sum(chem.fingerprint(r) for r in reactants)
        names = [i.name for i in self.filter.get_inputs()]
        return float(self.filter.run(None, {names[0]: prod[None, :], names[1]: rxn[None, :]})[0][0][0])

    def __call__(self, smiles: str, k: int = 10) -> list[Step]:
        from rdchiral.main import rdchiralReactants, rdchiralRun

        try:
            rct = rdchiralReactants(smiles)
        except Exception:
            return []
        steps = []
        ranked = self.ranked(smiles)
        norm = sum(p for _, p in ranked) or 1.0
        for idx, prior in ranked:
            rxn = self._reaction(idx)
            if rxn is None:
                continue
            try:
                outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
            except Exception:
                continue
            for out in outcomes:
                reactants = out.split(".")
                feas = self.feasibility(smiles, reactants)
                if feas < self.cutoff:
                    continue
                step = _accept(smiles, reactants, prior / norm, "templates",
                               {"template": idx, "prior": round(prior, 4),
                                "feasibility": round(feas, 3)})
                if step is not None:
                    steps.append(step)
        return _dedupe(steps, k)


def _templates(**options):
    return _Templates(**options)


# ---------------------------------------------------------------- reactiont5
class _ReactionT5:
    """Beam search of the retrosynthesis model. The default model (trained on
    the Open Reaction Database) also writes reagents, often as ions
    ([Al+3].[H-]... for LiAlH4); carbon-free ions and metal species are
    moved to the step's info as reagents, not precursors to make. With
    `roundtrip`, the forward model is given each proposal: one that does not
    give the target back among its top 3 keeps `roundtrip_penalty` of its
    weight (a soft check: the forward model misses reactions written without
    their reagents, while template-free models can write plausible but wrong
    reactants)."""

    def __init__(self, *, model: str = "sagawa/ReactionT5v2-retrosynthesis",
                 forward_model: str = "sagawa/ReactionT5v2-forward", roundtrip: bool = True,
                 roundtrip_penalty: float = 0.2, device: str = "auto",
                 beams: int = 10, max_length: int = 300, threads: int = 4, say=None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.set_num_threads(int(threads))   # torch takes every core otherwise
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, self.beams, self.max_length, self.name = device, int(beams), int(max_length), model
        self.penalty = float(roundtrip_penalty)
        names = [model] + ([forward_model] if roundtrip else [])
        if say:
            say(f"Loading {' and '.join(names)} (from Hugging Face on first use, ~0.8 GB each) on {device} ...")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(device).eval()
        self.fwd = self.fwd_tok = None
        if roundtrip:
            self.fwd_tok = AutoTokenizer.from_pretrained(forward_model)
            self.fwd = AutoModelForSeq2SeqLM.from_pretrained(forward_model).to(device).eval()

    def _generate(self, model, tok, texts: list[str], n: int):
        import torch

        inp = tok(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = model.generate(**inp, num_beams=n, num_return_sequences=n, max_length=self.max_length,
                                 return_dict_in_generate=True, output_scores=True)
        seqs = tok.batch_decode(out.sequences, skip_special_tokens=True)
        # Beam scores are log-probabilities divided by length: undo that, so a
        # softmax over the beams gives the model's own preference.
        lengths = (out.sequences != tok.pad_token_id).sum(dim=1).cpu().numpy()
        raw = getattr(out, "sequences_scores", None)
        scores = raw.cpu().numpy() * np.maximum(lengths - 1, 1) if raw is not None else np.zeros(len(seqs))
        return [s.replace(" ", "").rstrip(".") for s in seqs], scores

    def _roundtrip(self, target: str, steps: list[Step]) -> list[Step]:
        if self.fwd is None or not steps:
            return steps
        texts = ["REACTANT:" + ".".join([*s.reactants, *s.info.get("reagents", [])]) + "REAGENT: " for s in steps]
        preds, _ = self._generate(self.fwd, self.fwd_tok, texts, 3)
        for k, s in enumerate(steps):
            got = {chem.canonical(p) for p in preds[3 * k:3 * k + 3]}
            # the target among the forward model's top 3 (or a fragment of one: it may add byproducts)
            hit = target in got or any(p and target in p.split(".") for p in got)
            s.info["roundtrip"] = bool(hit)
            if not hit:
                s.score *= self.penalty
        return steps

    def __call__(self, smiles: str, k: int = 10) -> list[Step]:
        n = max(k, self.beams)
        texts, logp = self._generate(self.model, self.tok, [smiles], n)
        w = np.exp(logp - logp.max())
        w /= w.sum()
        steps = []
        for text, p in zip(texts, w):
            parts = [s for s in text.split(".") if s]
            reagents = sorted({chem.canonical(s) or s for s in parts if is_reagent(s)})
            info = {"model": self.name, **({"reagents": reagents} if reagents else {})}
            step = _accept(smiles, [s for s in parts if not is_reagent(s)], float(p), "reactiont5", info)
            if step is not None:
                steps.append(step)
        steps = self._roundtrip(chem.canonical(smiles), _dedupe(steps, n))
        total = sum(s.score for s in steps) or 1.0
        for s in steps:
            s.score /= total
        return _dedupe(steps, k)


_METALS = {"Li", "Na", "K", "Cs", "Mg", "Ca", "Zn", "Al", "B", "Pd", "Pt", "Ni", "Cu", "Fe", "Mn", "Cr", "Ti", "Sn",
           "Ag", "Rh", "Ru", "Os", "Ir", "Co", "Ce", "Hg", "Ba", "Sr", "Rb"}


def is_reagent(smiles: str) -> bool:
    """A species a step uses rather than builds on: carbon-free ions and
    anything holding a metal ([Na+], [OH-], [H-], [Al+3], Pd/C...)."""
    m = chem.mol(smiles)
    if m is None:
        return False
    symbols = {a.GetSymbol() for a in m.GetAtoms()}
    charged = any(a.GetFormalCharge() for a in m.GetAtoms())
    return bool(symbols & _METALS) or (charged and "C" not in symbols)


def _reactiont5(**options):
    return _ReactionT5(**options)


# ----------------------------------------------------------------- local LLM
LLM_PROMPT = """You are an expert synthetic organic chemist doing retrosynthetic analysis.
Target molecule (SMILES): {smiles}

Propose up to {k} different single-step disconnections: for each, the reactants
(starting materials and the key reagent that contributes atoms) that give the
target in ONE well-known reaction. Prefer reliable, commonly used reactions and
commercially available or simpler precursors. Give each a confidence in (0, 1].

Answer with JSON only, no prose:
{{"steps": [{{"reaction": "<name of the reaction>", "reactants": ["<SMILES>", "<SMILES>"], "confidence": 0.8}}]}}"""


class _LocalLLM:
    """An OpenAI-compatible chat endpoint on this machine (Ollama:
    http://localhost:11434/v1, llama.cpp server: http://localhost:8080/v1,
    vLLM / LM Studio likewise). Nothing leaves the machine unless you point
    `url` elsewhere."""

    def __init__(self, *, url: str = "http://localhost:11434/v1", model: str = "", temperature: float = 0.3,
                 timeout: float = 300.0, samples: int = 1, say=None):
        self.url, self.temperature, self.timeout, self.samples = url.rstrip("/"), float(temperature), \
            float(timeout), int(samples)
        self.model = model or self._first_model()
        if say:
            say(f"Local LLM: {self.model} at {self.url}")

    def _request(self, route: str, payload: Optional[dict] = None) -> dict:
        import json
        import urllib.request

        req = urllib.request.Request(f"{self.url}{route}", data=None if payload is None else
                                     json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def _first_model(self) -> str:
        try:
            models = [m["id"] for m in self._request("/models").get("data", [])]
        except Exception as exc:
            raise RuntimeError(f"no local LLM server answers at {self.url} ({exc}); start one (e.g. `ollama serve` "
                               "and `ollama pull qwen3:8b`) or pass url=...") from None
        if not models:
            raise RuntimeError(f"the server at {self.url} has no models loaded")
        return models[0]

    def ask(self, smiles: str, k: int) -> str:
        msg = [{"role": "user", "content": LLM_PROMPT.format(smiles=smiles, k=k)}]
        res = self._request("/chat/completions", {"model": self.model, "messages": msg,
                                                  "temperature": self.temperature, "stream": False})
        return res["choices"][0]["message"]["content"]

    def __call__(self, smiles: str, k: int = 10) -> list[Step]:
        steps = []
        for _ in range(self.samples):
            try:
                text = self.ask(smiles, k)
            except Exception:
                continue
            steps.extend(parse_llm_steps(text, smiles, self.model))
        # Repeated proposals across samples are more trustworthy: sum their confidences.
        merged: dict = {}
        for s in steps:
            if s.reactants in merged:
                merged[s.reactants].score += s.score
            else:
                merged[s.reactants] = s
        out = list(merged.values())
        total = sum(s.score for s in out) or 1.0
        for s in out:
            s.score /= total
        return _dedupe(out, k)


def parse_llm_steps(text: str, smiles: str, model: str = "") -> list[Step]:
    """Steps from an LLM answer: the JSON object in it (reasoning models put
    a <think> block first), each proposal kept only if RDKit reads every
    reactant and the reactants supply the product's heavy atoms."""
    import json
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return []
    items = data.get("steps") if isinstance(data, dict) else data
    steps = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        reactants = [str(r) for r in it.get("reactants") or [] if r]
        try:
            conf = min(1.0, max(1e-3, float(it.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        if not reactants or not _supplies_heavy_atoms(reactants, smiles):
            continue
        step = _accept(smiles, reactants, conf, "local-llm", {"reaction": str(it.get("reaction", ""))[:120],
                                                              "model": model})
        if step is not None:
            steps.append(step)
    return steps


def _supplies_heavy_atoms(reactants, product) -> bool:
    try:
        have = chem.formula(reactants)
        need = chem.formula([product])
    except ValueError:
        return False
    return all(have[el] >= n for el, n in need.items() if el not in ("H", "+"))


def _aizynth(**options):
    raise RuntimeError("aizynthfinder plans whole routes; run it through mepd.retro.aizynth.plan")


PROPOSERS: dict[str, Proposer] = {p.name: p for p in [
    Proposer("templates", "Templates",
             "USPTO reaction templates (rdchiral) ranked by AiZynthFinder's policy network, run here. Seconds.",
             _templates, packages=("rdchiral",), install="pip install 'mepd[retro]'",
             options={"policy": True, "filter": True, "top_templates": 50, "filter_cutoff": 0.05},
             references=("search", "templates")),
    Proposer("reactiont5", "ReactionT5",
             "A local T5 language model trained to write the reactants of a product (beam search). Seconds.",
             _reactiont5, packages=("transformers", "torch"), install="pip install 'mepd[retro-ml]'",
             options={"model": "sagawa/ReactionT5v2-retrosynthesis", "beams": 10, "roundtrip": True,
                      "device": "auto"},
             references=("search", "reactiont5")),
    Proposer("local-llm", "Local LLM",
             "An open-weight LLM on your machine (Ollama, llama.cpp, vLLM) proposes disconnections; RDKit checks "
             "each one.", _LocalLLM, install="run a local OpenAI-compatible server, e.g. `ollama serve`",
             options={"url": "http://localhost:11434/v1", "model": "", "temperature": 0.3, "samples": 1},
             references=("search", "local-llm")),
    Proposer("aizynthfinder", "AiZynthFinder",
             "AiZynthFinder's own tree search with its public USPTO models and ZINC stock, in its own "
             "environment.", _aizynth, install="mepd retro setup --aizynthfinder (~790 MB)",
             references=("aizynthfinder",), multistep=True),
]}


def make(name: str, **options) -> Callable[[str, int], list[Step]]:
    try:
        p = PROPOSERS[name]
    except KeyError:
        raise ValueError(f"unknown retrosynthesis method {name!r}; choose from {', '.join(PROPOSERS)}") from None
    problem = p.problem()
    if problem:
        raise RuntimeError(problem)
    return p.factory(**options)
