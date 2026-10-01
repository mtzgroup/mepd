"""Retrosynthesis: routes from purchasable building blocks to a target.

One search, several single-step methods ("proposers"), one stock:

    templates     reaction templates extracted from USPTO patents, applied
                  with rdchiral; ranked by AiZynthFinder's public template
                  policy network (ONNX, run here with onnxruntime; no
                  AiZynthFinder install) or, without it, by how often each
                  template occurs; optionally screened by its feasibility
                  filter network
    reactiont5    ReactionT5v2, a T5 language model trained to write the
                  reactants of a product (local, transformers)
    local-llm     any open-weight LLM behind a local OpenAI-compatible
                  server (Ollama, llama.cpp, vLLM, LM Studio), asked for
                  disconnections; every answer is checked by RDKit
    aizynthfinder AiZynthFinder's own MCTS planner, run in an isolated
                  environment (it pins numpy < 2)

The search (search.py) is an AND-OR best-first search in the manner of
Retro* over any proposer's steps: a step costs -log(its probability), a
molecule in stock costs nothing. Routes then optionally go through mepd:
each step's atoms are mapped, its two ends built in 3D and optimized, and a
path search + TS + IRC gives its barrier (steps.py, `--verify`).

mepd reimplements the search from the published method (see REFERENCES);
it bundles no code from AiZynthFinder or ReactionT5. Data (templates, ONNX models, stocks) is
downloaded on first use into the cache (data.py), with its license.
"""
from __future__ import annotations

REFERENCES = {
    "search": {
        "method": "AND-OR best-first route search in the manner of Retro* (mepd's reimplementation; no learned value "
                  "function: an unsolved molecule is estimated from its synthetic accessibility)",
        "cite": ["B. Chen, C. Li, H. Dai, L. Song, Proc. ICML 2020, PMLR 119, 1608-1616, arXiv:2006.15820"],
    },
    "templates": {
        "method": "reaction templates extracted from USPTO and applied stereo-correctly with rdchiral; template "
                  "policy and feasibility filter networks from AiZynthFinder's public USPTO models (CC-BY 4.0)",
        "cite": ["C. W. Coley, W. H. Green, K. F. Jensen, J. Chem. Inf. Model. 59, 2529-2537 (2019), "
                 "doi:10.1021/acs.jcim.9b00286",
                 "S. Genheden, A. Thakkar, V. Chadimová, J.-L. Reymond, O. Engkvist, E. Bjerrum, J. Cheminform. 12, "
                 "70 (2020), doi:10.1186/s13321-020-00472-1",
                 "M. H. S. Segler, M. P. Waller, Chem. Eur. J. 23, 5966-5971 (2017), doi:10.1002/chem.201605499",
                 "S. Genheden, O. Engkvist, E. Bjerrum, Digital Discovery 1, 527-539 (2022) (PaRoutes; models and "
                 "stocks), doi:10.1039/D2DD00015F"],
    },
    "reactiont5": {
        "method": "ReactionT5v2 retrosynthesis model (MIT): a T5 language model pre-trained on the Open Reaction "
                  "Database, run locally with beam search",
        "cite": ["T. Sagawa, R. Kojima, J. Cheminform. 17, 126 (2025), doi:10.1186/s13321-025-01075-4"],
    },
    "local-llm": {
        "method": "a local open-weight LLM proposes disconnections; RDKit checks every proposal (valid SMILES, "
                  "atoms of the target accounted for) before it enters the search, as LLM retrosynthesis "
                  "planners do",
        "cite": ["H. Wang et al., LLM-Augmented Chemical Synthesis and Design Decision Programs, ICML 2025, "
                 "arXiv:2505.07027",
                 "S. Sathyanarayana et al., DeepRetro: Retrosynthetic Pathway Discovery using Iterative LLM "
                 "Reasoning, arXiv:2507.07060"],
    },
    "aizynthfinder": {
        "method": "AiZynthFinder (MIT), run as its own program in an isolated environment: template-based Monte "
                  "Carlo tree search with its public USPTO models and ZINC stock",
        "cite": ["L. Saigiridharan, A. K. Hassen, H. Lai, P. Torren-Peraire, O. Engkvist, S. Genheden, "
                 "J. Cheminform. 16, 57 (2024), doi:10.1186/s13321-024-00860-x"],
    },
    "synthetic_accessibility": {
        "method": "SA score (RDKit Contrib) as the cost estimate of a molecule not yet solved",
        "cite": ["P. Ertl, A. Schuffenhauer, J. Cheminform. 1, 8 (2009), doi:10.1186/1758-2946-1-8"],
    },
    "verification": {
        "method": "each step: atoms mapped with SLAPMapper, ends built in 3D (precursors relaxed out of the "
                  "product geometry), optimized, then mepd's recursive path search, TS optimization and IRC",
        "cite": ["S. Koda, ChemRxiv (2025), doi:10.26434/chemrxiv-2025-hthwn"],
    },
}
