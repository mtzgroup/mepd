"""The methods mepd's features are built on, and where to cite them: served
to the web UI's References tab, so the rest of the UI can explain features
without inline citations.

Network expansion's entries come from `mepd.discovery.network_expansion.
REFERENCES` (also written into its summary.json); the rest mirror the
citations in the docstrings of the code that implements them.
"""

from __future__ import annotations

import re

_DOI = re.compile(r"doi:\s*(10\.\S+?)[.,;)]*$")

_OTHER = [
    {
        "feature": "QM/MM",
        "where": "Put in explicit solvent (QM/MM), QM/MM systems",
        "items": [
            {"what": "Subtractive (ONIOM) QM/MM embedding",
             "cite": ["M. Svensson, S. Humbel, R. D. J. Froese, T. Matsubara, S. Sieber, K. Morokuma, J. Phys. Chem. "
                      "100, 19357 (1996), doi:10.1021/jp962071j"]},
            {"what": "Link atoms capping the QM region",
             "cite": ["U. C. Singh, P. A. Kollman, J. Comput. Chem. 7, 718 (1986), doi:10.1002/jcc.540070604"]},
            {"what": "QM/MM methods (additive scheme, boundaries, mechanical embedding)",
             "cite": ["H. M. Senn, W. Thiel, Angew. Chem. Int. Ed. 48, 1198 (2009), doi:10.1002/anie.200802019"]},
            {"what": "GFN-FF force field (the default environment)",
             "cite": ["S. Spicher, S. Grimme, Angew. Chem. Int. Ed. 59, 15665 (2020), doi:10.1002/anie.202004239"]},
            {"what": "OpenMM (AMBER environments)",
             "cite": ["P. Eastman et al., PLoS Comput. Biol. 13, e1005659 (2017), doi:10.1371/journal.pcbi.1005659"]},
            {"what": "Free-energy paths of the QM region (mean force over environment dynamics)",
             "cite": ["H. Hu, Z. Lu, W. Yang, J. Chem. Theory Comput. 3, 390 (2007), doi:10.1021/ct600240y",
                      "H. Hu, Z. Lu, J. M. Parks, S. K. Burger, W. Yang, J. Chem. Phys. 128, 034105 (2008), "
                      "doi:10.1063/1.2816557"]},
            {"what": "Partial Hessian vibrational analysis (frozen environment)",
             "cite": ["H. Li, J. H. Jensen, Theor. Chem. Acc. 107, 211 (2002), doi:10.1007/s00214-002-0356-6"]},
        ],
    },
    {
        "feature": "Initial path",
        "where": "Settings › Initial path",
        "items": [
            {"what": "Geodesic interpolation (the default)",
             "cite": ["X. Zhu, K. C. Thompson, T. J. Martínez, J. Chem. Phys. 150, 164103 (2019), "
                      "doi:10.1063/1.5090303"]},
            {"what": "Linear synchronous transit (LST)",
             "cite": ["T. A. Halgren, W. N. Lipscomb, Chem. Phys. Lett. 49, 225 (1977), "
                      "doi:10.1016/0009-2614(77)80574-5"]},
            {"what": "Image-dependent pair potential (IDPP)",
             "cite": ["S. Smidstrup, A. Pedersen, K. Stokbro, H. Jónsson, J. Chem. Phys. 140, 214106 (2014), "
                      "doi:10.1063/1.4878664"]},
        ],
    },
    {
        "feature": "Path search",
        "where": "Settings › Advanced",
        "items": [
            {"what": "Energy-weighted spring constants (k, delta_k)",
             "cite": ["doi:10.1021/acs.jctc.1c00462"]},
            {"what": "MLP-GI path method (geodesic path length on the energy surface)",
             "cite": ["Efficient Discovery of Transition States on Machine-Learned Potential Energy Surfaces via "
                      "Geodesic Path Optimization, J. Chem. Theory Comput. (2025), doi:10.1021/acs.jctc.5c01221"]},
            {"what": "FIRE chain optimizer",
             "cite": ["E. Bitzek, P. Koskinen, F. Gähler, M. Moseler, P. Gumbsch, Phys. Rev. Lett. 97, 170201 "
                      "(2006), doi:10.1103/PhysRevLett.97.170201"]},
        ],
    },
    {
        "feature": "Atom mapping",
        "where": "Transition state, Reaction channels › Atom mapping",
        "items": [
            {"what": "SLAPMapper: sequential linear-assignment atom-to-atom mapping",
             "cite": ["S. Koda, “General and scalable atom-to-atom mapping via Weisfeiler-Lehman-like approximate "
                      "graph matching”, ChemRxiv (2025), doi:10.26434/chemrxiv-2025-hthwn"]},
        ],
    },
    {
        "feature": "Reaction channels",
        "where": "Reaction channels › conformers",
        "items": [
            {"what": "Conformer embedding budget from the rotatable-bond count",
             "cite": ["J.-P. Ebejer, G. M. Morris, C. M. Deane, J. Chem. Inf. Model. (2012)"]},
        ],
    },
    {
        "feature": "Conformers",
        "where": "Conformers (a structure's calculations); Reaction channels › conformer pools",
        "items": [
            {"what": "RDKit sampler: ETKDG distance-geometry embedding",
             "cite": ["S. Riniker, G. A. Landrum, J. Chem. Inf. Model. 55, 2562–2574 (2015), "
                      "doi:10.1021/acs.jcim.5b00654"]},
            {"what": "RDKit sampler: MMFF94 relaxation of each embedding",
             "cite": ["T. A. Halgren, J. Comput. Chem. 17, 490–519 (1996), "
                      "doi:10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P"]},
            {"what": "CREST sampler: iterative metadynamics conformer search",
             "cite": ["P. Pracht, F. Bohle, S. Grimme, Phys. Chem. Chem. Phys. 22, 7169–7192 (2020), "
                      "doi:10.1039/C9CP06869D"]},
        ],
    },
    {
        "feature": "Solvent effects and reaction conditions",
        "where": "Solvent effects (a TS search's follow-ups); Explore › Conditions",
        "items": [
            {"what": "Implicit solvent: ALPB (analytical linearized Poisson–Boltzmann), the default",
             "cite": ["S. Ehlert, M. Stahn, S. Spicher, S. Grimme, J. Chem. Theory Comput. 17, 4250–4261 (2021), "
                      "doi:10.1021/acs.jctc.1c00471"]},
            {"what": "Implicit solvent: GBSA (generalized Born, surface area) in xtb",
             "cite": ["S. Grimme, C. Bannwarth, P. Shushkov, J. Chem. Theory Comput. 13, 1989–2009 (2017), "
                      "doi:10.1021/acs.jctc.7b00118"]},
            {"what": "Implicit solvent: CPCM-X",
             "cite": ["M. Stahn, S. Ehlert, S. Grimme, J. Phys. Chem. A 127, 7036–7043 (2023), "
                      "doi:10.1021/acs.jpca.3c04382"]},
            {"what": "GFN2-xTB, which gives the solvation free energy added to other engines' energies",
             "note": "mepd's own composite: E(engine) + [E(GFN2-xTB, solvent) − E(GFN2-xTB, gas)]",
             "cite": ["C. Bannwarth, S. Ehlert, S. Grimme, J. Chem. Theory Comput. 15, 1652–1671 (2019), "
                      "doi:10.1021/acs.jctc.8b01176"]},
            {"what": "Rates, half-lives and predicted outcomes (Eyring equation)",
             "note": "electronic barriers stand in for free energies of activation",
             "cite": ["H. Eyring, J. Chem. Phys. 3, 107–115 (1935), doi:10.1063/1.1749604"]},
        ],
    },
    {
        "feature": "Network properties",
        "where": "Explore › Conditions › Analyze network, Compare",
        "items": [
            {"what": "Degree of rate control (what controls a network-level quantity)",
             "note": "generalized to any quantity: yields, formation rates",
             "cite": ["C. T. Campbell, J. Catal. 204, 520–524 (2001), doi:10.1006/jcat.2001.3396",
                      "C. T. Campbell, ACS Catal. 7, 2770–2779 (2017), doi:10.1021/acscatal.7b00115"]},
            {"what": "Networks as Markov chains: relaxation times, first-passage times",
             "cite": ["D. J. Wales, Int. Rev. Phys. Chem. 25, 237–282 (2006), doi:10.1080/01442350600676921"]},
        ],
    },
    {
        "feature": "Substituent effects",
        "where": "Substituent effects (a TS search's or channels run's follow-ups)",
        "items": [
            {"what": "Hammett σp constants (the electronic scale shifts are read against)",
             "note": "defined for para-substituted benzoic acids; elsewhere a rough donor/acceptor ranking",
             "cite": ["C. Hansch, A. Leo, R. W. Taft, Chem. Rev. 91, 165–195 (1991), doi:10.1021/cr00002a004"]},
            {"what": "3D groups: RDKit ETKDG embedding and MMFF94",
             "cite": ["S. Riniker, G. A. Landrum, J. Chem. Inf. Model. 55, 2562–2574 (2015), "
                      "doi:10.1021/acs.jcim.5b00654"]},
        ],
    },
    {
        "feature": "Valley-ridge inflection",
        "where": "Valley-ridge inflection, Check the bifurcation",
        "items": [
            {"what": "Frequencies orthogonal to the path (projected Hessian along the IRC)",
             "cite": ["W. H. Miller, N. C. Handy, J. E. Adams, J. Chem. Phys. 72, 99 (1980), doi:10.1063/1.438959"]},
            {"what": "Converging to the exact VRI point",
             "cite": ["B. Schmidt, W. Quapp, Theor. Chem. Acc. 132, 1305 (2013)"]},
            {"what": "Covalent radii for bond detection",
             "cite": ["B. Cordero et al., Dalton Trans. 2832 (2008), doi:10.1039/b801115j"]},
        ],
    },
]


def _with_links(entry: dict) -> dict:
    items = []
    for item in entry["items"]:
        cites = []
        for text in item["cite"]:
            m = _DOI.search(text)
            cites.append({"text": text, "url": f"https://doi.org/{m.group(1)}" if m else None})
        items.append({**item, "cite": cites})
    return {**entry, "items": items}


def references() -> list[dict]:
    """[{feature, where, items: [{what, note?, cite: [{text, url}]}]}]"""
    from mepd.discovery.network_expansion import REFERENCES as EXPANSION

    from mepd.discovery.generators import GENERATORS

    expansion = {
        "feature": "Reaction network expansion",
        "where": "Reaction network expansion",
        "items": [{"what": stage.replace("_", " ").capitalize(), "note": m["method"], "cite": m["cite"]}
                  for stage, m in EXPANSION.items()]
        # Product generators beyond the bond rules (e.g. CREST msreact).
        + [{"what": gen.label, "note": ref["method"], "cite": ref["cite"]}
           for gen in GENERATORS.values() for ref in gen.references.values()],
    }
    from mepd.discovery.nanoreactor import REFERENCES as NANOREACTOR

    nanoreactor = {
        "feature": "Nanoreactor",
        "where": "Reaction network expansion › Nanoreactor",
        "items": [{"what": stage.replace("_", " ").capitalize(), "note": m["method"], "cite": m["cite"]}
                  for stage, m in NANOREACTOR.items()],
    }
    kinetics = {
        "feature": "Kinetics",
        "where": "Analyze › Kinetics",
        "items": [
            {"what": "Degree of rate control (which barriers control what forms)",
             "cite": ["C. T. Campbell, J. Catal. 204, 520–524 (2001), doi:10.1006/jcat.2001.3396",
                      "C. T. Campbell, ACS Catal. 7, 2770–2779 (2017), doi:10.1021/acscatal.7b00115"]},
            {"what": "Thermodynamic degree of rate control (which species trap material)",
             "cite": ["C. Stegelmann, A. Andreasen, C. T. Campbell, J. Am. Chem. Soc. 131, 8077–8082 (2009), "
                      "doi:10.1021/ja9000097"]},
        ],
    }
    complexes = {
        "feature": "Complexes",
        "where": "Explore › select species › Combine into a complex",
        "items": [
            {"what": "Dock: automated interaction site screening (aISS) in xtb",
             "cite": ["C. Plett, S. Grimme, Angew. Chem. Int. Ed. 62, e202214477 (2023), doi:10.1002/anie.202214477"]},
            {"what": "Ensemble: CREST conformer search in NCI mode",
             "cite": ["P. Pracht, F. Bohle, S. Grimme, Phys. Chem. Chem. Phys. 22, 7169–7192 (2020), "
                      "doi:10.1039/C9CP06869D"]},
            {"what": "Solvation shell: quantum cluster growth (QCG) in CREST",
             "cite": ["S. Spicher, C. Plett, P. Pracht, A. Hansen, S. Grimme, J. Chem. Theory Comput. 18, 3174–3189 "
                      "(2022), doi:10.1021/acs.jctc.2c00239"]},
            {"what": "Packed: Packmol (when installed; else the nanoreactor's own random packing)",
             "cite": ["L. Martínez, R. Andrade, E. G. Birgin, J. M. Martínez, J. Comput. Chem. 30, 2157–2164 (2009), "
                      "doi:10.1002/jcc.21224"]},
        ],
    }
    from mepd.retro import REFERENCES as RETRO

    retro = {
        "feature": "Retrosynthesis",
        "where": "Reaction network expansion › Retrosynthesis",
        "items": [{"what": stage.replace("_", " ").capitalize().replace("Local-llm", "Local LLM")
                   .replace("Reactiont5", "ReactionT5").replace("Aizynthfinder", "AiZynthFinder"),
                   "note": m["method"], "cite": m["cite"]} for stage, m in RETRO.items()],
    }
    return [_with_links(e) for e in [*_OTHER, expansion, nanoreactor, retro, kinetics, complexes]]
