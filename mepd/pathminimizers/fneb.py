from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from mepd.geodesic_interpolation2.morsegeodesic import MorseGeodesic

import mepd.chainhelpers as ch
from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.nodes.node import StructureNode
from mepd.helper_functions import RMSD, get_maxene_node, project_rigid_body_forces
from mepd.pathminimizers.pathminimizer import PathMinimizer
from mepd.optimizers.optimizer import Optimizer
from mepd.elementarystep import IS_ELEM_STEP, elem_step_check_kwargs, check_if_elem_step
from mepd.inputs import RunInputs
from mepd.progress import log_at_level, print_chain_step

import traceback

DRSTEP = 0.1
PHI = 0.5


DISTANCE_METRICS = ["GEODESIC", "RMSD", "LINEAR"]


def _valid_tangent(tangent: np.ndarray | None) -> bool:
    if tangent is None:
        return False
    return bool(np.all(np.isfinite(tangent)) and np.linalg.norm(tangent) > 0)


def _linear_tangent(chain: Chain, ind_node: int) -> np.ndarray:
    return chain[ind_node + 1].coords - chain[ind_node - 1].coords


def _geodesic_drstep(parameters: SimpleNamespace) -> float:
    return float(getattr(parameters, "drstep", DRSTEP))


@dataclass
class FreezingNEB(PathMinimizer):
    initial_chain: Chain
    engine: Engine
    optimizer: Optimizer
    chain_trajectory: list[Chain] = field(default_factory=list)
    parameters: SimpleNamespace = None
    gi_inputs: SimpleNamespace = None

    def __post_init__(self):

        if self.parameters is None:
            ri = RunInputs(path_min_method='FNEB')
            self.parameters = ri.path_min_inputs
        if self.gi_inputs is None:
            ri = RunInputs(path_min_method='FNEB')
            self.gi_inputs = ri.gi_inputs

        self.grad_calls_made = 0
        self.geom_grad_calls_made = 0

    def _log(self, *parts, level: str = "info", verbose: int = 1):
        if self.parameters.verbosity < verbose:
            return
        log_at_level(" ".join(str(p) for p in parts), level)

    def _append_chain_snapshot(self, chain: Chain, caption: str) -> None:
        snapshot = chain.copy()
        self.chain_trajectory.append(snapshot)
        print_chain_step(snapshot, caption, force_update=True)

    def _distance_function(self, node1: StructureNode, node2: StructureNode):
        if self.parameters.distance_metric.upper() == "RMSD":
            return RMSD(node1.coords, node2.coords)[0]
        elif self.parameters.distance_metric.upper() == "GEODESIC":
            return ch.calculate_geodesic_distance(
                node1,
                node2,
                nudge=self.gi_inputs.nudge,
                nimages=self.gi_inputs.nimages,
                random_seed=self.gi_inputs.random_seed,
            )
        elif self.parameters.distance_metric.upper() == "LINEAR":
            return np.linalg.norm(node1.coords - node2.coords)
        elif self.parameters.distance_metric.upper() == "XTBGI":
            return abs(ch.calculate_geodesic_xtb_barrier(node1, node2))
        else:
            raise ValueError(
                f"Invalid distance metric: {self.parameters.distance_metric}. Use one of {DISTANCE_METRICS}"
            )

    def optimize_chain(
        self,
    ):
        """
        will run freezing string on chain.
        dr --> the requested path resolution in Bohr
        """
        np.random.seed(0)
        chain = self.initial_chain.copy()
        ch._reset_cache(chain=chain)
        self.optimizer.g_old = None
        chain.nodes = [
            chain.nodes[0],
            chain.nodes[-1],
        ]  # need to make sure I only use the endpoints
        self.engine.compute_energies(chain)
        self.grad_calls_made += 2
        chain.nodes[0].converged = True
        chain.nodes[1].converged = True
        self.chain_trajectory = [chain]

        d0 = self._distance_function(chain[0], chain[1])
        self.d0 = d0
        # +1 so that if only one image is requested, node is placed at 50% of path
        dr = d0 / (self.parameters.min_images+1)

        converged = False
        nsteps = 0
        last_grown_ind = 0

        while not converged and nsteps < self.parameters.max_grow_iter:
            self._log(f"FNEB step {nsteps}")

            # grow nodes
            self._log("Growing nodes")

            if self.parameters.todd_way:
                self._log(f"Growing at index {last_grown_ind}")
                grown_chain, tangents, idx_grown, dr = self.grow_nodes(
                    chain, dr=dr, indices=(last_grown_ind, last_grown_ind+1)
                )
                self._append_chain_snapshot(
                    grown_chain, f"FNEB grow step {nsteps}"
                )
                self._log(grown_chain.energies, verbose=2)

                # this section will stop opt if grown nodes are too low in energy
                node0_done = grown_chain[idx_grown[0]
                                         ].energy < grown_chain.get_ts_node().energy
                node1_done = True
                if idx_grown[1] is not None:
                    node1_done = grown_chain[idx_grown[1]
                                             ].energy < grown_chain.get_ts_node().energy

                if node0_done and node1_done:
                    if nsteps == 0:  # if this is the first step, we need to make sure that the grown nodes are higher than both endpoints
                        result = self.grow_nodes_maxene(
                            chain, last_grown_ind=last_grown_ind, nimg=self.gi_inputs.nimages, nudge=self.gi_inputs.nudge)
                        grown_chain = result[0]
                        self._log(
                            "Initial grown node was lower than endpoints. Using max-ene growth.",
                            level="warning",
                        )
                        idx_grown = (idx_grown[0], None)
                        self._append_chain_snapshot(
                            grown_chain, f"FNEB max-energy grow step {nsteps}"
                        )
                    else:
                        self._log(
                            "Grown nodes are lower in energy than TS guess. Stopping optimization.",
                            level="warning",
                        )
                        converged = True
                        break

                min_chain = self.minimize_nodes(
                    chain=grown_chain, node_tangents=tangents, dr=dr, idx_grown=idx_grown
                )

                # check convergence
                self._log("Checking convergence")
                converged, last_grown_ind = self.chain_converged(
                    min_chain, dr, indices=idx_grown)

                self._log(f"Last grown index: {last_grown_ind}")

                self.optimizer.g_old = None
            else:
                result = self.grow_nodes_maxene(
                    chain, last_grown_ind=last_grown_ind, nimg=self.gi_inputs.nimages, nudge=self.gi_inputs.nudge)
                grown_chain, node_ind = result[0], result[1]

                no_growth = len(grown_chain) == len(chain)
                no_barrier_change = abs(grown_chain.get_eA_chain(
                ) - chain.get_eA_chain()) < self.parameters.barrier_thre

                if no_growth or no_barrier_change:
                    self._log(
                        f"Converged! No growth: {no_growth} | No barrier change: {no_barrier_change}",
                        level="success",
                    )
                    converged = True
                    min_chain = grown_chain.copy()
                else:

                    self._append_chain_snapshot(
                        grown_chain, f"FNEB max-energy grow step {nsteps}"
                    )

                    # minimize nodes
                    self._log("Minimizing nodes")
                    min_chain = self.minimize_node_maxene(chain=grown_chain, node_ind=node_ind)

                    self._append_chain_snapshot(
                        min_chain, f"FNEB minimize step {nsteps}"
                    )
                    last_grown_ind = node_ind
                    self._log(f"Last grown index: {last_grown_ind}")

            chain = min_chain.copy()
            nsteps += 1

            self.optimized = self.chain_trajectory[-1]
            self._log(f"Converged? {converged}")

        if self.parameters.do_elem_step_checks:
            short_chain = Chain.model_validate(
                {"nodes": [chain[0], chain.get_ts_node(), chain[-1]], "parameters": chain.parameters})
            elem_step_results = check_if_elem_step(
                short_chain,
                engine=self.engine,
                **elem_step_check_kwargs(self.parameters),
                geodesic_kwargs={
                    "nimages": self.gi_inputs.nimages,
                    "nudge": self.gi_inputs.nudge,
                    "friction": self.gi_inputs.friction,
                    "align": self.gi_inputs.align,
                    "random_seed": self.gi_inputs.random_seed,
                },
            )
            self.geom_grad_calls_made += elem_step_results.number_grad_calls
        else:
            elem_step_results = IS_ELEM_STEP
        return elem_step_results

    def minimize_nodes(self, chain: Chain, node_tangents: list, dr, idx_grown: tuple):
        raw_chain = chain.copy()
        idx1, idx2 = idx_grown
        if idx2 is not None:
            return self._min_two_nodes(
                raw_chain, tangents=node_tangents, ind_node1=idx1, ind_node2=idx2, dr=dr
            )
        return self._min_node(raw_chain, tangent=node_tangents[0], ind_node=idx1)

    def _geodesic_tangent(self, nodes, ref_ind: int) -> np.ndarray:
        geoms = ch.calculate_geodesic_tangent(
            nodes, ref_ind,
            dr=_geodesic_drstep(self.parameters),
            nimages=self.gi_inputs.nimages,
            nudge=self.gi_inputs.nudge,
            friction=self.gi_inputs.friction,
            align=self.gi_inputs.align,
            random_seed=self.gi_inputs.random_seed,
        )
        return geoms[2].coords - geoms[0].coords

    def _valid_or_linear(self, tangent, chain: Chain, ind: int, which: str = "") -> np.ndarray:
        if _valid_tangent(tangent):
            return tangent
        self._log(f"Invalid geodesic tangent{which}; using linear tangent.", level="warning")
        return _linear_tangent(chain, ind)

    def _pair_geodesic_tangents(self, raw_chain: Chain, ind_node1: int, ind_node2: int):
        return (
            self._valid_or_linear(self._geodesic_tangent(raw_chain, ind_node1) / 2,
                                  raw_chain, ind_node1, " for left node"),
            self._valid_or_linear(self._geodesic_tangent(raw_chain, ind_node2) / 2,
                                  raw_chain, ind_node2, " for right node"),
        )

    def _min_two_nodes(
        self,
        raw_chain: 'Chain',
        tangents: np.array,
        dr: float,
        ind_node1: int,
        ind_node2: int,
    ):
        """Minimize two grown nodes simultaneously, each along the gradient
        perpendicular to its own tangent. Stops when both are converged, when
        they have moved apart by more than `phi * dr`, or after max_min_iter."""
        n_nodes = len(raw_chain.nodes)
        if ind_node1 == ind_node2 or not (0 <= ind_node1 < n_nodes and 0 <= ind_node2 < n_nodes):
            self._log("Invalid node indices for node-pair minimization.", level="error")
            return raw_chain

        if self.parameters.tangent == 'geodesic':
            tangent1, tangent2 = self._pair_geodesic_tangents(raw_chain, ind_node1, ind_node2)
        elif self.parameters.tangent == 'linear':
            def _two_point(i):
                t = raw_chain[min(i + 1, n_nodes - 1)].coords - raw_chain[max(i - 1, 0)].coords
                return t / np.linalg.norm(t)
            tangent1, tangent2 = _two_point(ind_node1), _two_point(ind_node2)
        else:
            raise ValueError(f"Invalid tangent type {self.parameters.tangent} specified. Select 'geodesic' or 'linear'.")

        phi = float(getattr(self.parameters, "phi", PHI))
        nsteps = 1  # Account for an implicit initial gradient call if this is part of a larger process
        init_d = self._distance_function(raw_chain[ind_node1], raw_chain[ind_node2])
        while True:
            try:
                node1_opt = raw_chain[ind_node1]
                node2_opt = raw_chain[ind_node2]
                if self._distance_function(node1_opt, node2_opt) >= init_d + phi * dr:
                    self._log(f"Nodes fell by {phi} times dr. Stopping minimization.", level="warning")
                    break
                if nsteps >= self.parameters.max_min_iter:
                    self._log(
                        f"Stopping minimization: reached maximum iterations ({self.parameters.max_min_iter}).",
                        level="warning",
                    )
                    break

                unit_tan1 = None if tangent1 is None else tangent1 / np.linalg.norm(tangent1)
                unit_tan2 = None if tangent2 is None else tangent2 / np.linalg.norm(tangent2)
                if not _valid_tangent(unit_tan1) or not _valid_tangent(unit_tan2):
                    self._log("Invalid tangent after fallback; stopping node-pair minimization.", level="warning")
                    return raw_chain

                direction1 = ch.get_nudged_pe_grad(unit_tangent=unit_tan1, gradient=node1_opt.gradient)
                direction2 = ch.get_nudged_pe_grad(unit_tangent=unit_tan2, gradient=node2_opt.gradient)
                # The step uses the nudged gradients as is; the rigid-body
                # projection only enters the convergence check.
                step_gradients = np.array([direction1, direction2])
                grad_inf_norm1 = np.amax(abs(project_rigid_body_forces(node1_opt.coords, direction1, masses=None)))
                grad_inf_norm2 = np.amax(abs(project_rigid_body_forces(node2_opt.coords, direction2, masses=None)))
                combined = max(grad_inf_norm1, grad_inf_norm2)
                self._log(
                    f"MIN: Node1 Grad: {grad_inf_norm1:.4f} | Node2 Grad: {grad_inf_norm2:.4f} | Combined Max Grad: {combined:.4f}",
                    verbose=1,
                )
                if combined <= self.parameters.grad_tol:
                    break

                out_chain = self.optimizer.optimize_step(
                    chain=Chain.model_validate({"nodes": [node1_opt, node2_opt]}),
                    chain_gradients=step_gradients,
                )
                new_node1, new_node2 = out_chain.nodes[0], out_chain.nodes[1]
                self.engine.compute_energies([new_node1, new_node2])
                self.grad_calls_made += 2
                raw_chain.nodes[ind_node1] = new_node1
                raw_chain.nodes[ind_node2] = new_node2
                self._append_chain_snapshot(raw_chain, f"FNEB node-pair minimize step {nsteps}")
                nsteps += 1
                # Tangents are always refreshed geodesically after a step,
                # whatever `tangent` selected for the first one.
                tangent1, tangent2 = self._pair_geodesic_tangents(raw_chain, ind_node1, ind_node2)
            except Exception:
                self._log(traceback.format_exc(), level="error", verbose=2)
                return raw_chain

        self._log(f"Minimization converged in {nsteps} steps.")
        return raw_chain

    def _min_node(self, raw_chain: Chain, tangent: np.array, ind_node: int, nsteps: int = 1):
        """Minimize the single node at `ind_node` along the gradient
        perpendicular to its tangent."""
        while True:
            try:
                if nsteps >= self.parameters.max_min_iter:
                    break
                node_to_opt = raw_chain[ind_node]
                if self.parameters.tangent == 'geodesic':
                    tangent = self._valid_or_linear(
                        self._geodesic_tangent(raw_chain[ind_node-1:ind_node+2], 1),
                        raw_chain, ind_node,
                    )
                elif self.parameters.tangent == 'linear':
                    tangent = _linear_tangent(raw_chain, ind_node)
                if not _valid_tangent(tangent):
                    self._log("Invalid tangent after fallback; stopping node minimization.", level="warning")
                    return raw_chain

                gperp1 = ch.get_nudged_pe_grad(
                    unit_tangent=tangent / np.linalg.norm(tangent), gradient=node_to_opt.gradient)
                direction = project_rigid_body_forces(node_to_opt.coords, gperp1, masses=None)
                out_chain = self.optimizer.optimize_step(
                    chain=Chain.model_validate({"nodes": [node_to_opt]}),
                    chain_gradients=np.array([direction]))
                new_node1 = out_chain.nodes[0]
                self.engine.compute_energies([new_node1])
                self.grad_calls_made += 1
                raw_chain.nodes[ind_node] = new_node1
                self._append_chain_snapshot(raw_chain, f"FNEB max-energy node minimize step {nsteps}")
                nsteps += 1

                grad_inf_norm = np.amax(abs(gperp1))
                self._log("MIN:", grad_inf_norm, verbose=2)
                if grad_inf_norm <= self.parameters.grad_tol:
                    break
            except Exception:
                self._log(traceback.format_exc(), level="error", verbose=2)
                return raw_chain
        self._log(f"Converged in {nsteps} steps")
        return raw_chain

    def minimize_node_maxene(self, chain: Chain, node_ind: int):
        chain_opt = self._min_node(chain.copy(), tangent=None, ind_node=node_ind)
        self.engine.g_old = None  # reset the conjugate gradient memory
        return chain_opt

    def grow_nodes(self, chain: Chain, dr: float, indices: tuple = None):
        sub_chain = [chain[indices[0]], chain[indices[1]]]
        metric = self.parameters.distance_metric.upper()
        tan1 = tan2 = None
        if metric == "GEODESIC":
            _, smoother = ch.run_geodesic(
                sub_chain, nimages=self.gi_inputs.nimages,
                nudge=self.gi_inputs.nudge,
                friction=self.gi_inputs.friction,
                align=self.gi_inputs.align,
                random_seed=self.gi_inputs.random_seed,
                return_smoother=True,
            )
            interpolated = ch.gi_path_to_nodes(
                xyz_coords=smoother.path,
                symbols=sub_chain[0].structure.symbols,
                charge=sub_chain[0].structure.charge,
                spinmult=sub_chain[0].structure.multiplicity,
            )
            for node in interpolated:
                node.has_molecular_graph = chain[0].has_molecular_graph
            add_two_nodes = not (smoother.length <= 2 * dr)
            if not add_two_nodes:
                self._log("Less than 2*dr, adding only one node")
                dr = smoother.length / 2

            def _select(nodes):
                return self._select_node_at_dist(
                    chain=nodes, dist=dr, direction=1,
                    dist_err=self.parameters.dist_err * dr, smoother=smoother,
                )

            node1, tan1 = _select(interpolated)
            if not node1:
                raise ValueError("Failed to select a new node at the requested distance.")
            node2 = None
            if add_two_nodes:
                node2, tan2 = _select(interpolated[::-1])
                if not node2:
                    add_two_nodes, tan2 = False, None
        elif metric == "LINEAR":
            a, b = sub_chain[0].coords, sub_chain[1].coords
            direction = (b - a) / np.linalg.norm((b - a))
            node1 = sub_chain[0].update_coords(a + direction * dr)
            node2 = sub_chain[1].update_coords(b - direction * dr)
            add_two_nodes = True
        else:
            raise ValueError(
                f"Invalid tangent type: {self.parameters.tangent}. Use one of LINEAR or GEODESIC"
            )

        self.engine.compute_energies([node2, node1] if add_two_nodes else [node1])
        self.grad_calls_made += 2 if add_two_nodes else 1
        grown_chain = chain.copy()
        insert_index = indices[1]
        if add_two_nodes:
            grown_chain.nodes.insert(insert_index, node2)
        grown_chain.nodes.insert(insert_index, node1)
        idx2 = insert_index + 1 if add_two_nodes else None
        return grown_chain, [tan1, tan2], (insert_index, idx2), dr

    def _max_energy_node(self, chain: Chain, node_a, node_b, nimg: int, nudge: float):
        """The highest-energy node of the geodesic interpolation between two
        nodes (`get_maxene_node`'s dict), and how many images it had."""
        _, smoother = ch.run_geodesic([node_a, node_b],
                                      nimages=nimg, nudge=nudge,
                                      align=self.gi_inputs.align,
                                      random_seed=self.gi_inputs.random_seed,
                                      return_smoother=True)
        gi = ch.gi_path_to_nodes(
            xyz_coords=smoother.path,
            symbols=chain[0].symbols,
            charge=chain[0].structure.charge,
            spinmult=chain[0].structure.multiplicity,
        )
        if self.parameters.use_xtb_grow:
            self._log("Using xtb to select max energy node")
            data = get_maxene_node(gi, engine=RunInputs().engine)
            data['node']._cached_energy = None
            data['node']._cached_gradient = None
            self.engine.compute_energies([data['node']])
            self.grad_calls_made += 1
        else:
            data = get_maxene_node(gi, engine=self.engine)
            self.grad_calls_made += data['grad_calls']
        return data, len(gi)

    def grow_nodes_maxene(self, chain: Chain, last_grown_ind: int = 0, nimg: int = 20, nudge=0.1):
        """
        will return a chain with 1 new node which is the highest energy interpolated
        node between the last node added and its nearest neighbors, and the new
        node's index.

        If no node is added, will return the input chain.
        """
        if last_grown_ind == 0:  # initial case
            data, _ = self._max_energy_node(chain, chain[0], chain[-1], nimg, nudge)
            ind_max, node = data['index'], data['node']
            barrier_climb_kcal = (node._cached_energy - chain.energies.max())*627.5
            if ind_max == 0 or ind_max == len(chain)-1:
                self._log("No TS found between endpoints. Returning input chain.", level="warning")
                return chain, 1
            if barrier_climb_kcal <= self.parameters.barrier_thre:
                self._log("Barrier climb is too low. Returning input chain.", level="warning")
                return chain, chain.energies.argmax()
            grown_chain = chain.copy()
            grown_chain.nodes = [chain[0], node, chain[-1]]
            return grown_chain, 1

        left, n_left = self._max_energy_node(
            chain, chain[last_grown_ind-1], chain[last_grown_ind], nimg, nudge)
        right, _ = self._max_energy_node(
            chain, chain[last_grown_ind], chain[last_grown_ind+1], nimg, nudge)
        e_max = chain.energies.max()
        if all((d['node'].energy - e_max)*627.5 < self.parameters.barrier_thre for d in (left, right)):
            left_converged = right_converged = True
        else:
            left_converged = left['index'] in (0, n_left - 1)
            right_converged = right['index'] == 0
        if not left_converged and not right_converged:
            self._log("Two potential directions found. Choosing highest ascent")
            if left['node'].energy > right['node'].energy:
                right_converged = True
            else:
                left_converged = True

        if left_converged and right_converged:
            self._log("TS guess found. Returning input chain.")
            return chain, last_grown_ind
        grown_chain = chain.copy()
        if left_converged:
            self._log("Growing rightwards...")
            grown_chain.nodes.insert(last_grown_ind+1, right['node'])
            return grown_chain, last_grown_ind+1
        self._log("Growing leftwards...")
        grown_chain.nodes.insert(last_grown_ind, left['node'])
        return grown_chain, last_grown_ind

    def _get_closest_node_ind(self, smoother_obj, reference):
        smallest_dist = 1e10
        ind = None
        for i, geom in enumerate(smoother_obj.path):
            dist, _ = RMSD(geom, reference)
            if dist < smallest_dist:
                smallest_dist = dist
                ind = i
        return ind

    def _select_node_at_dist(
        self,
        chain: Chain,
        dist: float,
        direction: int,
        smoother: MorseGeodesic = None,
        dist_err: float = 0.1,
    ):
        """
        will iterate through chain and select the node that is 'dist' away up to 'dist_err'

        dist - -> the requested distance where you want the nodes
        direction - -> whether to go forward(1) or backwards(-1). if forward, will pick the requested node
                    to the first node. if backwards, will pick the requested node to the last node

        """
        input_chain = [node.copy() for node in chain]
        if direction == -1:
            input_chain.reverse()

        start_node = input_chain[0]
        best_node = None
        best_dist_err = 10000.0
        best_node_tangent = None
        closest_node = None
        closest_dist_err = 10000.0
        closest_node_tangent = None
        for i, node in enumerate(input_chain[1:-1], start=1):
            if self.parameters.distance_metric.upper() == "GEODESIC":
                if direction == -1:
                    start = len(smoother.path) - i
                    end = -1

                elif direction == 1:
                    start = 1
                    end = i + 1

                curr_dist = smoother.segment_lengths[start-1:end-1].sum()
            else:
                curr_dist = self._distance_function(
                    node1=start_node, node2=node)
            curr_dist_err = np.abs(curr_dist - dist)
            if self.parameters.verbosity > 1:
                self._log(f"{curr_dist_err=} vs {dist_err=} || {direction=}", verbose=2)

            prev_node = input_chain[i - 1]
            next_node = input_chain[i + 1]
            if self.parameters.tangent == "geodesic":
                tau_plus = next_node.coords - node.coords
                tau_minus = node.coords - prev_node.coords
                cand_tangent = (tau_plus + tau_minus) / 2
            else:
                cand_tangent = None

            if curr_dist_err < closest_dist_err:
                closest_node = node
                closest_dist_err = curr_dist_err
                closest_node_tangent = cand_tangent

            if curr_dist_err <= dist_err and curr_dist_err < best_dist_err:
                best_node = node
                best_dist_err = curr_dist_err
                best_node_tangent = cand_tangent

                # break

        if best_node is None:
            return closest_node, closest_node_tangent

        return best_node, best_node_tangent

    def chain_converged(self, chain: Chain, dr: float, indices):
        node1_ind, node2_ind = indices
        self._log("Indices:", indices, verbose=2)

        node1 = chain[node1_ind]

        next_grow_ind = node1_ind

        if indices[1] is None:

            node2_1 = chain[node1_ind + 1]
            node2_2 = chain[node1_ind - 1]

            dist_1 = self._distance_function(node1, node2_1)
            enedist_1 = abs(node1.energy - node2_1.energy)
            dist_2 = self._distance_function(node1, node2_2)
            enedist_2 = abs(node1.energy - node2_2.energy)
            self._log(f"Distances: {dist_1} {dist_2}", verbose=2)
            # if dist_1 > dist_2:
            if enedist_1 > enedist_2:
                dist = dist_1
            else:
                next_grow_ind = node1_ind - 1
                dist = dist_2
        else:
            node2 = chain[node2_ind]
            dist = self._distance_function(node1, node2)
        self._log(f"Distance between innermost nodes {dist}")
        return dist <= dr, next_grow_ind
