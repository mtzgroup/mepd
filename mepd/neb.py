from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import copy

import numpy as np
from mepd.convergence_helpers import chain_converged
from numpy.typing import NDArray
from openbabel import pybel
from mepd.errors import ExternalProgramError

import mepd.chainhelpers as ch
from mepd.chain import Chain
from mepd.elementarystep import IS_ELEM_STEP, elem_step_check_kwargs, ElemStepResults, check_if_elem_step
from mepd.engines import Engine
from mepd.errors import ElectronicStructureError, NoneConvergedException
from mepd.inputs import ChainInputs, GIInputs, NEBInputs
from mepd.nodes.node import Node
from mepd.optimizers.optimizer import Optimizer
from mepd.pathminimizers.pathminimizer import PathMinimizer
from mepd.optimizers.vpo import VelocityProjectedOptimizer
from mepd.progress import print_chain_step, format_neb_caption, update_status

# Rich imports for flashy CLI output
try:
    from rich.console import Console
    from rich.panel import Panel
    _console = Console()
    _rich_available = True
except ImportError:
    _console = None
    _rich_available = False


pybel.ob.obErrorLog.SetOutputLevel(0)


MAX_NRETRIES = 3
NMINIMA_STEPS = 5

NSTEPS_STRAIN = 10000000  # turning this off
NADD = 1
HARTREE_TO_KCAL_MOL = 627.509474
ENERGY_INVERSION_WARN_KCAL = 1.0

CLIMBING_IMAGE_MAX_STEPS = 100
PLATEAU_EXIT_MESSAGE = (
    "Chain convergence metrics have plateaued and adaptive resolution cannot add more images. "
    "The chain will likely not converge with more minimization steps; lower spring constants "
    "or higher bead density would be needed."
)


def _endpoint_energy_inversion_warning_text(
    energies: np.ndarray,
) -> str | None:
    """Describe suspicious endpoint-vs-interior energy ordering when present."""
    if energies is None:
        return None
    enes = np.array(energies, dtype=float)
    if enes.size < 3:
        return None

    ts_ind = int(np.argmax(enes))
    if ts_ind not in (0, len(enes) - 1):
        return None

    endpoint_min = float(min(enes[0], enes[-1]))
    interior_min = float(np.min(enes[1:-1]))
    drop_kcal = (endpoint_min - interior_min) * HARTREE_TO_KCAL_MOL
    if drop_kcal < ENERGY_INVERSION_WARN_KCAL:
        return None

    msg = (
        "Endpoint energies are higher than all interior images while the "
        f"lowest interior image is {drop_kcal:.2f} kcal/mol below the lower-energy endpoint. "
        "This can produce an endpoint TS guess and unstable NEB behavior."
    )
    return msg


@dataclass
class NEB(PathMinimizer):
    """
    Class for running, storing, and visualizing nudged elastic band minimizations.
    Main functions to use are:
    - self.optimize_chain()
    - self.plot_opt_history()

    !!! Note
        Colton said this rocks

    """

    initial_chain: Chain
    optimizer: Optimizer
    parameters: NEBInputs
    engine: Engine

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)

    def __post_init__(self):
        self.n_steps_still_chain = 0
        self.grad_calls_made = 0
        self.geom_grad_calls_made = 0
        self.orig_timestep = self.optimizer.timestep
        self._last_resolution_insert_step = -10**9
        self._adaptive_metric_history: list[dict[str, float]] = []
        self._plateau_exit_metric_history: list[dict[str, float]] = []
        # NEBInputs is the authoritative source for these when the optimizer
        # supports them (currently only ConjugateGradient), overriding
        # whatever the optimizer itself was constructed with.
        for attr in ("negative_steps_thre", "positive_steps_thre"):
            if hasattr(self.optimizer, attr) and hasattr(self.parameters, attr):
                setattr(self.optimizer, attr, getattr(self.parameters, attr))

    def _reset_optimizer_history(self) -> None:
        reset_fn = getattr(self.optimizer, "reset", None)
        if callable(reset_fn):
            reset_fn()
            return

        for attr in ("g_old", "p_old", "x_old"):
            if hasattr(self.optimizer, attr):
                setattr(self.optimizer, attr, None)
        if hasattr(self.optimizer, "s_history"):
            self.optimizer.s_history.clear()
        if hasattr(self.optimizer, "y_history"):
            self.optimizer.y_history.clear()

    def _optimizer_detail_lines(self) -> list[str]:
        timestep = getattr(self.optimizer, "timestep", None)
        if timestep is None:
            return []
        return [f"dt={float(timestep):.6g}"]

    def _build_inserted_node(self, left_node: Node, right_node: Node, chain: Chain) -> Node:
        import mepd.chainhelpers as ch

        use_gi = bool(getattr(chain.parameters, "use_geodesic_interpolation", False))
        if use_gi and hasattr(left_node, "symbols") and hasattr(right_node, "symbols"):
            try:
                gi = ch.run_geodesic([left_node, right_node], nimages=3, align=False)
                inserted = gi[1]
            except Exception:
                inserted = left_node.update_coords(0.5 * (left_node.coords + right_node.coords))
        else:
            inserted = left_node.update_coords(0.5 * (left_node.coords + right_node.coords))

        inserted.converged = False
        inserted.do_climb = False
        if hasattr(inserted, "_cached_result"):
            inserted._cached_result = None
        if hasattr(inserted, "_cached_energy"):
            inserted._cached_energy = None
        if hasattr(inserted, "_cached_gradient"):
            inserted._cached_gradient = None
        return inserted

    def _collect_adaptive_convergence_metrics(self, chain: Chain) -> dict[str, float]:
        import mepd.chainhelpers as ch

        rms_gperps = np.array(chain.rms_gradients, dtype=float)
        max_rms_gperp = float(np.max(rms_gperps)) if rms_gperps.size else 0.0
        mean_rms_gperp = float(np.mean(rms_gperps)) if rms_gperps.size else 0.0

        ts_grad = max_rms_gperp
        try:
            ts_ind = int(np.argmax(chain.energies))
            ts_grad = float(np.amax(np.abs(ch.get_g_perps(chain)[ts_ind])))
        except Exception:
            pass

        max_spring = 0.0
        try:
            springgrads = [
                float(np.amax(np.abs(springgrad)))
                for springgrad in chain.springgradients
            ]
            max_spring = max(springgrads) if springgrads else 0.0
        except Exception:
            pass

        ts_triplet_gspring = max_spring
        try:
            ts_triplet_gspring = float(chain.ts_triplet_gspring_infnorm)
        except Exception:
            pass

        return {
            "max_rms_gperp": max_rms_gperp,
            "mean_rms_gperp": mean_rms_gperp,
            "ts_grad": ts_grad,
            "ts_triplet_gspring": ts_triplet_gspring,
            "max_spring": max_spring,
        }

    def _metrics_plateauing(
        self,
        history: list[dict[str, float]],
        metrics: dict[str, float],
        *,
        window: int,
        rtol: float,
    ) -> bool:
        window = max(0, int(window))
        rtol = max(0.0, float(rtol))

        history.append(metrics)
        needed = window + 1
        if len(history) > needed:
            del history[:-needed]
        if len(history) < needed:
            return False

        start = history[0]
        end = history[-1]
        keys = (
            "max_rms_gperp",
            "mean_rms_gperp",
            "ts_grad",
            "ts_triplet_gspring",
            "max_spring",
        )
        for key in keys:
            start_v = float(start[key])
            end_v = float(end[key])
            scale = max(abs(start_v), 1e-12)
            rel_improvement = (start_v - end_v) / scale
            if rel_improvement > rtol:
                return False
        return True

    def _adaptive_metrics_plateauing(self, metrics: dict[str, float]) -> bool:
        return self._metrics_plateauing(
            self._adaptive_metric_history,
            metrics,
            window=getattr(self.parameters, "adaptive_plateau_window", 3),
            rtol=getattr(self.parameters, "adaptive_plateau_rtol", 0.05),
        )

    def _adaptive_resolution_exhausted(self, chain: Chain) -> bool:
        if not bool(getattr(self.parameters, "adaptive_resolution", False)):
            return True
        max_images = int(getattr(self.parameters, "adaptive_max_images", 25))
        return len(chain) >= max_images

    def _plateau_exit_triggered(self, chain: Chain) -> bool:
        if not self._adaptive_resolution_exhausted(chain):
            return False

        metrics = self._collect_adaptive_convergence_metrics(chain)
        return self._metrics_plateauing(
            self._plateau_exit_metric_history,
            metrics,
            window=getattr(self.parameters, "adaptive_plateau_window", 3),
            rtol=getattr(self.parameters, "adaptive_plateau_rtol", 0.05),
        )

    def _segment_around_node(self, chain: Chain, node_index: int) -> int | None:
        if len(chain) < 2:
            return None

        node_index = max(0, min(int(node_index), len(chain) - 1))
        if node_index <= 0:
            return 0
        if node_index >= len(chain) - 1:
            return len(chain) - 2

        candidate_segments = (node_index - 1, node_index)
        energies = np.asarray(chain.energies, dtype=float)
        if energies.size >= len(chain):
            energy_deltas = np.abs(np.diff(energies))
            return max(candidate_segments, key=lambda i: energy_deltas[i])

        segment_lengths = np.diff(chain.path_length)
        if segment_lengths.size >= len(chain) - 1:
            return max(candidate_segments, key=lambda i: segment_lengths[i])
        return node_index - 1

    def _max_rms_gperp_segment(self, chain: Chain) -> int | None:
        rms_gperps = np.asarray(chain.rms_gradients, dtype=float)
        if rms_gperps.size == 0:
            return None
        return self._segment_around_node(chain, int(np.argmax(rms_gperps)))

    def _maybe_adapt_chain_resolution(self, chain: Chain, step: int) -> tuple[Chain, bool]:
        if not bool(getattr(self.parameters, "adaptive_resolution", False)):
            return chain, False

        max_images = int(getattr(self.parameters, "adaptive_max_images", 25))
        if len(chain) >= max_images:
            return chain, False

        cooldown = int(getattr(self.parameters, "adaptive_cooldown_steps", 2))
        if step - self._last_resolution_insert_step <= cooldown:
            return chain, False

        metrics = self._collect_adaptive_convergence_metrics(chain)
        if not self._adaptive_metrics_plateauing(metrics):
            return chain, False

        max_rms_gperp_is_limiting = (
            metrics["max_rms_gperp"] > float(self.parameters.max_rms_grad_thre)
            and metrics["mean_rms_gperp"] <= float(self.parameters.rms_grad_thre)
            and metrics["ts_grad"] <= float(self.parameters.ts_grad_thre)
            and metrics["ts_triplet_gspring"] <= float(self.parameters.ts_spring_thre)
            and metrics["max_spring"] <= float(self.parameters.ts_spring_thre)
        )
        if not max_rms_gperp_is_limiting:
            return chain, False

        seg_index = self._max_rms_gperp_segment(chain)
        if seg_index is None:
            return chain, False

        inserted_node = self._build_inserted_node(
            left_node=chain[seg_index],
            right_node=chain[seg_index + 1],
            chain=chain,
        )
        refined_chain = chain.copy()
        refined_chain.nodes.insert(seg_index + 1, inserted_node)
        refined_chain._zero_velocity()

        grads = self.engine.compute_gradients(refined_chain)
        enes = self.engine.compute_energies(refined_chain)
        self._update_cache(refined_chain, grads, enes)

        self._reset_optimizer_history()
        self._last_resolution_insert_step = step
        self._adaptive_metric_history = []
        self._plateau_exit_metric_history = []

        msg = (
            f"Adaptive resolution inserted an image between nodes {seg_index} and "
            f"{seg_index + 1} (MAX RMS_GPERP is the limiting convergence criterion)."
        )
        self._say(msg)
        return refined_chain, True

    def _get_climbing_pair_indices(self, chain: Chain) -> tuple[int, int]:
        energies = np.asarray(chain.energies, dtype=float)
        candidate_indices = np.arange(1, len(chain) - 1, dtype=int)
        if candidate_indices.size < 2:
            candidate_indices = np.arange(len(chain), dtype=int)
        if candidate_indices.size < 2:
            raise ElectronicStructureError(
                msg="Need at least two images to define a climbing-image insertion pair.",
                obj=None,
            )

        ordered = candidate_indices[np.argsort(energies[candidate_indices])[::-1]]
        i_high = int(ordered[0])
        i_second = int(ordered[1])
        if abs(i_high - i_second) == 1:
            return min(i_high, i_second), max(i_high, i_second)

        neighs = []
        if i_high - 1 >= 0:
            neighs.append(i_high - 1)
        if i_high + 1 < len(chain):
            neighs.append(i_high + 1)
        if not neighs:
            return min(i_high, i_second), max(i_high, i_second)

        i_local = int(max(neighs, key=lambda idx: energies[idx]))
        return min(i_high, i_local), max(i_high, i_local)

    def _climbing_node_perp_grad(self, chain: Chain, climb_index: int) -> float:
        gperps = ch.get_g_perps(chain)
        if climb_index < 0 or climb_index >= len(gperps):
            return float("inf")
        return float(np.amax(np.abs(gperps[climb_index])))

    def _run_post_convergence_climbing_refinement(self, chain: Chain) -> Chain:
        if len(chain) < 2:
            return chain

        left_idx, right_idx = self._get_climbing_pair_indices(chain)
        inserted_node = self._build_inserted_node(
            left_node=chain[left_idx],
            right_node=chain[right_idx],
            chain=chain,
        )
        refined_chain = chain.copy()
        climb_index = left_idx + 1
        refined_chain.nodes.insert(climb_index, inserted_node)
        refined_chain._zero_velocity()

        grads = self.engine.compute_gradients(refined_chain)
        enes = self.engine.compute_energies(refined_chain)
        self._update_cache(refined_chain, grads, enes)

        for idx, node in enumerate(refined_chain.nodes):
            node.do_climb = idx == climb_index
            node.converged = idx != climb_index

        self._reset_optimizer_history()

        msg = (
            f"Running post-convergence climbing-image refinement on node {climb_index} "
            f"(inserted between nodes {left_idx} and {right_idx})."
        )
        self._say(msg)

        grad_threshold = float(getattr(self.parameters, "ts_grad_thre", self.parameters.rms_grad_thre))
        for ci_step in range(1, CLIMBING_IMAGE_MAX_STEPS + 1):
            ci_grad = self._climbing_node_perp_grad(refined_chain, climb_index=climb_index)
            if ci_grad <= grad_threshold:
                done_msg = (
                    f"Climbing-image refinement converged in {ci_step - 1} steps "
                    f"(max |g_perp|={ci_grad:.4e} <= {grad_threshold:.4e})."
                )
                self._say(done_msg)
                return refined_chain

            refined_chain = self.update_chain(chain=refined_chain)
            self.chain_trajectory.append(refined_chain)
            self.gradient_trajectory.append(refined_chain.gradients)

        final_grad = self._climbing_node_perp_grad(refined_chain, climb_index=climb_index)
        end_msg = (
            f"Climbing-image refinement reached {CLIMBING_IMAGE_MAX_STEPS} steps "
            f"(max |g_perp|={final_grad:.4e}, threshold={grad_threshold:.4e})."
        )
        self._say(end_msg)
        return refined_chain

    def _do_early_stop_check(self, chain: Chain) -> Tuple[bool, ElemStepResults]:
        """
        this function calls geometry minimizations to verify if
        chain is an elementary step

        Args:
            chain (Chain): chain to check

        Returns:
            tuple(boolean, ElemStepResults) : boolean of whether
                    to stop early, and an ElemStepResults objects
        """

        elem_step_results = check_if_elem_step(
            inp_chain=chain,
            engine=self.engine,
            verbose=self.parameters.v,
            **elem_step_check_kwargs(self.parameters),
        )

        if not elem_step_results.is_elem_step:
            msg = "Stopped early because chain is not an elementary step."
            msg2 = f"Split chain based on: {elem_step_results.splitting_criterion}"
            if self.parameters.v:
                print(f"\n{msg}")
                print(msg2)
            else:
                update_status(msg)
                update_status(msg2)
            self.optimized = chain
            return True, elem_step_results

        else:
            self.n_steps_still_chain = 0
            return False, elem_step_results

    def _check_early_stop(self, chain: Chain, force_check: bool = False) -> Tuple[bool, ElemStepResults]:
        """
        this function computes chain distances and checks gradient
        values in order to decide whether the expensive minimization of
        the chain should be done.
        """
        import mepd.chainhelpers as ch

        ind_ts_guess = np.argmax(chain.energies)
        ts_guess_grad = np.amax(np.abs(ch.get_g_perps(chain)[ind_ts_guess]))
        ts_triplet_spring = chain.ts_triplet_gspring_infnorm

        early_stop_ready = (
            ts_guess_grad < self.parameters.early_stop_force_thre
            and ts_triplet_spring < self.parameters.early_stop_force_thre
        )

        if early_stop_ready or force_check:

            new_params = copy.deepcopy(self.parameters)
            new_params.early_stop_force_thre = 0.0
            self.parameters = new_params

            try:
                stop_early, elem_step_results = self._do_early_stop_check(chain)
            except ElectronicStructureError as exc:
                msg = (
                    "Early elementary-step check was inconclusive because an "
                    "auxiliary electronic-structure calculation failed; continuing "
                    f"path minimization ({type(exc).__name__}: {exc})."
                )
                if self.parameters.v:
                    print(f"\n{msg}")
                else:
                    update_status(msg)
                stop_early = False
                elem_step_results = ElemStepResults(
                    is_elem_step=None,
                    is_concave=None,
                    splitting_criterion=None,
                    minimization_results=[],
                    number_grad_calls=0,
                )

            self.parameters.early_stop_force_thre = (
                0.0  # setting it to 0 so we don't check it over and over
            )
            if (
                not stop_early
                and early_stop_ready
                and elem_step_results.is_elem_step
                and getattr(self.parameters, "ts_converged_stop", False)
            ):
                # The TS region has converged and the chain is one elementary
                # step: the rest of the band won't change either answer.
                self._say("TS region converged and the chain is an elementary step; stopping.")
                self.optimized = chain
                return True, elem_step_results
            return stop_early, elem_step_results

        else:
            return False, ElemStepResults(
                is_elem_step=None,
                is_concave=None,
                splitting_criterion=None,
                minimization_results=[],
                number_grad_calls=0,
            )

    # @Jan: This should be a more general function so that the
    # lower level of theory can be whatever the user wants.
    def _say(self, msg: str) -> None:
        if self.parameters.v:
            print(msg)
        else:
            update_status(msg)

    def _ts_region_converged(self, prev: Chain, new: Chain, ind_ts: int, ts_grad: float) -> bool:
        """The TS guess is converged even if the rest of the band isn't: an
        interior highest image whose perpendicular gradient and neighbouring
        springs are within threshold, with a barrier that has stopped moving."""
        if not getattr(self.parameters, "ts_converged_stop", False) or ind_ts in (0, len(new) - 1):
            return False
        return (
            ts_grad <= self.parameters.ts_grad_thre
            and new.ts_triplet_gspring_infnorm <= self.parameters.ts_spring_thre
            and abs(new.get_eA_chain() - prev.get_eA_chain()) <= self.parameters.barrier_thre
        )

    def _elem_step_check(self, chain: Chain) -> ElemStepResults:
        results = check_if_elem_step(
            inp_chain=chain, engine=self.engine, verbose=self.parameters.v,
            **elem_step_check_kwargs(self.parameters),
        )
        self.geom_grad_calls_made += results.number_grad_calls
        return results

    def optimize_chain(self) -> ElemStepResults:
        """
        Main function. After an NEB object has been created, running this function will
        minimize the chain and return the elementary step results from the final minimized chain.

        Running this function will populate the `.chain_trajectory` object variable, which
        contains the history of the chains minimized. Once it is completed, you can use
        `.plot_opt_history()` to view the optimization over time.

        Args:
            self: initialized NEB object
        Raises:
            NoneConvergedException: If chain did not converge in alloted steps.
        """
        import mepd.chainhelpers as ch

        nsteps = 1
        nsteps_minima_present = 0
        already_forced_check = False

        nsteps_strained_node = 0
        prev_most_strained_node = 0
        warned_endpoint_energy_inversion = False

        chain_previous = self.initial_chain.copy()
        self.chain_trajectory.append(chain_previous)
        chain_previous._zero_velocity()
        self.optimizer.g_old = None

        while nsteps < self.parameters.max_steps + 1:

            if nsteps > 1:
                if not self.parameters.v:
                    status = "Optimizing path... Step {}".format(nsteps)
                    timestep = getattr(self.optimizer, "timestep", None)
                    if timestep is not None:
                        status += f" | dt={float(timestep):.6g}"
                    update_status(status)

                chain_previous, _ = self._maybe_adapt_chain_resolution(
                    chain_previous, step=nsteps
                )

                # check if minima are present
                minima_present = len(ch._get_ind_minima(chain_previous)) >= 1
                if minima_present:
                    nsteps_minima_present += 1
                else:
                    nsteps_minima_present = 0
                force_check = nsteps_minima_present >= NMINIMA_STEPS
                if force_check and not already_forced_check:
                    msg = f"A local minimum has been present for {nsteps_minima_present} steps, forcing early stop check."
                    self._say(msg)
                    nsteps_minima_present = 0
                    already_forced_check = True
                elif already_forced_check:
                    force_check = False

                # check if any node is too strained
                most_strained_node = np.argmax(chain_previous.rms_gradients)
                if most_strained_node == prev_most_strained_node:
                    nsteps_strained_node += 1
                else:
                    prev_most_strained_node = most_strained_node
                    nsteps_strained_node = 0

                if nsteps_strained_node >= NSTEPS_STRAIN:
                    msg = f"Node {most_strained_node} has been the most strained for {nsteps_strained_node} steps, upsampling chain"
                    self._say(msg)
                    chain_previous = ch.upsample_chain(
                        chain_previous, engine=self.engine, nimages=NADD)

                    self.grad_calls_made += NADD  # two new nodes need gradients
                    nsteps_strained_node = 0
                    self.optimizer.g_old = None  # reset optimizer history, since vector changes shape

                if self.parameters.do_elem_step_checks:
                    stop_early, elem_step_results = self._check_early_stop(
                        chain_previous, force_check=force_check)
                    self.geom_grad_calls_made += elem_step_results.number_grad_calls
                    if stop_early:
                        return elem_step_results
            try:
                new_chain = self.update_chain(chain=chain_previous)

            except ExternalProgramError:
                msg = "Electronic structure error during NEB step, rolling back 5 steps and trying again."
                if self.parameters.v:
                    print(f"\n!!!{msg}")
                else:
                    update_status(msg)
                rollback_steps = min(5, len(self.chain_trajectory)-1)
                chain_previous = self.chain_trajectory[-(rollback_steps+1)]
                self.optimizer.reset()
                new_chain = self.update_chain(chain=chain_previous)


            max_rms_grad_val = np.amax(new_chain.rms_gradients)
            ind_ts_guess = np.argmax(new_chain.energies)
            ts_guess_grad = np.amax(
                np.abs(ch.get_g_perps(new_chain)[ind_ts_guess]))
            if not warned_endpoint_energy_inversion:
                warning_text = _endpoint_energy_inversion_warning_text(
                    energies=np.array(new_chain.energies, dtype=float),
                )
                if warning_text:
                    if self.parameters.v and _rich_available:
                        _console.print(Panel.fit(
                            f"[yellow]⚠ {warning_text}[/yellow]",
                            border_style="yellow",
                            title="[bold yellow]Energy Profile Warning[/bold yellow]",
                        ))
                    elif self.parameters.v:
                        print(f"Warning: {warning_text}")
                    else:
                        update_status(f"Warning: {warning_text}")
                    warned_endpoint_energy_inversion = True
            converged = chain_converged(
                chain_prev=chain_previous,
                chain_new=new_chain,
                parameters=self.parameters,
                verbose=self.parameters.v,
                detail_prefix_lines=self._optimizer_detail_lines(),
            )
            if not converged and self._ts_region_converged(
                chain_previous, new_chain, ind_ts_guess, ts_guess_grad
            ):
                self._say(f"Step {nsteps}: TS region converged; stopping before the rest of the band.")
                converged = True

            n_nodes_frozen = 0
            for node in new_chain:
                if node.converged:
                    n_nodes_frozen += 1

            grad_calls_made = len(new_chain) - n_nodes_frozen
            self.grad_calls_made += grad_calls_made

            grad_corr = ch._gradient_correlation(new_chain, chain_previous)
            timestep_update = getattr(self.optimizer, "update_timestep_from_correlation", None)
            if callable(timestep_update):
                timestep_message = timestep_update(grad_corr)
                if timestep_message:
                    if self.parameters.v:
                        print(f"\n{timestep_message}")
                    else:
                        update_status(timestep_message)

            caption = format_neb_caption(
                step=nsteps,
                ts_grad=np.amax(np.abs(ts_guess_grad)),
                max_rms_grad=max_rms_grad_val,
                ts_triplet_gspring=new_chain.ts_triplet_gspring_infnorm,
                nodes_frozen=n_nodes_frozen,
                timestep=self.optimizer.timestep,
                grad_corr=grad_corr,
            )
            print_chain_step(new_chain, caption, force_update=True)

            self.chain_trajectory.append(new_chain)
            self.gradient_trajectory.append(new_chain.gradients)

            if converged:
                if self.parameters.v:
                    print("\nChain converged!")
                else:
                    update_status("Chain converged.")

                final_chain = new_chain
                if self.parameters.do_elem_step_checks:
                    elem_step_results = self._elem_step_check(final_chain)
                else:
                    elem_step_results = IS_ELEM_STEP

                if self.parameters.climb and elem_step_results.is_elem_step:
                    final_chain = self._run_post_convergence_climbing_refinement(new_chain)
                self.optimized = final_chain
                return elem_step_results

            if self._plateau_exit_triggered(new_chain):
                msg = (
                    f"{PLATEAU_EXIT_MESSAGE} "
                    f"(plateau window: {int(getattr(self.parameters, 'adaptive_plateau_window', 3))} steps)"
                )
                if self.parameters.v:
                    print(f"\n{msg}")
                else:
                    update_status(msg)
                self.optimized = new_chain
                raise NoneConvergedException(
                    trajectory=self.chain_trajectory,
                    msg=msg,
                    obj=self,
                )

            chain_previous = new_chain
            nsteps += 1

        new_chain = self.update_chain(chain=chain_previous)
        if not chain_converged(
            chain_prev=chain_previous,
            chain_new=new_chain,
            parameters=self.parameters,
            verbose=self.parameters.v,
            detail_prefix_lines=self._optimizer_detail_lines(),
        ):
            if self.parameters.climb and self.parameters.do_elem_step_checks:
                elem_step_results = self._elem_step_check(new_chain)
                if elem_step_results.is_elem_step:
                    self.optimized = self._run_post_convergence_climbing_refinement(new_chain)
                    return elem_step_results
            raise NoneConvergedException(
                trajectory=self.chain_trajectory,
                msg=f"\nChain did not converge at step {nsteps}",
                obj=self,
            )

    def _update_cache(
        self, chain: Chain, gradients: NDArray, energies: NDArray
    ) -> None:
        """
        will update the `_cached_energy` and `_cached_gradient` attributes in the chain
        nodes based on the input `gradients` and `energies`
        """
        from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults

        for node, grad, ene in zip(chain, gradients, energies):
            if not hasattr(node, "_cached_result"):
                res = FakeQCIOResults(energy=ene, gradient=grad)
                outp = FakeQCIOOutput(results=res)
                node._cached_result = outp
                node._cached_energy = ene
                node._cached_gradient = grad

    def update_chain(self, chain: Chain) -> Chain:
        import mepd.chainhelpers as ch

        if len(chain) == 0:
            raise ElectronicStructureError(
                msg="Cannot update an empty chain.",
                obj=None,
            )

        grads = self.engine.compute_gradients(chain)
        enes = self.engine.compute_energies(chain)
        if len(grads) == 0 or len(enes) == 0:
            raise ElectronicStructureError(
                msg="Engine returned empty gradients or energies during NEB update.",
                obj=None,
            )
        self._update_cache(chain, grads, enes)

        try:
            grad_step = ch.compute_NEB_gradient(
                chain,
                geodesic_tangent=self.parameters.use_geodesic_tangent,
            )
            converged_mask = np.array([node.converged for node in chain.nodes], dtype=bool)
            if converged_mask.any():
                grad_step = np.array(grad_step, dtype=float, copy=True)
                grad_step[converged_mask] = 0.0
        except IndexError as exc:
            raise ElectronicStructureError(
                msg="Failed to compute NEB gradient (empty or malformed chain gradients).",
                obj=None,
            ) from exc

        if chain.parameters.frozen_atom_indices:
            inds = chain.parameters.frozen_atom_indices
            for index in inds:
                for image_ind in range(grad_step.shape[0]):
                    grad_step[image_ind][index] = np.array([0.0, 0.0, 0.0])

        alpha = 1.0
        ntries = 0
        grads_success = False
        while not grads_success and ntries < MAX_NRETRIES:
            try:
                new_chain = self.optimizer.optimize_step(
                    chain=chain, chain_gradients=grad_step*alpha)

                # keep frozen nodes fixed (coords + cache) during all optimizer steps
                for node_index, (new_node, old_node) in enumerate(zip(new_chain.nodes, chain.nodes)):
                    if old_node.converged:
                        new_chain.nodes[node_index] = old_node.copy()
                        new_chain.nodes[node_index].converged = True
                        new_chain.nodes[node_index].do_climb = old_node.do_climb

                self.engine.compute_gradients(new_chain)
                grads_success = True

            except ExternalProgramError:
                print("SHRINKING")
                self.optimizer.g_old = None
                alpha *= .8
                ntries += 1
        if not grads_success and ntries >= MAX_NRETRIES:
            print("!!!Electronic structure error! Smoothing current chain with GI")
            new_chain = ch.run_geodesic(
                chain, chain_inputs=chain.parameters, nimages=len(chain))
            self.engine.compute_gradients(new_chain)

        return new_chain

    def read_from_disk(
        fp: Path,
        history_folder: Path = None,
        chain_parameters=ChainInputs(),
        neb_parameters=NEBInputs(),
        gi_parameters=GIInputs(),
        optimizer=VelocityProjectedOptimizer(),
        engine: Engine = None,
        charge: int = 0,
        multiplicity: int = 1,
    ):
        if isinstance(fp, str):
            fp = Path(fp)

        if history_folder is None:
            history_folder = fp.parent / (str(fp.stem) + "_history")

        if not history_folder.exists():
            raise ValueError("No history exists for this. Cannot load object.")
        else:
            history_files = list(history_folder.glob("*.xyz"))
            history = [
                Chain.from_xyz(
                    history_folder / f"traj_{i}.xyz", parameters=chain_parameters,
                    charge=charge,
                    spinmult=multiplicity
                )
                for i, _ in enumerate(history_files)
            ]

        n = NEB(
            initial_chain=history[0],
            parameters=neb_parameters,
            optimized=history[-1],
            chain_trajectory=history,
            optimizer=optimizer,
            engine=engine,
        )
        return n
