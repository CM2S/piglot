"""Module for managing data-driven optimisation campaigns."""
from typing import Any, Callable, Literal, Optional, TypeVar
import warnings
from threading import Lock
from abc import ABC, abstractmethod
from concurrent import futures
import numpy as np
import torch
from piglot.settings import Settings
from piglot.objective import ObjectiveResult, Objective
from piglot.data.dataset import ObjectiveDataset
from piglot.data.surrogate import ObjectiveModel
from piglot.optimisers.generic.containers import (
    OptimisationSettings,
    MultiObjectiveStateData,
    OptimisationState,
)
from piglot.optimisers.generic.acquisitions import get_best_posterior_mean


T = TypeVar('T', bound='CandidatePolicy')


class OptimisationCampaign:
    """Container for a data-driven optimisation campaign."""

    def __init__(
        self,
        settings: Settings,
        optim_settings: OptimisationSettings,
        objective: Objective,
    ) -> None:
        self.settings = settings
        self.optim_settings = optim_settings
        self.objective = objective
        self.dataset = ObjectiveDataset(settings, objective)
        self.pending: list[list[float]] = []
        self.state = OptimisationState(num_evaluations=0)
        self.lock = Lock()
        self.__model: ObjectiveModel = None

    def is_stochastic(self) -> bool:
        """Check if the optimisation campaign is stochastic.

        Returns
        -------
        bool
            True if the campaign is stochastic, False otherwise.
        """
        return self.objective.has_variance() or self.optim_settings.noisy

    def evaluate(self, params: np.ndarray) -> ObjectiveResult:
        """Evaluate the objective for the given parameters and store the observation.

        Parameters
        ----------
        params : np.ndarray
            The parameters at which to evaluate the objective.

        Returns
        -------
        ObjectiveResult
            The result of the objective evaluation.
        """
        # Register the pending evaluation
        with self.lock:
            concurrent = len(self.pending) > 0
            self.pending.append(params.tolist())

        # Evaluate the objective
        result = self.dataset.evaluate(params, concurrent=concurrent)

        # Update the state after evaluation
        with self.lock:
            self.pending.remove(params.tolist())
            self.update_state()
        return result

    def get_model(self) -> ObjectiveModel:
        """Update and return the surrogate model based on the current dataset.

        Returns
        -------
        ObjectiveModel
            The updated surrogate model.
        """
        if self.__model is None:
            self.__model = ObjectiveModel(
                self.dataset, self.optim_settings.surrogate_settings
            )
        return self.__model

    def get_extra_info(self) -> dict[str, str]:
        """Get extra information from the optimisation campaign.

        Returns
        -------
        dict[str, str]
            Extra information from the optimisation campaign.
        """
        extra_info: dict[str, str] = {}
        if self.__model is not None:
            # Number of PCA components with composition
            if self.objective.is_composite():
                extra_info['PCs'] = str(self.__model.gp.num_outputs)
        return extra_info

    def update_state(self) -> None:
        """Update the state of the optimisation campaign."""
        self.state.num_evaluations = len(self.dataset.data)

        # Extract extra info from the model
        self.state.extra_info = ", ".join(
            f"{key}: {value}" for key, value in self.get_extra_info().items()
        )

        # Flag the model as outdated
        self.__model = None

        # Multi-objective case: update the MO state data
        if self.objective.is_multi_objective():
            self.state.mo_state = MultiObjectiveStateData.from_dataset(
                self.dataset, self.optim_settings
            )
            self.state.best_value = -self.state.mo_state.hypervolume

        # Deterministic single-objective case: find the best observation
        elif not self.is_stochastic():
            y_points = torch.tensor(
                [self.objective.get_objective_value(obs.result) for obs in self.dataset.data]
            )
            idx = int(torch.argmin(y_points).item())
            self.state.best_value = float(y_points[idx].item())
            self.state.best_params = self.dataset.data[idx].params
            self.state.best_result = self.dataset.data[idx].result

        # Stochastic single-objective case with a single observation: it is our best guess
        elif self.state.num_evaluations == 1:
            value = self.objective.get_objective_value(self.dataset.data[0].result)
            variance = self.objective.get_objective_variance(self.dataset.data[0].result)
            self.state.best_value = value
            self.state.best_params = self.dataset.data[0].params
            self.state.best_result = self.dataset.data[0].result
            self.state.conf_interval = (
                -(value + 1.96 * np.sqrt(variance)),
                -(value - 1.96 * np.sqrt(variance))
            )

        # Stochastic single-objective case: find the value by optimising the model's posterior mean
        else:
            model = self.get_model()
            best_params, best_value = get_best_posterior_mean(
                model, self.settings.parameters, self.state
            )
            self.state.best_value = float(best_value.item())
            self.state.best_params = best_params.cpu().reshape(-1).numpy()
            self.state.best_result = None
            # Sample from the model to estimate the confidence interval for the best point
            samples = model.objective_samples(
                best_params, sample_shape=torch.Size([1024]), seed=self.settings.seed
            )
            self.state.conf_interval = (
                -float(torch.quantile(samples, 0.975).item()),
                -float(torch.quantile(samples, 0.025).item()),
            )


class CandidatePolicy(ABC):
    """Abstract base class for candidate selection policies."""

    def __init__(
        self,
        rounds: int,
        requires_model: bool = False,
        num_workers: Optional[int] = None,
        num_candidates_per_round: Optional[int] = None,
        mode: Optional[Literal['sequential', 'batched', 'async']] = None,
    ) -> None:
        # Set up defaults if not specified
        if num_workers is None:
            # Assume that our compute budget is limited: run in sequential mode by default
            num_workers = 1
            if mode is None:
                mode = 'sequential'
            if num_candidates_per_round is None:
                num_candidates_per_round = 1
        else:
            if mode is None:
                # Select the default mode based on the number of candidates per round
                if num_candidates_per_round is None:
                    mode = 'async' if num_workers > 1 else 'sequential'
                    num_candidates_per_round = 1
                else:
                    if num_candidates_per_round > 1:
                        mode = 'batched'
                    else:
                        mode = 'async' if num_workers > 1 else 'sequential'
            elif num_candidates_per_round is None:
                # We have a mode and number of workers, but not the number of candidates per round
                num_candidates_per_round = num_workers if mode == 'batched' else 1

        # Sanity checks
        if rounds <= 0:
            raise ValueError("At least one round is required.")
        if num_workers <= 0:
            raise ValueError("At least one worker is required.")
        if num_candidates_per_round <= 0:
            raise ValueError("Number of candidates per round must be positive if specified.")
        if mode not in ['sequential', 'batched', 'async']:
            raise ValueError(
                f"Invalid mode '{mode}' for candidate policy. "
                "Must be one of 'sequential', 'batched', or 'async'."
            )
        if mode == 'async' and num_candidates_per_round != 1:
            raise ValueError("Asynchronous candidate policies must have 1 candidate per round.")

        # Warnings
        if mode in ('batched', 'async') and num_workers == 1:
            warnings.warn(
                f"Candidate policy is in '{mode}' mode but only 1 worker is specified. "
                "Switching to 'sequential' mode to avoid unnecessary overhead."
            )
            mode = 'sequential'

        # Initialise attributes
        self.mode = mode
        self.rounds = rounds
        self.num_workers = num_workers
        self.requires_model = requires_model
        self.num_candidates_per_round = num_candidates_per_round

    def get_num_rounds(self) -> int:
        """Get the number of rounds for this candidate policy.

        Returns
        -------
        int
            The number of rounds for this candidate policy.
        """
        return self.rounds

    def _run_sequential(
        self,
        campaign: OptimisationCampaign,
        round_callback: Callable[[int, list[tuple[np.ndarray, ObjectiveResult]]], bool],
        eval_callback: Callable[[np.ndarray, ObjectiveResult], None],
    ) -> bool:
        """Run the candidate evaluation policy (in sequential mode).

        Parameters
        ----------
        campaign : OptimisationCampaign
            The optimisation campaign for which to run the policy.
        round_callback : Callable[[int, list[tuple[np.ndarray, ObjectiveResult]]], bool]
            Callback function to report the completion of a round. Takes the round number and a
            list of candidates and their evaluation results for the completed round, and returns
            whether to stop the campaign.
        eval_callback : Callable[[np.ndarray, ObjectiveResult], None]
            Callback function to report the completion of an evaluation. Takes the candidate and
            its evaluation result.

        Returns
        -------
        bool
            If we should stop the campaign after this evaluation.
        """
        for round_num in range(self.rounds):
            # Generate the next batch of candidates and evaluate them sequentially
            candidates = self.get_next_candidates(
                self.num_candidates_per_round,
                campaign.state,
                campaign.dataset,
                campaign.pending,
                campaign.get_model() if self.requires_model else None,
            )
            results = []
            for candidate in candidates:
                result = campaign.evaluate(candidate)
                results.append((candidate, result))
                eval_callback(candidate, result)

            # Report round completion and check if we should stop the campaign
            if round_callback(round_num, results):
                return True

        # All rounds completed without stopping the campaign
        return False

    def _run_batched(
        self,
        campaign: OptimisationCampaign,
        round_callback: Callable[[int, list[tuple[np.ndarray, ObjectiveResult]]], bool],
        eval_callback: Callable[[np.ndarray, ObjectiveResult], None],
    ) -> bool:
        """Run the candidate evaluation policy (in batched mode).

        Parameters
        ----------
        campaign : OptimisationCampaign
            The optimisation campaign for which to run the policy.
        round_callback : Callable[[int, list[tuple[np.ndarray, ObjectiveResult]]], bool]
            Callback function to report the completion of a round. Takes the round number and a
            list of candidates and their evaluation results for the completed round, and returns
            whether to stop the campaign.
        eval_callback : Callable[[np.ndarray, ObjectiveResult], None]
            Callback function to report the completion of an evaluation. Takes the candidate and
            its evaluation result.

        Returns
        -------
        bool
            If we should stop the campaign after this evaluation.
        """
        # If only one worker is specified, run sequentially to avoid unnecessary overhead
        if self.num_workers <= 1:
            return self._run_sequential(campaign, round_callback, eval_callback)

        # Otherwise, run with parallelism using a thread pool
        with futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for round_num in range(self.rounds):
                # Generate the next batch of candidates and submit them for evaluation
                candidates = self.get_next_candidates(
                    self.num_candidates_per_round,
                    campaign.state,
                    campaign.dataset,
                    campaign.pending,
                    campaign.get_model() if self.requires_model else None,
                )
                future_to_candidate = {
                    executor.submit(campaign.evaluate, candidate): candidate
                    for candidate in candidates
                }

                # As evaluations complete, report the results to the policy
                for future in futures.as_completed(future_to_candidate):
                    candidate = future_to_candidate[future]
                    eval_callback(candidate, future.result())

                # Report round completion and check if we should stop the campaign
                results = [(cand, fut.result()) for fut, cand in future_to_candidate.items()]
                if round_callback(round_num, results):
                    # Wait for all evaluations to complete before stopping the campaign
                    futures.wait(future_to_candidate, return_when=futures.ALL_COMPLETED)
                    return True

        # All rounds completed without stopping the campaign
        return False

    def _run_async(
        self,
        campaign: OptimisationCampaign,
        round_callback: Callable[[int, list[tuple[np.ndarray, ObjectiveResult]]], bool],
        eval_callback: Callable[[np.ndarray, ObjectiveResult], None],
    ) -> bool:
        """Run the candidate evaluation policy (in asynchronous mode).

        Parameters
        ----------
        campaign : OptimisationCampaign
            The optimisation campaign for which to run the policy.
        round_callback : Callable[[int, list[tuple[np.ndarray, ObjectiveResult]]], bool]
            Callback function to report the completion of a round. Takes the round number and a
            list of candidates and their evaluation results for the completed round, and returns
            whether to stop the campaign.
        eval_callback : Callable[[np.ndarray, ObjectiveResult], None]
            Callback function to report the completion of an evaluation. Takes the candidate and
            its evaluation result.

        Returns
        -------
        bool
            If we should stop the campaign after this evaluation.
        """
        # If only one worker is specified, run sequentially to avoid unnecessary overhead
        if self.num_workers <= 1:
            return self._run_sequential(campaign, round_callback, eval_callback)

        with futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Generate and submit the initial batch of candidates to start the campaign
            num_evals = 0
            num_init = min(self.num_workers, self.rounds)
            candidates = self.get_next_candidates(
                num_init,
                campaign.state,
                campaign.dataset,
                campaign.pending,
                campaign.get_model() if self.requires_model else None,
            )
            future_to_candidate = {
                executor.submit(campaign.evaluate, candidate): candidate
                for candidate in candidates
            }

            # Remaining rounds: submit new candidates as evaluations complete
            while num_evals < self.rounds:
                # Wait for the next evaluation(s) to complete
                done, _ = futures.wait(
                    future_to_candidate, return_when=futures.FIRST_COMPLETED
                )

                # Handle each completed evaluation
                for future in done:
                    # Report the completed evaluation (and check if we should stop the campaign)
                    candidate = future_to_candidate[future]
                    eval_callback(candidate, future.result())
                    if round_callback(num_evals, [(candidate, future.result())]):
                        # Wait for all evaluations to complete
                        futures.wait(future_to_candidate, return_when=futures.ALL_COMPLETED)
                        return True

                    # Update state
                    num_evals += 1
                    del future_to_candidate[future]

                    # Submit a new candidate if we have more rounds to go: we trust the policy to
                    # use the latest state, model, and list of pending evaluations when generating
                    # the next candidate
                    if num_evals + len(future_to_candidate) < self.rounds:
                        candidates = self.get_next_candidates(
                            1,
                            campaign.state,
                            campaign.dataset,
                            campaign.pending,
                            campaign.get_model() if self.requires_model else None,
                        )
                        future = executor.submit(campaign.evaluate, candidates[0])
                        future_to_candidate[future] = candidates[0]

        # All rounds completed without stopping the campaign
        return False

    def run(
        self,
        campaign: OptimisationCampaign,
        round_callback: Callable[[int, list[tuple[np.ndarray, ObjectiveResult]]], bool],
        eval_callback: Callable[[np.ndarray, ObjectiveResult], None],
    ) -> bool:
        """Run the candidate evaluation policy.

        Parameters
        ----------
        campaign : OptimisationCampaign
            The optimisation campaign for which to run the policy.
        round_callback : Callable[[int, list[tuple[np.ndarray, ObjectiveResult]]], bool]
            Callback function to report the completion of a round. Takes the round number and a
            list of candidates and their evaluation results for the completed round, and returns
            whether to stop the campaign.
        eval_callback : Callable[[np.ndarray, ObjectiveResult], None]
            Callback function to report the completion of an evaluation. Takes the candidate and
            its evaluation result.

        Returns
        -------
        bool
            If we should stop the campaign after this evaluation.
        """
        # Even if we have a non-sequential model, run sequentially when we have a single worker
        if self.num_workers <= 1 or self.mode == 'sequential':
            return self._run_sequential(campaign, round_callback, eval_callback)

        # Batched mode: submit batches of candidates and wait for completion after each batch
        if self.mode == 'batched':
            return self._run_batched(campaign, round_callback, eval_callback)

        # Asynchronous mode: submit a new candidate as soon as one evaluation completes
        if self.mode == 'async':
            return self._run_async(campaign, round_callback, eval_callback)
        raise ValueError(f"Unsupported candidate mode: {self.mode}")

    @abstractmethod
    def get_next_candidates(
        self,
        num_candidates: int,
        state: OptimisationState,
        dataset: ObjectiveDataset,
        pending: list[np.ndarray],
        model: Optional[ObjectiveModel],
    ) -> list[np.ndarray]:
        """Get the next candidates to evaluate based on the current state.

        Parameters
        ----------
        num_candidates : int
            Number of candidates to generate.
        state : OptimisationState
            Current state of the optimisation campaign.
        dataset : ObjectiveDataset
            Dataset containing the observations from the optimisation campaign.
        pending : list[np.ndarray]
            List of candidates that have been submitted for evaluation but have not completed yet.
        model : Optional[ObjectiveModel]
            Surrogate model for the objective function, if required.

        Returns
        -------
        list[np.ndarray]
            List of parameters for the next candidates to evaluate.
        """

    @classmethod
    @abstractmethod
    def read(cls: type[T], config: dict[str, Any]) -> T:
        """Read a candidate policy from the given configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary for the candidate policy.

        Returns
        -------
        T
            The created candidate policy instance.
        """
