"""Module with the generic optimiser class for piglot."""
from typing import Callable, TypeVar, Any
from functools import partial
import numpy as np
from tqdm import tqdm
from piglot.settings import Settings
from piglot.objective import Objective, ObjectiveResult
from piglot.optimiser import Optimiser, OptimisationResult
from piglot.optimisers.generic.policies import read_policy
from piglot.optimisers.generic.campaign import (
    OptimisationCampaign,
    CandidatePolicy,
    OptimisationSettings,
)


T = TypeVar('T', bound='GenericOptimiser')


class GenericOptimiser(Optimiser):
    """Generic optimiser class for piglot."""

    def __init__(
        self,
        settings: Settings,
        objective: Objective,
        policies: dict[str, CandidatePolicy],
        optim_settings: OptimisationSettings
    ) -> None:
        super().__init__(settings, objective)
        self.campaign = OptimisationCampaign(settings, optim_settings, objective)
        self.policies = policies
        self.total_iters = sum(policy.get_num_rounds() for policy in policies.values())

    def _progress_report_prepare(self) -> None:
        """Initialising the progress bar."""
        if not self.settings.quiet:
            self.pbar = tqdm(total=self.total_iters, desc=self.name())

    def _progress_report_update(self, i_iter: int, extra_info: dict[str, str]) -> None:
        """Update the progress bar.

        Parameters
        ----------
        i_iter : int
            Current iteration number.
        result : OptimisationResult
            Result of the current iteration.
        extra_info : dict[str, str]
            Additional information to pass to user.
        """

    def name(self) -> str:
        """Name of the optimiser.

        Returns
        -------
        str
            Name of the optimiser.
        """
        return 'piglot'

    def update_progress_name(self, name: str, policy: CandidatePolicy, round_num: int) -> None:
        """Update the progress bar name based on the current policy and round.

        Parameters
        ----------
        name : str
            The name of the policy.
        policy : CandidatePolicy
            The policy used for selecting candidates.
        round_num : int
            The number of the current round.
        """
        description = name
        if policy.get_num_rounds() > 1:
            description += f' (round {round_num}/{policy.get_num_rounds()})'
        self.pbar.set_description(description)

    @classmethod
    def validate_problem(cls, objective: Objective) -> None:
        """Validate the combination of optimiser and objective.

        Parameters
        ----------
        objective : Objective
            Objective to optimise.
        """

    def _optimise(
        self, callback: Callable[[int, OptimisationResult, dict[str, str]], bool]
    ) -> OptimisationResult:
        """Abstract method for optimising the objective.

        Parameters
        ----------
        callback : Callable[[OptimisationResult, dict[str, str]], bool]
            Callback function for reporting the optimiser progress and checking for termination.
            The first argument is the current optimisation result, while the second argument is a
            dictionary with additional information to pass to the user. Call this function at the
            end of each iteration, and if it returns True, stop the optimisation.

        Returns
        -------
        OptimisationResult
            Result of the optimisation.
        """

        # Set up callbacks
        def report_evaluation(candidate: np.ndarray, result: ObjectiveResult) -> None:
            """Report the result of a candidate evaluation to the policy.

            Parameters
            ----------
            candidate : np.ndarray
                The candidate that was evaluated.
            result : ObjectiveResult
                The result of the evaluation.
            """
            if self.pbar is not None:
                state = self.campaign.state.get_result()
                info = f'Loss: {state.value:6.3e}'
                if (
                    self.objective.has_variance()
                    and state.conf_interval
                    and all(state.conf_interval)
                ):
                    delta = (state.conf_interval[1] - state.conf_interval[0]) / 2
                    info += f' ± {delta:6.3e}'
                self.pbar.set_postfix_str(info)

        def report_round(
            name: str,
            policy: CandidatePolicy,
            round_num: int,
            results: list[tuple[np.ndarray, ObjectiveResult]],
        ) -> bool:
            """Report the completion of a round to the policy.

            Parameters
            ----------
            name : str
                The name of the policy.
            policy : CandidatePolicy
                The policy used for selecting candidates.
            round_num : int
                The number of the round that was completed.
            results : list[tuple[np.ndarray, ObjectiveResult]]
                List of candidates and their evaluation results for the completed round.

            Returns
            -------
            bool
                If we should stop the campaign after this round.
            """
            result = OptimisationResult(
                value=self.campaign.state.best_value,
                params=self.campaign.state.best_params,
                conf_interval=self.campaign.state.conf_interval,
            )
            if self.pbar is not None:
                self.update_progress_name(name, policy, round_num + 1)
                self.pbar.update(1)
            return callback(round_num, result, {})

        # Run the campaign
        for name, policy in self.policies.items():
            # Update the name of the progress bar to reflect the current policy
            self.update_progress_name(name, policy, 0)
            # Run the policy for the current campaign
            if policy.run(self.campaign, partial(report_round, name, policy), report_evaluation):
                break

        # Return the best result found
        return self.campaign.state.get_result()

    @classmethod
    def read(cls: type[T], config: dict[str, Any], settings: Settings, objective: Objective) -> T:
        """Read an optimiser from the given configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary for the optimiser.
        settings : Settings
            Settings for the optimiser.
        objective : Objective
            Objective to optimise.

        Returns
        -------
        T
            The created optimiser instance.
        """
        # Mandatory fields
        if 'policies' not in config:
            raise ValueError('Missing "policies" field in optimiser configuration')
        policies = []
        policies_dict = config.pop('policies')
        policies = {
            name: read_policy(policy_config) for name, policy_config in policies_dict.items()
        }
        optim_settings = OptimisationSettings.read(config)
        return cls(settings, objective, policies, optim_settings)
