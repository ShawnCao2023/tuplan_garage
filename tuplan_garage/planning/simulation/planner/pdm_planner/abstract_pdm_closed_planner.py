from typing import List, Optional

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory, )
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    state_array_to_ego_state, state_array_to_ego_states)
from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import (
    AbstractPDMPlanner, )
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation, )
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy, )
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_generator import (
    PDMGenerator, )
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_proposal import (
    PDMProposalManager, )
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer, )
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator, )
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import (
    PDMEmergencyBrake, )
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    parallel_discrete_path, )
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.plot_ego_states import (
    plot_ego_states, plot_multi_paths, plot_multi_paths_baseon_car_frame)


class AbstractPDMClosedPlanner(AbstractPDMPlanner):
    """
    Interface for planners incorporating PDM-Closed. Used for PDM-Closed and PDM-Hybrid.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
    ):
        """
        Constructor for AbstractPDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """

        super(AbstractPDMClosedPlanner, self).__init__(map_radius)

        assert (
            trajectory_sampling.interval_length ==
            proposal_sampling.interval_length
        ), "AbstractPDMClosedPlanner: Proposals and Trajectory must have equal interval length!"

        # config parameters
        self._trajectory_sampling: int = trajectory_sampling
        self._proposal_sampling: int = proposal_sampling
        self._idm_policies: BatchIDMPolicy = idm_policies
        self._lateral_offsets: Optional[List[float]] = lateral_offsets

        # observation/forecasting class
        self._observation = PDMObservation(trajectory_sampling,
                                           proposal_sampling, map_radius)

        # proposal/trajectory related classes
        self._generator = PDMGenerator(trajectory_sampling, proposal_sampling)
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)
        self._emergency_brake = PDMEmergencyBrake(trajectory_sampling)

        # lazy loaded
        self._proposal_manager: Optional[PDMProposalManager] = None

    def _update_proposal_manager(self, ego_state: EgoState):
        """
        Updates or initializes PDMProposalManager class
        :param ego_state: state of ego-vehicle
        """

        current_lane = self._get_starting_lane(ego_state)

        # TODO: Find additional conditions to trigger re-planning
        create_new_proposals = self._iteration == 0

        if create_new_proposals:
            proposal_paths: List[PDMPath] = self._get_proposal_paths(
                current_lane)

            self._proposal_manager = PDMProposalManager(
                lateral_proposals=proposal_paths,
                longitudinal_policies=self._idm_policies,
            )

        # update proposals
        self._proposal_manager.update(current_lane.speed_limit_mps)

    def _get_proposal_paths(
            self, current_lane: LaneGraphEdgeMapObject) -> List[PDMPath]:
        """
        Returns a list of path's to follow for the proposals. Inits a centerline.
        :param current_lane: current or starting lane of path-planning
        :return: lists of paths (0-index is centerline)
        """
        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        self._centerline = PDMPath(centerline_discrete_path)

        # 1. save centerline path (necessary for progress metric)
        output_paths: List[PDMPath] = [self._centerline]

        # 2. add additional paths with lateral offset of centerline
        if self._lateral_offsets is not None:
            for lateral_offset in self._lateral_offsets:
                offset_discrete_path = parallel_discrete_path(
                    discrete_path=centerline_discrete_path,
                    offset=lateral_offset)
                output_paths.append(PDMPath(offset_discrete_path))

        return output_paths

    def _get_closed_loop_trajectory(
        self,
        current_input: PlannerInput,
    ) -> InterpolatedTrajectory:
        """
        Creates the closed-loop trajectory for PDM-Closed planner.
        :param current_input: planner input
        :return: trajectory
        """

        sim_start_time = TimePoint(current_input.iteration.time_point.time_us)
        sim_start_state = current_input.history.current_state[0]
        tree_search_time = TimePoint(
            current_input.iteration.time_point.time_us)

        vehicle_parameter = current_input.history.current_state[
            0].car_footprint.vehicle_parameters

        iteration_times = 0
        sim_results = []
        proposal_results = []
        final_sim_trajectory = np.array([])
        simulated_proposals_array = []
        proposal_scores = []

        while iteration_times < 2:
            if iteration_times == 0:
                ego_state, observation = current_input.history.current_state
            elif iteration_times == 1:
                best_score_final_state = simulated_proposals_array[np.argmax(
                    proposal_scores)]
                tree_search_time = tree_search_time + TimeDuration.from_s(
                    self._proposal_sampling.time_horizon)
                ego_state = state_array_to_ego_state(
                    best_score_final_state[-1], tree_search_time,
                    vehicle_parameter)
                # TODO(caoxiaoxu) get the simulated observation

            iteration_times += 1
            # 1. Environment forecast and observation update
            self._observation.update(
                ego_state,
                observation,
                current_input.traffic_light_data,
                self._route_lane_dict,
            )

            # 2. Centerline extraction and proposal update
            self._update_proposal_manager(ego_state)

            # 3. Generate/Unroll proposals
            proposals_array = self._generator.generate_proposals(
                ego_state, self._observation, self._proposal_manager)

            # 4. Simulate proposals
            simulated_proposals_array = self._simulator.simulate_proposals(
                proposals_array, ego_state)

            # 5. Score proposals
            proposal_scores = self._scorer.score_proposals(
                simulated_proposals_array,
                ego_state,
                self._observation,
                self._centerline,
                self._route_lane_dict,
                self._drivable_area_map,
                self._map_api,
            )
            if final_sim_trajectory.size == 0:
                final_sim_trajectory = simulated_proposals_array[np.argmax(
                    proposal_scores)]
            else:
                final_sim_trajectory = np.vstack(
                    (final_sim_trajectory,
                     simulated_proposals_array[np.argmax(proposal_scores)]))

            pesudo_time_list = [tree_search_time
                                ] * simulated_proposals_array.shape[1]
            for item in proposals_array:
                proposal_results.append(
                    state_array_to_ego_states(item, pesudo_time_list,
                                              vehicle_parameter))
            for item in simulated_proposals_array:
                sim_results.append(
                    state_array_to_ego_states(item, pesudo_time_list,
                                              vehicle_parameter))

        plot_multi_paths_baseon_car_frame(
            sim_results, sim_start_state,
            '/home/PJLAB/caoxiaoxu/ws/tuplan_garage/sim_results.png')
        plot_multi_paths_baseon_car_frame(
            proposal_results, sim_start_state,
            '/home/PJLAB/caoxiaoxu/ws/tuplan_garage/proposal_results.png')

        # # 6.a Apply brake if emergency is expected
        # trajectory = self._emergency_brake.brake_if_emergency(
        #     ego_state, proposal_scores, self._scorer)

        # # 6.b Otherwise, extend and output best proposal
        # if trajectory is None:
        #     trajectory = self._generator.generate_trajectory(
        #         np.argmax(proposal_scores))

        time_list = []
        # drop the first elemetn of the final_sim_trajectory, it's the current pose
        # final_sim_trajectory = np.delete(final_sim_trajectory, 0, axis=0)
        for i in range(len(final_sim_trajectory)):
            time_list.append(sim_start_time + TimeDuration.from_s(
                self._proposal_sampling.interval_length * i))

        trajectory = InterpolatedTrajectory(
            state_array_to_ego_states(final_sim_trajectory, time_list,
                                      vehicle_parameter))

        return trajectory
