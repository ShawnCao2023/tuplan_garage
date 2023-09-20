from typing import List, Optional

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState

from matplotlib import pyplot as plt
import matplotlib.patches as patches


def plot_ego_states(ego_states: List[EgoState], color: str = "k") -> None:
    """
    Plots ego states in 2D.
    :param ego_states: list of ego states
    :param color: color of the plot
    """
    plt.figure(dpi=150)
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    x = np.array([state.car_footprint.center.x for state in ego_states])
    y = np.array([state.car_footprint.center.y for state in ego_states])
    heading = np.array(
        [state.car_footprint.center.heading for state in ego_states])
    u = np.cos(heading)
    v = np.sin(heading)
    q = ax.quiver(x, y, u, v)


def plot_multi_paths(ego_path: List[List[EgoState]], color: str = "k") -> None:
    """
    Plots ego states in 2D.
    :param ego_states: list of ego states
    :param color: color of the plot
    """
    plt.figure(dpi=150)
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    label_num = 0
    for ego_state in ego_path:
        x = np.array([state.car_footprint.center.x for state in ego_state])
        y = np.array([state.car_footprint.center.y for state in ego_state])
        heading = np.array(
            [state.car_footprint.center.heading for state in ego_state])
        u = np.cos(heading)
        v = np.sin(heading)
        q = ax.quiver(x, y, u, v, label=str(label_num))
    plt.savefig("/home/PJLAB/caoxiaoxu/ws/tuplan_garage/ego_states.png")


def plot_multi_pose_baseon_car_frame(ego_path: List[List[EgoState]],
                                     base: EgoState,
                                     color: str = "k") -> None:
    """
    Plots ego states in 2D.
    :param ego_states: list of ego states
    :param color: color of the plot
    """
    fig, ax = plt.subplots()

    ax.set_aspect(1)
    label_num = 0
    for ego_state in ego_path:
        x = np.array([
            state.car_footprint.center.x - base.car_footprint.center.x
            for state in ego_state
        ])
        y = np.array([
            state.car_footprint.center.y - base.car_footprint.center.y
            for state in ego_state
        ])
        heading = np.array([
            state.car_footprint.center.heading -
            base.car_footprint.center.heading for state in ego_state
        ])
        u = np.cos(heading)
        v = np.sin(heading)
        q = ax.quiver(x,
                      y,
                      u,
                      v,
                      scale=0.1,
                      label=str(label_num),
                      units='width')
    plt.savefig("/home/PJLAB/caoxiaoxu/ws/tuplan_garage/ego_states.png",
                dpi=300)


def plot_multi_paths_baseon_car_frame(ego_path: List[List[EgoState]],
                                      base: EgoState,
                                      file_name="default.png",
                                      color: str = "k") -> None:
    """
    Plots ego states in 2D.
    :param ego_states: list of ego states
    :param color: color of the plot
    """
    fig, ax = plt.subplots()

    # ax.set_aspect(1)
    label_num = 0
    for ego_state in ego_path:
        x = np.array([
            state.car_footprint.center.x - base.car_footprint.center.x
            for state in ego_state
        ])
        y = np.array([
            state.car_footprint.center.y - base.car_footprint.center.y
            for state in ego_state
        ])
        theta = base.car_footprint.center.heading
        trans_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
        inverse_trans_matrix = np.transpose(trans_matrix)
        m = np.array([x, y])
        xy_in_car_frame = np.dot(inverse_trans_matrix, m)
        rect = patches.Rectangle(
            (-0.5 * base.car_footprint.vehicle_parameters.length,
             -0.5 * base.car_footprint.vehicle_parameters.width),
            base.car_footprint.vehicle_parameters.length,
            base.car_footprint.vehicle_parameters.width,
            linewidth=1,
            edgecolor='r',
            facecolor='none')
        ax.add_patch(rect)
        ax.plot(xy_in_car_frame[0, :],
                xy_in_car_frame[1, :],
                label=str(label_num),
                linewidth=0.5,
                marker='o',
                markersize=1.0)
        label_num += 1
    plt.savefig(file_name, dpi=1000)
