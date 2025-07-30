#!/usr/bin/env python3

# Followed code from https://github.com/3dlg-hcvc/hssd/

import itertools
import os
import random
from typing import List
import re

import cv2
import imageio
import numpy as np
import json
from scipy.spatial import distance

import habitat_sim
from utils import (
    COLOR_PALETTE,
    draw_obj_bbox_on_topdown_map,
    get_topdown_map,
)
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.datasets.pointnav.pointnav_generator import (
    # ISLAND_RADIUS_LIMIT,
    _ratio_sample_rate,
)
from habitat.datasets.utils import get_action_shortest_path
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.multi_object_nav_task import (
    MultiObjectGoal,
    MultiObjectGoalNavEpisode,
    ObjectViewLocation,
)
from habitat.tasks.utils import compute_pixel_coverage
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_two_vectors,
)
from habitat.utils.visualizations.utils import observations_to_image
from habitat_sim.errors import GreedyFollowerError
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_to_angle_axis,
    quat_to_coeffs,
)

VISIBILITY_THRESHOLD = 0.001
ISLAND_RADIUS_LIMIT = 2.5
INSTRUCTION_FORMAT = {
    "single_objects": [
            "Find the {}.",
            "Go to the {}."
        ],
    "spatial_top": [
            "Find the {} on the {}.",
            "Go to the {} on the {}."
        ]
    }

def _direction_to_quaternion(direction_vector: np.array):
    origin_vector = np.array([0, 0, -1])
    output = quaternion_from_two_vectors(origin_vector, direction_vector)
    output = output.normalized()
    return output


def _get_multipath(sim: HabitatSim, start, ends):
    multi_goal = habitat_sim.MultiGoalShortestPath()
    multi_goal.requested_start = start
    multi_goal.requested_ends = ends
    sim.pathfinder.find_path(multi_goal)
    return multi_goal


def _get_action_shortest_path(
    sim: HabitatSim, start_pos, start_rot, goal_pos, goal_radius=0.05
):
    sim.set_agent_state(start_pos, start_rot, reset_sensors=True)
    greedy_follower = sim.make_greedy_follower()
    return greedy_follower.find_path(goal_pos)


def is_compatible_episode(
    source_position,
    goal_positions,
    sim: HabitatSim,
    goals: List[MultiObjectGoal],
    near_dist,
    far_dist,
    geodesic_to_euclid_ratio,
    same_floor_flag=False,
):
    FAIL_TUPLE = False, 0, 0
    s = np.array(source_position)

    if same_floor_flag:
        valid = []
        for gt in goal_positions:
            gt = np.array(gt)
            valid.append(np.abs(gt[1] - s[1]) < 1.5)
            
            if not valid:
                return FAIL_TUPLE

    shortest_paths_to_goals = []
    pf = sim.pathfinder
    geod_distances = []
    total_geodesic_dist = 0
    euc_distances = []
    total_euc_dist = 0
    for gt in goal_positions:
        
        # Find path between start location and object location
        geod_distance = sim.geodesic_distance(s, gt)
        geod_distances.append(geod_distance)
        total_geodesic_dist += geod_distance
        
        euc_dist = distance.euclidean(s, gt)
        euc_distances.append(euc_dist)
        total_euc_dist += euc_dist

        # Is there a shortest path between these two points
        if np.isinf(geod_distance):
            return FAIL_TUPLE

        shortest_paths_to_goals.append(sim.get_straight_shortest_path_points(s, gt))
        
    # Geodesic Distance constraint for the first goal
    geod_distance = geod_distances[0]
    if geod_distance < 2:
        return FAIL_TUPLE
    
    # Check ratio of Geodesic Distance to Euclidean Distance for the first goal
    euclidean_distance = euc_distances[0]
    if geod_distance/euclidean_distance < geodesic_to_euclid_ratio:
        return FAIL_TUPLE
        
    angle = np.random.uniform(0, 2 * np.pi)
    source_rotation = [
        0,
        np.sin(angle / 2),
        0,
        np.cos(angle / 2),
    ]  # Pick random starting rotation

    return (
        True,
        source_rotation,
        shortest_paths_to_goals,
        total_geodesic_dist,
        total_euc_dist,
    )


def build_goal(
    sim: HabitatSim,
    object_id: int,
    object_name_id: str,
    object_category_name: str,
    object_category_id: int,
    object_category_desc: str,
    obj_instance_name: str,
    object_position,
    grid_radius: float = 10.0,
):
    
    pf = sim.pathfinder
    
    # Check whether there are any navigable positions around the object within a distance of grid_radius (meters)
    pt_nav = pf.get_random_navigable_point_near(
                object_position,
                grid_radius,
                max_tries=10000)
    
    if np.isnan(pt_nav).any():
        return None, None

    # reject if on an island of radius less than ISLAND_RADIUS_LIMIT
    if sim.island_radius(pt_nav) < ISLAND_RADIUS_LIMIT:
        return None, None
            
    # reject if the object is located outdoors
    if sim.pathfinder.get_island(pt_nav) not in sim.indoor_islands:
        return None, None

    object_instruction = decode_instruction(object_category_desc)

    goal = MultiObjectGoal(
        position=np.array(object_position).tolist(),
        # view_points=view_locations,
        object_id=object_id,
        object_category=object_category_name,
        object_name=object_name_id,
        language_instruction=object_instruction,
        obj_instance_name=obj_instance_name,
    )

    return goal, None

def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))

def clean_description(description):
    description = description.replace('_', ' ').lower()
    return re.sub('[^a-zA-Z0-9 \n\.]', '', description)

def decode_instruction(obj_str):
    obj_det = json.loads(obj_str)
    
    if len(obj_det["parent_wnsynsetkey"]) > 0:
        instr = random.choice(INSTRUCTION_FORMAT["spatial_top"])
        # goal_name = clean_description(obj_det["obj_name"])
        # parent_name = clean_description(obj_det["parent_name"])
        
        goal_obj_name = obj_det["obj_name"]
        if has_numbers(goal_obj_name):
            goal_obj_name = obj_det["obj_main_cat"]
            
        goal_name = clean_description(goal_obj_name)
        
        if len(obj_det["parent_main_cat"]) > 0:
            parent_name = clean_description(obj_det["parent_main_cat"])
        elif len(obj_det["parent_super_cat"]) > 0:
            parent_name = clean_description(obj_det["parent_super_cat"])
        elif len(obj_det["parent_wnsynsetkey"]) > 0:
            parent_name = clean_description(obj_det["parent_wnsynsetkey"])
        instr = instr.format(goal_name, parent_name)
    else:
        instr = random.choice(INSTRUCTION_FORMAT["single_objects"])
        goal_obj_name = obj_det["obj_name"]
        if has_numbers(goal_obj_name):
            goal_obj_name = obj_det["obj_main_cat"]
            
        goal_name = clean_description(goal_obj_name)
        instr = instr.format(goal_name)
    
    return instr

def _create_episode(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    goals,
    shortest_paths=None,
    scene_state=None,
    info=None,
    scene_dataset_config="default",
):
    
    return MultiObjectGoalNavEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        object_category=[goals[0].object_category],
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
        scene_dataset_config=scene_dataset_config,
    )

def generate_multion_episode(
    sim: HabitatSim,
    goals: List[MultiObjectGoal],
    closest_dist_limit: float = 0.2,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.05,
    number_retries_per_cluster: int = 1000,
    scene_dataset_config: str = "default",
    same_floor_flag: bool = False,
    eps_generated: int = 0,
    goal_radius: float = 5.0,
):
    r"""Generator function that generates PointGoal navigation episodes.
    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.
    :param sim: simulator with loaded scene for generation.
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    :param same_floor_flag should object exist on same floor as agent's start?
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    
    episode_id = eps_generated
    
    goal_positions = [g.position for g in goals]
    goal_pos = goal_positions[0]
    
    _tries = 0
    is_compatible = False
    while _tries < 10000 and not is_compatible:
        _tries += 1
        source_position = (
                        sim.pathfinder.get_random_navigable_point_near(
                            goal_pos, goal_radius, max_tries=10000
                        )
                    )
        
        compat_outputs = is_compatible_episode(
            source_position,
            goal_positions,
            sim,
            goals,
            near_dist=closest_dist_limit,
            far_dist=furthest_dist_limit,
            geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            same_floor_flag=same_floor_flag,
        )
        is_compatible = compat_outputs[0]

    if is_compatible:
        (
            is_compatible,
            source_rotation,
            shortest_paths,
            geodesic_dist,
            euclid_dist
        ) = compat_outputs
        
        episode = _create_episode(
            episode_id=episode_id,
            scene_id=sim.habitat_config.scene,
            start_position=source_position,
            start_rotation=source_rotation,
            shortest_paths=shortest_paths,
            info={
                "geodesic_distance": geodesic_dist,
                "euclidean_distance": euclid_dist,
                "closest_goal_object_id": goals[0].object_id,
            },
            goals=goals,
            scene_dataset_config=scene_dataset_config,
        )

        return episode
    
    return None


def update_objectnav_episode_v2(
    sim: HabitatSim,
    goals: List[MultiObjectGoal],
    episode: MultiObjectGoalNavEpisode,
):
    r"""Updates an existing episode with the goals.
    :param sim: simulator with loaded scene for generation.
    """
    ############################################################################
    # Compute distances
    ############################################################################
    source_position = episode.start_position
    source_position = np.array(source_position)
    goal_targets = [
        [vp.agent_state.position for vp in goal.view_points] for goal in goals
    ]
    closest_goal_targets = (
        sim.geodesic_distance(source_position, vps) for vps in goal_targets
    )
    closest_goal_targets, goals_sorted = zip(
        *sorted(zip(closest_goal_targets, goals), key=lambda x: x[0])
    )
    d_separation = closest_goal_targets[0]
    shortest_path = None
    euclid_dist = np.linalg.norm(source_position - goals_sorted[0].position)
    ############################################################################
    # Create new episode with updated information
    ############################################################################
    if shortest_path is None:
        shortest_paths = None
    else:
        shortest_paths = [shortest_path]
    episode_new = _create_episode(
        episode_id=episode.episode_id,
        scene_id=sim.habitat_config.scene,
        start_position=episode.start_position,
        start_rotation=episode.start_rotation,
        shortest_paths=shortest_paths,
        scene_state=None,
        info={
            "geodesic_distance": d_separation,
            "euclidean_distance": euclid_dist,
            "closest_goal_object_id": goals_sorted[0].object_id,
        },
        goals=goals_sorted,
        scene_dataset_config=episode.scene_dataset_config,
    )
    return episode_new

