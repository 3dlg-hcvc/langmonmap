# eval utils
import csv
import gc
import collections
import torch
from eval import within_fov_cone
from eval.actor import Actor
from eval.dataset_utils.gibson_dataset import load_gibson_episodes
from mapping import rerun_logger
from config import EvalConf
from onemap_utils import monochannel_to_inferno_rgb, generate_video, add_sim_maps_to_image, add_sim_maps_to_image_paper, monochannel_to_gray
from mapping.mlfm_utils import *
from eval.dataset_utils import *
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

# os / filsystem
import bz2
import os
from os import listdir
import gzip
import json
import pathlib
import tqdm

# cv2
import cv2

# numpy
import numpy as np

# skimage
import skimage
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# dataclasses
from dataclasses import dataclass

# quaternion
import quaternion

# typing
from typing import Tuple, List, Dict
import enum

# habitat
import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
from habitat_sim.utils import common as utils

# tabulate
from tabulate import tabulate

# rerun
import rerun as rr

# pandas
import pandas as pd

# pickle
import pickle

# scipy
from scipy.spatial.transform import Rotation as R


class Result(enum.Enum):
    NO_FAILURE = 1
    FAILURE_MISDETECT = 2
    FAILURE_STUCK = 3
    FAILURE_OOT = 4
    FAILURE_NOT_REACHED = 5
    FAILURE_ALL_EXPLORED = 6
    FAILURE_EXCEPTION = 7
    FAILURE_MISDETECT_ON_MAP = 8


class Metrics:
    def __init__(self, ep_id) -> None:
        self.sequence_lengths = []
        self.sequence_results = []
        self.sequence_poses = []
        self.ep_id = ep_id
        self.sequence_object = []

    def add_sequence(
        self, sequence: np.ndarray, result: Result, target_object: str
    ) -> None:
        start_id = 0
        if len(self.sequence_poses) > 0:
            start_id = sum([len(seq) for seq in self.sequence_poses])
        seq_poses = sequence[start_id:, :]
        self.sequence_poses.append(seq_poses)
        length = np.linalg.norm(seq_poses[1:, :2] - seq_poses[:-1, :2])
        self.sequence_results.append(result)
        self.sequence_lengths.append(length)
        self.sequence_object.append(target_object)

    def get_progress(self, num_seq):
        return self.sequence_results.count(Result.NO_FAILURE) / num_seq


class HabitatMultiEvaluator:
    def __init__(
        self,
        config: EvalConf,
        actor: Actor,
    ) -> None:
        self.config = config
        self.multi_object = config.multi_object
        self.max_steps = config.max_steps
        self.max_explore_steps = config.max_explore_steps
        self.max_exploit_steps = self.max_steps - self.max_explore_steps
        self.max_dist = config.max_dist
        self.controller = config.controller
        self.mapping = config.mapping
        self.planner = config.planner
        self.log_rerun = config.log_rerun
        self.save_video = config.save_video
        self.save_maps = config.save_maps
        self.object_nav_path = config.object_nav_path
        self.scene_path = config.scene_path
        self.scene_data = {}
        self.episodes = []
        self.exclude_ids = []
        self.include_ids = []
        self.is_gibson = config.is_gibson

        self.sim = None
        self.actor = actor
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True
        self.control_frequency = config.controller.control_freq
        self.max_vel = config.controller.max_vel
        self.max_ang_vel = config.controller.max_ang_vel
        self.time_step = 1.0 / self.control_frequency
        self.num_seq = config.num_seq
        self.square = config.square_im

        self.episodes, self.scene_data = LangMonDataset.load_hm3d_multi_episodes(
            self.episodes, self.scene_data, self.object_nav_path
        )

        if self.actor is not None:
            self.logger = (
                rerun_logger.RerunLogger(self.actor.mapper, False, "")
                if self.log_rerun
                else None
            )
        self.results_path = config.results_path if len(config.results_path) > 0 else "results_langmon/"
        self.object_metadata_file = pathlib.Path("datasets/scene_datasets/fphab/semantics/objects.csv")

        state_dir = os.path.join(self.results_path, "state")
        os.makedirs(state_dir, exist_ok=True)
        self.exclude_ids = [p.split('state_')[-1].split('.txt')[0] for p in os.listdir(state_dir)]
        self.include_ids = [] #['104348028_171512877__0']
        os.makedirs(os.path.join(self.results_path, 'trajectories'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'similarities'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'saved_maps'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'exceptions'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'extras'), exist_ok=True)

    def load_scene(self, scene_id: str):
        if self.sim is not None:
            self.sim.close()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
        backend_cfg.override_scene_light_defaults = True
        backend_cfg.pbr_image_based_lighting = True
        backend_cfg.scene_id = scene_id
        backend_cfg.enable_physics = True

        backend_cfg.scene_dataset_config_file = (
            self.scene_path + "hssd-hab-mon.scene_dataset_config.json"
        )
        backend_cfg.navmesh_settings = habitat_sim.nav.NavMeshSettings()
        backend_cfg.navmesh_settings.set_defaults()
        backend_cfg.navmesh_settings.agent_radius = 0.1
        backend_cfg.navmesh_settings.agent_height = 1.5
        backend_cfg.navmesh_settings.include_static_objects = True

        self.hfov = 90 if self.square else 79
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.hfov = self.hfov
        rgb.position = np.array([0, 1.5, 0])
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        res_x = 640
        res_y = 640 if self.square else 480
        rgb.resolution = [res_y, res_x]

        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth"
        depth.hfov = self.hfov
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.position = np.array([0, 1.5, 0])
        depth.resolution = [res_y, res_x]
        agent_cfg = habitat_sim.agent.AgentConfiguration(
            height=1.5,
            radius=0.1,
            action_space=dict(
                move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
                turn_left=ActionSpec("turn_left", ActuationSpec(amount=30.0)),
                turn_right=ActionSpec("turn_right", ActuationSpec(amount=30.0)),
            )
        )
        agent_cfg.sensor_specifications = [rgb, depth]
        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(sim_cfg)
        # self.sim = HabitatSim(sim_cfg)
        # if self.scene_data[scene_id].objects_loaded:
        #     return
        # self.scene_data = HM3DDataset.load_hm3d_objects(self.scene_data, self.sim.semantic_scene.objects, scene_id)

    def execute_action(self, action: Dict):
        if "discrete" in action.keys():
            # We have a discrete actor
            self.sim.step(action["discrete"])

        elif "continuous" in action.keys():
            # We have a continuous actor
            self.vel_control.angular_velocity = action["continuous"]["angular"]
            self.vel_control.linear_velocity = action["continuous"]["linear"]
            agent_state = self.sim.get_agent(0).state
            previous_rigid_state = habitat_sim.RigidState(
                utils.quat_to_magnum(agent_state.rotation), agent_state.position
            )

            # manually integrate the rigid state
            target_rigid_state = self.vel_control.integrate_transform(
                self.time_step, previous_rigid_state
            )

            # snap rigid state to navmesh and set state to object/sim
            # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
            end_pos = self.sim.step_filter(
                previous_rigid_state.translation, target_rigid_state.translation
            )

            # set the computed state
            agent_state.position = end_pos
            agent_state.rotation = utils.quat_from_magnum(target_rigid_state.rotation)
            self.sim.get_agent(0).set_state(agent_state)
            self.sim.step_physics(self.time_step)

    def load_object_metadata(self):
        ## load object metadata
        self.object_metadata = {}
        with open(self.object_metadata_file, "r") as f:
            _csv_reader = csv.reader(f)
            headings = next(_csv_reader)
            for row in _csv_reader:
                _data = row[:]
                m_json = {}
                for _ind, _val in enumerate(_data):
                    m_json[headings[_ind]] = _val
                self.object_metadata[_data[0]] = m_json

    def get_all_scene_objects(self):

        # load all objects in the scene
        rgm = self.sim.get_rigid_object_manager()
        object_info = rgm.get_objects_info()
        scene_objects_headers = object_info[0].split(",")
        scene_objects = [x.split(",") for x in object_info[1:]]
        scene_objects_metadata = {
            obj[0]: {"scene_data": obj, "metadata": self.object_metadata[obj[0].split("_")[0]]}
            for obj in scene_objects
        }

        scene_objects_metadata_by_main_category = {}
        scene_objects_metadata_by_wnsynsetkey = {}
        scene_objects_metadata_by_name = {}
        for obj_detail in scene_objects_metadata.values():
            scene_obj = obj_detail["scene_data"]
            obj = obj_detail["metadata"]
            if len(obj["main_category"]) > 0:
                cat = " ".join(obj["main_category"].split("_")).lower()
                if cat not in scene_objects_metadata_by_main_category:
                    scene_objects_metadata_by_main_category[cat] = []
                scene_objects_metadata_by_main_category[cat].append(
                    {"scene_data": scene_obj, "metadata": obj}
                )
            if len(obj["wnsynsetkey"]) > 0:
                cat = " ".join(obj["wnsynsetkey"].split(".n")[0].split("_")).lower()
                if cat not in scene_objects_metadata_by_wnsynsetkey:
                    scene_objects_metadata_by_wnsynsetkey[cat] = []
                scene_objects_metadata_by_wnsynsetkey[cat].append(
                    {"scene_data": scene_obj, "metadata": obj}
                )
            if len(obj["name"]) > 0:
                cat = obj["name"].lower()
                if cat not in scene_objects_metadata_by_name:
                    scene_objects_metadata_by_name[cat] = []
                scene_objects_metadata_by_name[cat].append(
                    {"scene_data": scene_obj, "metadata": obj}
                )

        return (
            rgm,
            scene_objects_headers,
            scene_objects_metadata,
            scene_objects_metadata_by_main_category,
            scene_objects_metadata_by_wnsynsetkey,
            scene_objects_metadata_by_name,
        )
    
    def get_closest_dist(self, pos, aabbs: List):
        min_dist = np.inf
        if self.sim is not None:
            for _goal in aabbs:
                for _viewpoint in _goal['navigable_points']:
                    shortest_path = habitat_sim.nav.ShortestPath()
                    shortest_path.requested_start = pos
                    shortest_path.requested_end = [float(_viewpoint[0]), float(_viewpoint[1]), float(_viewpoint[2])]
                    self.sim.pathfinder.find_path(shortest_path)
                    min_dist = min(min_dist, shortest_path.geodesic_distance)
        else:
            pos = pos[[0,2]]
            for _goal in aabbs:
                for _viewpoint in _goal['navigable_points']:
                    dx = pos[0] - float(_viewpoint[0])
                    dy = pos[1] - float(_viewpoint[2])
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = min(min_dist, dist)
        return min_dist

    def read_results(self, sort_by, data_pkl=None):
        from eval.dataset_utils import gen_multiobject_dataset
        from eval.dataset_utils.object_nav_utils import object_nav_gen

        path = self.results_path
        state_dir = os.path.join(path, "state")
        state_results = {}
        extras_dir = os.path.join(path, "extras")

        # Check if the state directory exists
        if not os.path.isdir(state_dir):
            print(f"Error: {state_dir} is not a valid directory")
            return state_results
        pose_dir = os.path.join(
            os.path.abspath(os.path.join(state_dir, os.pardir)), "trajectories"
        )
        os.makedirs(os.path.join(path, "saved_maps_gt"), exist_ok=True)

        # load scene objects metadata
        self.load_object_metadata()

        # Iterate through all files in the state directory
        data = []
        data_per_attribute = []
        sum_successes = 0
        actual_labels = []
        predicted_labels = []
        if data_pkl is None:
            scene_data = {}
            episodes_json = {ep.episode_id: ep for ep in self.episodes}
            total_experiments_run = 0
            for filename in sorted(os.listdir(state_dir)):
                if filename.startswith("state_") and filename.endswith(".txt"):
                    try:
                        # Extract the experiment number from the filename
                        experiment_num = filename.split('state_')[-1].split('.txt')[0]
                        with open(os.path.join(state_dir, filename), "r") as file:
                            content = file.read().strip()
                        # load scene
                        scene_id = experiment_num.split('__')[0]
                        # if len(self.include_ids) > 0 and scene_id not in self.include_ids:
                        #     continue

                        # total_experiments_run += 1
                        if self.sim is None or not self.sim.curr_scene_name in scene_id:
                            self.load_scene(scene_id)
                            (
                                rgm,
                                scene_objects_headers,
                                scene_objects_metadata,
                                scene_objects_metadata_by_main_category,
                                scene_objects_metadata_by_wnsynsetkey,
                                scene_objects_metadata_by_name,
                            ) = self.get_all_scene_objects()

                        # Convert the content to a number (assuming it's a float)
                        state_values = content.split(",")
                        state_values = [int(val) for val in state_values]
                        # Store the result in the dictionary
                        # Create a row for each sequence in the experiment

                        for seq_num, value in enumerate(state_values):
                            if seq_num >= 3:
                                continue
                            object_goals = episodes_json[experiment_num].goals[seq_num]
                            language_properties = object_goals["extras"]["language_properties"]
                            attribute_types = language_properties["features"]
                            attributes = []
                            attribute_values = []
                            attribute_count = 0
                            for _obj in attribute_types:
                                _attr = list(_obj.values())[0]
                                if len(_attr) > 0:
                                    attribute_types_present = {a_name:a_value['explanation'] for a_name, a_value in _attr.items() if a_value["exists"]}
                                    for attribute, attribute_vals in attribute_types_present.items():
                                        attributes.append({attribute: attribute_vals})
                                        attribute_count += len(attribute_vals)

                            if "on " in  object_goals['language_instruction']:
                                attribute_values.append('on')
                                attributes.append({"support": ["on"]})
                                attribute_count += 1
                                
                            if "spatial_rel_type" in object_goals and len(object_goals["spatial_rel_type"]) > 0:
                                attribute_values.append(object_goals["spatial_rel_type"])
                                attributes.append({object_goals["spatial_rel_type"]: [object_goals["spatial_rel"]]})
                                attribute_count += 1

                            total_experiments_run += 1

                            ppl = 0
                            map_size = 0
                            try:
                                poses = np.genfromtxt(
                                        os.path.join(
                                            pose_dir,
                                            "poses_"
                                            + str(experiment_num)
                                            + "_"
                                            + str(seq_num)
                                            + ".csv",
                                        ),
                                        delimiter=",",
                                    )
                            except:
                                continue
                            if len(poses.shape) == 1:
                                poses = poses.reshape((1, 4))
                            path_length = np.linalg.norm(
                                poses[1:, :3] - poses[:-1, :3], axis=1
                            ).sum()
                            _g = episodes_json[experiment_num].goals[seq_num]
                            # compute the optimal path length
                            if episodes_json[experiment_num].shortest_dists is None or len(episodes_json[experiment_num].shortest_dists) == 0:
                                episodes_json[experiment_num].shortest_dists = [{"nearest_pos": [],"dist": 0}]
                            else:
                                episodes_json[experiment_num].shortest_dists.append({"nearest_pos": [],"dist": 0})
                            if seq_num == 0:
                                start_pos = episodes_json[experiment_num].start_position
                            else:
                                start_pos = episodes_json[experiment_num].shortest_dists[seq_num-1]["nearest_pos"]
                            
                            nearest_nav_points = []
                            shortest_dists = []
                            for _obj in _g["goal_object"]:
                                for p in _obj['navigable_points']:
                                    nearest_nav_points.append([float(p[0]), float(p[1]), float(p[2])])
                                    shortest_path = habitat_sim.nav.ShortestPath()
                                    shortest_path.requested_start = start_pos
                                    shortest_path.requested_end = nearest_nav_points[-1]
                                    self.sim.pathfinder.find_path(shortest_path)
                                    shortest_dists.append(shortest_path.geodesic_distance)
                            shortest_dists_index = np.argmin(np.array(shortest_dists))
                            best_dist = episodes_json[experiment_num].shortest_dists[seq_num]["dist"] = shortest_dists[shortest_dists_index]
                            episodes_json[experiment_num].shortest_dists[seq_num]["nearest_pos"] = nearest_nav_points[shortest_dists_index]
                            if value == 1:
                                sum_successes += 1
                                ppl = min(
                                    1.0, 1 * (best_dist / max(path_length, best_dist))
                                )
                            goal_query = object_goals['language_instruction']

                            ## distance to goal
                            min_dist = float('inf')
                            agent_last_pos = [-poses[-1][1], poses[-1][2], -poses[-1][0]]
                            object_sizes = []
                            for _obj in _g["goal_object"]:
                                for p in _obj['navigable_points']:
                                    # compute min dist to nearest goal from last pose
                                    shortest_path = habitat_sim.nav.ShortestPath()
                                    shortest_path.requested_start = agent_last_pos
                                    shortest_path.requested_end = [float(p[0]), float(p[1]), float(p[2])]
                                    self.sim.pathfinder.find_path(shortest_path)
                                    min_dist = min(min_dist, shortest_path.geodesic_distance)
                                if 'sizes' in _obj and len(_obj['sizes']) > 0:
                                    object_sizes.append(np.prod(_obj['sizes']))

                            data.append(
                                {
                                    "experiment": experiment_num,
                                    "sequence": seq_num,
                                    "state": value,
                                    "num_of_attributes": attribute_count,
                                    "ppl": ppl,
                                    "map_size": map_size,
                                    "path_length": path_length,
                                    "num_steps": len(poses),
                                    'object': goal_query,
                                    "scene": episodes_json[experiment_num].scene_id,
                                    'granularity': object_goals["granularity"] if "granularity" in object_goals else "",
                                    'support_relation': "on " in object_goals['language_instruction'],
                                    'distance_to_goal': min_dist,
                                    'goal_size': np.round(np.mean(np.array(object_sizes)), 2),
                                    'exploit_enabled': len(poses) > self.max_explore_steps
                                }
                            )

                            if len(attributes) == 0:
                                data_per_attribute.append({
                                    "experiment": experiment_num,
                                    "sequence": seq_num,
                                    "state": value,
                                    "attribute": "",
                                    "attribute_value": "",
                                    "num_of_attributes": 0,
                                    "ppl": ppl,
                                    "map_size": map_size,
                                    "path_length": path_length,
                                    "num_steps": len(poses),
                                    'object': goal_query,
                                    "scene": episodes_json[experiment_num].scene_id,
                                    'granularity': object_goals["granularity"] if "granularity" in object_goals else "",
                                    'support_relation': "on " in object_goals['language_instruction'],
                                    'distance_to_goal': min_dist,
                                    # 'goal_visible_frames_ratio': count_visible_values_true/total_visible_values,
                                    'goal_size': np.round(np.mean(np.array(object_sizes)), 2),
                                    # 'pred_label_similarity': np.round(similarity, 2),
                                    'exploit_enabled': len(poses) > self.max_explore_steps
                                })
                            else:
                                all_attr_names = [list(a.keys())[0] for a in attributes]
                                for _attribute in attributes:
                                    _attribute_name, _attribute_value = list(_attribute.items())[0]
                                    for _val in _attribute_value:
                                        data_per_attribute.append({
                                            "experiment": experiment_num,
                                            "sequence": seq_num,
                                            "state": value,
                                            "attribute": _attribute_name,
                                            "attribute_value": _val,
                                            "num_of_attributes": all_attr_names.count(_attribute_name),
                                            "ppl": ppl,
                                            "map_size": map_size,
                                            "path_length": path_length,
                                            "num_steps": len(poses),
                                            'object': goal_query,
                                            "scene": episodes_json[experiment_num].scene_id,
                                            'granularity': object_goals["granularity"] if "granularity" in object_goals else "",
                                            'support_relation': "on " in object_goals['language_instruction'],
                                            'distance_to_goal': min_dist,
                                            # 'goal_visible_frames_ratio': count_visible_values_true/total_visible_values,
                                            'goal_size': np.round(np.mean(np.array(object_sizes)), 2),
                                            # 'pred_label_similarity': np.round(similarity, 2),
                                            'exploit_enabled': len(poses) > self.max_explore_steps
                                        })

                        if episodes_json[experiment_num].episode_id != experiment_num:
                            print(
                                f"Warning, experiment_num {experiment_num} does not correctly resolve to episode_id {episodes_json[experiment_num].episode_id}"
                            )
                    except ValueError:
                        print(f"Warning: Skipping {filename} due to invalid format")
            data = pd.DataFrame(data)
            data_per_attribute = pd.DataFrame(data_per_attribute)
        else:
            with open(data_pkl, "rb") as f:
                data = pickle.load(f)
        states = sorted([r.value for r in Result])

        total_episodes = total_experiments_run
        print(f"\nTotal experiments: {total_episodes}. Successful experiments: {sum_successes}. Failed experiments: {total_episodes-sum_successes}.")

        # print(sum_successes/236)
        def has_success(group, seq_id):
            return (
                group[(group["sequence"] == seq_id) & (group["state"] == 1)].shape[0]
                > 0
            )

        def calc_prog_per_episode(group, num_seq):
            successes = group.groupby("experiment").apply(
                lambda x: (x["state"] == 1).sum()
            )
            progress = successes / num_seq
            return progress

        def calc_ppl_per_episode(group):
            spls_per_exp = group.groupby("experiment")["ppl"].mean()
            return spls_per_exp

        def calculate_percentages(group):
            total = len(group)
            result = pd.Series(
                {
                    Result(state).name: (group["state"] == state).sum() / total
                    for state in states
                }
            )
            progress = calc_prog_per_episode(group, total)
            ppl = calc_ppl_per_episode(group)
            s = progress[progress == 1]
            result["success_count"] = progress[progress > 0].count()
            result["Progress"] = progress.mean()
            result["PPL"] = ppl.mean()
            # result["opt_PL"] = group["opt_path"].mean()
            result["success"] = s.sum() / len(progress)
            result["SPL"] = ppl[progress == 1].sum() / len(progress)
            result["Path Length"] = group["path_length"].mean()
            result["episodes"] = ','.join(group['experiment'].unique())
            result["distance_to_goal"] = group[group['distance_to_goal']!=float('inf')]['distance_to_goal'].mean()
            # result['goal_visible_frames_ratio'] = group['goal_visible_frames_ratio'].mean()
            result['goal_size'] = group['goal_size'].mean()
            # result['pred_label_similarity'] = group['pred_label_similarity'].mean()
            result['support_relation'] = group['support_relation'].unique()
            result['exploit_enabled'] = group['exploit_enabled'].unique()
            result['num_steps'] = group["num_steps"].mean()
            result['num_of_attributes'] = group["num_of_attributes"].mean()

            # Calculate average SPL and multiply by 100
            # avg_spl = group['spl'].mean()
            # result['Average SPL'] = avg_spl

            return result

        def calculate_overall_percentages(group):
            result = pd.Series(
                {
                    Result(state).name: (group["state"] == state).sum() / total_episodes
                    for state in states
                }
            )
            progress = calc_prog_per_episode(group, 3)
            ppl = calc_ppl_per_episode(group)
            s = progress[progress == 1]
            result["Progress"] = progress.mean()
            result["PPL"] = ppl.mean()
            # result["opt_PL"] = group["opt_path"].mean()
            result["success"] = s.sum() / total_episodes
            result["SPL"] = ppl[progress == 1].sum() / total_episodes
            result["Path Length"] = group["path_length"].mean()
            result['num_steps'] = group["num_steps"].mean()
            result['num_steps_success'] = group[group['state']==1]["num_steps"].mean()
            result['num_steps_failure'] = group[group['state']!=1]["num_steps"].mean()
            result["dist_to_goal_for_successful_ep"] = group[(group['state']==1) & (group['distance_to_goal']!=float('inf'))]['distance_to_goal'].mean()
            # result['goal_visible_frames_ratio_all_failures'] = group[(group['state']!=1)]['goal_visible_frames_ratio'].mean()
            # result['goal_visible_frames_ratio_wrong_det'] = group[(group['state']==Result.FAILURE_MISDETECT.value)]['goal_visible_frames_ratio'].mean()
            result['goal_size_wrong_det'] = group[(group['state']==Result.FAILURE_MISDETECT.value)]['goal_size'].mean()
            # result['pred_label_sim_wrong_det'] = group[(group['state']==Result.FAILURE_MISDETECT.value)]['pred_label_similarity'].mean()
            result['support_relation_wrong_det'] = group[(group['state']==Result.FAILURE_MISDETECT.value)]['support_relation'].unique()
            result['exploit_enabled_wrong_det'] = group[(group['state']==Result.FAILURE_MISDETECT.value)]['exploit_enabled'].unique()

            return result

        # Function to format percentages
        def format_percentages(val):
            return f"{val:.2%}" if isinstance(val, float) else val

        # Per-object results
        object_results = (
            data.groupby("object").apply(calculate_percentages).reset_index()
        )
        object_results = object_results.rename(columns={"object": "Object"})

        ## Failure analysis
        object_results_failure = object_results[
            [
                "Object",
                "episodes",
                "NO_FAILURE",
                "FAILURE_NOT_REACHED",
                "FAILURE_MISDETECT",
                "FAILURE_STUCK",
                "FAILURE_ALL_EXPLORED",
                "FAILURE_OOT",
            ]
        ]
        object_results_failure = object_results_failure.sort_values(
            by=[
                "FAILURE_ALL_EXPLORED",
                "FAILURE_MISDETECT",
                "FAILURE_STUCK",
                "FAILURE_NOT_REACHED",
                "FAILURE_ALL_EXPLORED",
                "FAILURE_OOT",
            ],
            ascending=False,
        )

        # Per-granularity results
        granularity_results = data.groupby("granularity").apply(calculate_percentages).reset_index()
        granularity_results["count"] = pd.Series(data.groupby("granularity")["experiment"].agg(["count"]).reset_index()["count"])
        granularity_results = granularity_results.sort_values(by=sort_by, ascending=False)

        # per-attribute results
        attribute_results = data_per_attribute.groupby("attribute").apply(calculate_percentages).reset_index()
        attribute_results["count"] = pd.Series(data_per_attribute.groupby("attribute")["experiment"].agg(["count"]).reset_index()["count"])
        attribute_results = attribute_results.sort_values(by=sort_by, ascending=False)

        # Per-scene results
        scene_results = data.groupby("scene").apply(calculate_percentages).reset_index()
        scene_results = scene_results.rename(columns={"scene": "Scene"})

        # Overall results
        overall_percentages = calculate_percentages(data)

        # Sorting
        object_results = object_results.sort_values(by=sort_by, ascending=False)
        overall_row = pd.DataFrame(
            [{"Object": "Overall"} | overall_percentages.to_dict()]
        )
        object_results = pd.concat([overall_row, object_results], ignore_index=True)

        scene_results = scene_results.sort_values(by=sort_by, ascending=False)
        overall_row = pd.DataFrame(
            [{"Scene": "Overall"} | overall_percentages.to_dict()]
        )
        scene_results = pd.concat([overall_row, scene_results], ignore_index=True)

        # Apply formatting to all columns except the first one (Object/Scene)
        object_table = (
            object_results.iloc[:, 0]
            .to_frame()
            .join(object_results.iloc[:, 1:].applymap(format_percentages))
        )
        scene_table = (
            scene_results.iloc[:, 0]
            .to_frame()
            .join(scene_results.iloc[:, 1:].applymap(format_percentages))
        )

        object_results_failure = object_results_failure.rename(columns={"FAILURE_OOT": "FAILURE_OTHER"})
        object_failure_table = (
                    object_results_failure.iloc[:, 0]
                    .to_frame()
                    .join(object_results_failure.iloc[:, 1:].applymap(format_percentages))
                )
        overall_episode_percentages = calculate_overall_percentages(data)
        overall_episode_row = pd.DataFrame(
            [{"Object": "Overall"} | overall_episode_percentages.to_dict()]
        )
        print(f"Failure analysis (Overall):")
        print(tabulate(overall_episode_row, headers="keys", tablefmt="pretty", floatfmt=".2%",showindex=False))
        print(f"Failure analysis by Object:")
        print(tabulate(object_failure_table, headers="keys", tablefmt="pretty", floatfmt=".2%",showindex=False))

        granularity_table = (
            granularity_results.iloc[:, 0]
            .to_frame()
            .join(granularity_results.iloc[:, 1:].applymap(format_percentages))
        )
        print(f"Results by Granularity:")
        print(tabulate(granularity_table, headers="keys", tablefmt="pretty", floatfmt=".2%"))

        attribute_table = (
            attribute_results.iloc[:, 0]
            .to_frame()
            .join(attribute_results.iloc[:, 1:]) #.applymap(format_percentages))
        )
        print(f"Results by Attribute:")
        print(tabulate(attribute_table, headers="keys", tablefmt="pretty", floatfmt=".2%"))

        print(f"Results by Object (sorted by {sort_by} rate, descending):")
        print(tabulate(object_table, headers="keys", tablefmt="pretty", floatfmt=".2%"))

        print(f"\nResults by Scene (sorted by {sort_by} rate, descending):")
        print(tabulate(scene_table, headers="keys", tablefmt="pretty", floatfmt=".2%"))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        data_per_scene = data.groupby("scene")
        sr_per_scene = []
        ppl_per_scene = []
        all_successful_episode_ids = collections.defaultdict(list)
        for scene, scene_data in data_per_scene:
            print(f"\nScene: {scene}")
            success_rates = []
            ppl_values = []
            seq_numbers = []
            for i in range(self.num_seq):
                sequences = scene_data[scene_data["sequence"] == i]
                if len(sequences) > 0:
                    successful_experiments = sequences[sequences["state"] == 1]
                    all_successful_episode_ids[str(i)].extend(list(sequences[sequences["state"] == 1]["experiment"]))
                    ppl = sequences["ppl"].mean() * self.num_seq
                    success_rate = len(successful_experiments) / len(sequences)

                    success_rates.append(success_rate)
                    ppl_values.append(ppl)
                    seq_numbers.append(i)
                    print(f"  Sequence {i}:")
                    print(f"    Num of experiments: {len(sequences)}")
                    print(f"    Overall PPL: {ppl:.4f}")
                    print(f"    Fraction of successful experiments: {success_rate:.2%}")
                else:
                    print(f"  Sequence {i}: No data")
                    success_rates.append(0)
                    ppl_values.append(0)
            sr_per_scene.append(success_rates)
            ppl_per_scene.append(ppl_values)
        sr_per_scene = np.array(sr_per_scene)
        ppl_per_scene = np.array(ppl_per_scene)
        sr_per_scene = np.mean(sr_per_scene, axis=0)
        ppl_per_scene = np.mean(ppl_per_scene, axis=0)
        print(f"PPL: {ppl_per_scene}, Success Rate: {sr_per_scene}")

        with open(f"{self.results_path}/all_successful_episodes.json", "w") as f:
            json.dump(all_successful_episode_ids, f)

        episode_results = (
            data.groupby("experiment").apply(calculate_percentages).reset_index()
        )
        episode_results = episode_results.rename(columns={"object": "Object"})
        episode_table = (
            episode_results.iloc[:, 0]
            .to_frame()
            .join(episode_results.iloc[:, 1:].applymap(format_percentages))
        )
        print(f"Overall Results by episode:")
        print(tabulate(episode_table, headers="keys", tablefmt="pretty", floatfmt=".2%"))

        # Plot Success Rate
        ax1.plot(np.arange(self.num_seq), sr_per_scene, label=scene, marker="o")

        # Plot PPL
        ax2.plot(np.arange(self.num_seq), ppl_per_scene, label=scene, marker="o")
        # Set up Success Rate subplot
        ax1.set_xlabel("Sequence Number")
        ax1.set_ylabel("Success Rate")
        ax1.set_title("Success Rate per Sequence")
        # ax1.legend()
        ax1.grid(True)

        # Set up SPL subplot
        ax2.set_xlabel("Sequence Number")
        ax2.set_ylabel("PPL")
        ax2.set_title("PPL per Sequence")
        # ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("output_plot.png")

        plt.show()
        
        # plt.figure(figsize=(15, 10))
        # data_per_size = pd.DataFrame(data[data["state"]==1].groupby("goal_size"))
        # plt.hist(data_per_size.iloc[:,0], bins=[0.0,0.4,0.7,1.2,1.6], density=True)

        # # Set up Success Rate subplot
        # plt.xlabel("Object volumn")
        # plt.ylabel("Episodes succeeded")
        # plt.title("Successful episodes per goal object size")
        # plt.grid(True)

        # plt.tight_layout()
        # plt.savefig(os.path.join(self.results_path, "success_per_obj_size.png"))

        plt.figure(figsize=(15, 10))
        attribute_type_results = data[data["state"]!=1].groupby("num_of_attributes")["experiment"].agg(["count"]).reset_index()
        attribute_type_results["count"] = (attribute_type_results["count"]/total_episodes)*100.0
        attribute_type_results.plot(x='num_of_attributes', y='count', legend=False, kind='bar', rot=0)

        # Set up Success Rate subplot
        plt.xlabel("Number of attribute types")
        plt.ylabel("Success rate %")
        plt.title("Success rate varies with number of object attributes")

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, "success_per_attribute_type.png"))

        plt.figure(figsize=(15, 10))
        attributes_results = data_per_attribute[data_per_attribute["state"]!=1].groupby("attribute").agg({"experiment":"count"}).reset_index()
        attributes_results["experiment"] = (attributes_results["experiment"]/total_episodes)*100.0
        attributes_results = attributes_results.sort_values(by="experiment", ascending=False)
        attributes_results.plot(x='attribute', y='experiment', legend=False, kind='bar', rot=60)

        # Set up Success Rate subplot
        plt.xlabel("Number of attribute types")
        plt.ylabel("Success rate %")
        plt.title("Success rate varies with number of object attributes")

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, "success_per_attributes.png"))

        # plt.figure(figsize=(15, 10))
        # data_per_size = pd.DataFrame(data[data["state"]!=1].groupby("goal_size"))
        # plt.hist(data_per_size.iloc[:,0], bins=[0.0,0.5,1.0,1.6], density=True)

        # # Set up failure subplot
        # plt.xlabel("Object volumn")
        # plt.ylabel("Episodes failed")
        # plt.title("Failed episodes per goal object size")
        # plt.grid(True)

        # plt.tight_layout()
        # plt.savefig(os.path.join(self.results_path, "failure_per_obj_size.png"))

        # error_data = data[data["state"]==2]
        # if len(error_data) == 0:
        #     error_data = data[data["state"]==8]

        # plt.figure(figsize=(15, 10))
        # data_per_size = pd.DataFrame(error_data.groupby("goal_size"))
        # plt.hist(data_per_size.iloc[:,0], bins=[0.0,0.5,1.0,1.6], density=True)

        # # Set up failure subplot
        # plt.xlabel("Object volumn")
        # plt.ylabel("Episodes with wrong detection")
        # plt.title("Wrong detection failure cases per goal object size")
        # plt.grid(True)

        # plt.tight_layout()
        # plt.savefig(os.path.join(self.results_path, "failure_wrong_detect_per_obj_size.png"))

        # plt.figure(figsize=(15, 10))
        # data_per_size = pd.DataFrame(error_data.groupby("pred_label_similarity"))
        # plt.hist(data_per_size.iloc[:,0], bins=[0.0,0.5,0.8,1.1], density=True)

        # # Set up failure subplot
        # plt.xlabel("Similarity between detection and actual label")
        # plt.ylabel("Episodes with wrong detection")
        # plt.title("Wrong detection failure cases per prediction text similarity")
        # plt.grid(True)

        # plt.tight_layout()
        # plt.savefig(os.path.join(self.results_path, "failure_wrong_detect_per_pred_txt_sim.png"))

        # w_support_relation_total = len(data[data['support_relation']==True])
        # wo_support_relation_total = len(data[data['support_relation']==False])
        # plt.figure(figsize=(15, 10))
        # df = data[data["state"]==1].copy()
        # df = df[['support_relation','exploit_enabled']]
        # df = df.groupby(['support_relation','exploit_enabled'])['exploit_enabled'].apply(pd.value_counts, sort=False).reset_index(name='count')
        # df = df[['support_relation','exploit_enabled','count']]
        # df_pivot = df.pivot(index='support_relation',columns='exploit_enabled',values='count')
        # ax = df_pivot.plot.bar(stacked=True)
        # for p in ax.patches:
        #     ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        # # Set up failure subplot
        # plt.xlabel(f"Presence of support relation [True={w_support_relation_total}, False={wo_support_relation_total}]")
        # plt.ylabel("Successful episodes")
        # plt.title("Success for instructions with and without support relation")
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.results_path, "success_support_relation.png"))

        # plt.figure(figsize=(15, 10))
        # df = error_data.copy()
        # df = df[['support_relation','exploit_enabled']]
        # df = df.groupby(['support_relation','exploit_enabled'])['exploit_enabled'].apply(pd.value_counts, sort=False).reset_index(name='count')
        # df = df[['support_relation','exploit_enabled','count']]
        # df.pivot(index='support_relation',columns='exploit_enabled',values='count').plot.bar(stacked=True)
        # # Set up failure subplot
        # plt.xlabel(f"Presence of support relation [True={w_support_relation_total}, False={wo_support_relation_total}]")
        # plt.ylabel("Failed episodes with 'wrong detection'")
        # plt.title("Wrong detection failure cases for instructions with and without support relation")
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.results_path, "failure_wrong_detect_support_relation.png"))

        # fig, ax = plt.subplots(figsize=(20, 20))
        # all_classes = actual_labels + predicted_labels
        # cm = confusion_matrix(actual_labels, predicted_labels, labels=list(set(all_classes)), normalize='all')
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(all_classes)))
        # disp.plot(xticks_rotation='vertical',ax=ax)
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.results_path, "failure_wrong_detect_confusion_matrix.png"))

        return data

    def evaluate(self):
        n_eps = 0
        results = []
        pbar = tqdm.tqdm(total=len(self.episodes))
        for n_ep, episode in enumerate(self.episodes):
            poses = []
            map_poses_and_obs = []
            metric = Metrics(episode.episode_id)
            results.append(metric)
            if len(self.include_ids) > 0 and episode.episode_id not in self.include_ids:
                pbar.update()
                continue
            if episode.episode_id in self.exclude_ids:
                pbar.update()
                continue
            n_eps += 1
            if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                self.load_scene(episode.scene_id)

            self.sim.initialize_agent(
                0,
                habitat_sim.AgentState(episode.start_position, episode.start_rotation),
            )
            self.actor.reset()

            sequence_id = 0
            current_obj = episode.goals[sequence_id]
            if self.config.goal_query_type == "coarse":
                goal_query = "a " + " ".join(current_obj["object_category"].split('_'))
            elif self.config.goal_query_type == "fine":
                goal_query = "a " + " ".join(current_obj["extras"]["object_category"].split("_"))
            else:
                if self.config.goal_query_processing == "extract" or self.config.goal_query_processing == "extract_and_split_support":
                    goal_query = current_obj['language_instruction'].split('Find ')[-1].split('Go to ')[-1].split('.')[0]
                elif self.config.goal_query_processing == "extract_no_support":
                    goal_query = current_obj['language_instruction'].split('Find ')[-1].split('Go to ')[-1].split('.')[0].split(' on the ')[0]
                else:
                    goal_query = current_obj['language_instruction']

            if self.config.goal_query_processing == "extract_graph":
                full_query = goal_query
                text_graph = extract_graph_from_text(goal_query)
                text_graph_queries = text_graph["nodes"]
                if len(text_graph_queries) == 0:
                    text_graph_queries = full_query.split('Find ')[-1].split('Go to ')[-1].split('.')[0]
                elif len(text_graph["edges"]) > 0:
                    text_graph_queries.extend([e["relation"] for e in text_graph["edges"]])
                self.actor.set_queries(text_graph_queries, full_query)
            elif self.config.goal_query_processing == "mix":
                goal_query = current_obj['language_instruction']
                full_query = goal_query
                if "on " in goal_query:
                    goal_query = goal_query.split(' on ')[::-1]
                    self.actor.set_queries(goal_query, full_query)
                else:
                    self.actor.set_query(goal_query)
            elif self.config.goal_query_type == "detailed" and self.config.goal_query_processing == "extract_and_split_support" and " on " in goal_query:
                full_query = goal_query
                goal_query = goal_query.split(' on ')[::-1]
                self.actor.set_queries(goal_query, full_query)
            elif self.config.goal_query_type == "detailed" and self.config.goal_query_processing == "extract_and_split_support" and "above " in goal_query:
                full_query = goal_query
                goal_query = goal_query.split(' above ')[::-1]
                self.actor.set_queries(goal_query, full_query)
            elif self.config.goal_query_type == "detailed" and self.config.goal_query_processing == "extract_and_split_support" and "below " in goal_query:
                full_query = goal_query
                goal_query = goal_query.split(' below ')
                self.actor.set_queries(goal_query, full_query)
            # elif self.config.goal_query_type == "detailed" and self.config.goal_query_processing == "extract_and_split_support" and "next to " in goal_query:
            #     full_query = goal_query
            #     goal_query = goal_query.split(' next to ')
            #     # form kernel
            #     goal_query_mod = [goal_query[1], goal_query[1], goal_query[1],
            #                       goal_query[1], goal_query[0], goal_query[1],
            #                       goal_query[1], goal_query[1], goal_query[1]]
            #     self.actor.set_queries(goal_query_mod, full_query)
            else:
                self.actor.set_query(goal_query)
            
            rgb_frames = []
            is_object_in_frame = {}
            all_object_detections = {}
            while sequence_id < len(episode.goals):
                steps = 0
                not_failed = True
                running = True
                running_exploit = False
                map_poses_and_obs = []
                while steps < self.max_steps and running:
                    try:
                        observations = self.sim.get_sensor_observations()
                        # observations['depth'] = fill_depth_holes(observations['depth'])
                        observations["state"] = self.sim.get_agent(0).get_state()
                        pose = np.zeros((4,))
                        pose[0] = -observations["state"].position[2]
                        pose[1] = -observations["state"].position[0]
                        pose[2] = observations["state"].position[1]
                        # yaw
                        orientation = observations["state"].rotation
                        q0 = orientation.x
                        q1 = orientation.y
                        q2 = orientation.z
                        q3 = orientation.w
                        r = R.from_quat([q0, q1, q2, q3])
                        # r to euler
                        yaw, _, _1 = r.as_euler("yxz")
                        pose[3] = yaw

                        poses.append(pose)
                        if self.save_maps:
                            map_poses_and_obs.append(
                                {
                                    "pose_xyzyaw": pose,
                                    "pose_map": self.actor.mapper.one_map.metric_to_px(
                                        pose[0], pose[1]
                                    ),
                                    "obs_from_pose": observations,
                                }
                            )
                        if self.log_rerun:
                            rr.log("logs", rr.TextLog(f"\"{current_obj['language_instruction']}\""))
                            cam_x = -self.sim.get_agent(0).get_state().position[2]
                            cam_y = -self.sim.get_agent(0).get_state().position[0]
                            rr.log(
                                "camera/rgb",
                                rr.Image(observations["rgb"]).compress(jpeg_quality=50),
                            )
                            rr.log(
                                "camera/depth",
                                rr.Image(
                                    (observations["depth"] - observations["depth"].min())
                                    / (
                                        observations["depth"].max()
                                        - observations["depth"].min()
                                    )
                                ),
                            )
                            self.logger.log_pos(cam_x, cam_y)

                        action, called_found = self.actor.act(observations)
                        self.execute_action(action)

                        ## logging goal seen
                        object_centroids = [[-g['centroid'][2], -g['centroid'][0], g['centroid'][1]] for g in current_obj['goal_object']]
                        object_visible = within_fov_cone(
                            cone_origin=pose[:3],
                            cone_angle=pose[3],
                            cone_fov=np.deg2rad(self.hfov),
                            cone_range=2.5,
                            points=np.array(object_centroids),
                        )
                        if sequence_id not in is_object_in_frame:
                            is_object_in_frame[sequence_id] = []
                        if sequence_id not in all_object_detections:
                            all_object_detections[sequence_id] = []
                        is_object_in_frame[sequence_id].append(object_visible)
                        if (
                            self.actor.mapper.detections is not None
                            and len(self.actor.mapper.detections) > 0
                            and len(self.actor.mapper.detections["boxes"]) > 0
                        ):
                            if self.actor.mapper.chosen_detection is not None:
                                pos = self.actor.mapper.chosen_detection
                                pos_metric = self.actor.mapper.one_map.px_to_metric(
                                    pos[0], pos[1]
                                )
                            else:
                                pos = []
                                pos_metric = []
                            agent_pose_map = self.actor.mapper.one_map.metric_to_px(
                                -observations["state"].position[2], -observations["state"].position[0]
                            )
                            all_object_detections[sequence_id].append({
                                "detections": json.dumps(self.actor.mapper.detections),
                                "goal_pos_map": pos,
                                "goal_pos": pos_metric,
                                "agent_pose": pose,
                                "agent_pose_map": agent_pose_map
                            })

                        if self.log_rerun:
                            self.logger.log_map()
                            pts = []
                            viewpts = []
                            for obj in current_obj["goal_object"]:
                                pt = obj["centroid"]
                                pt = (-pt[2], -pt[0])
                                pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                                for v in obj['navigable_points']:
                                    vt = (-float(v[2]), -float(v[0]))
                                    viewpts.append(self.actor.mapper.one_map.metric_to_px(*vt))
                            pts = np.array(pts)
                            rr.log(
                                "map/ground_truth",
                                rr.Points2D(pts, colors=[[150, 5, 200]], radii=[2]),
                            )
                            rr.log(
                                "map/ground_truth_viewpoints",
                                rr.Points2D(viewpts, colors=[[150, 5, 170]], radii=[0.5]),
                            )
                        if self.save_video:
                            _pad = 5
                            pose_m = self.actor.mapper.one_map.metric_to_px(
                                -observations["state"].position[2], -observations["state"].position[0]
                            )
                            similarities = self.actor.mapper.get_map()
                            sim_map_layers = []
                            final_sim = None
                            if len(similarities.shape) > 3 and similarities.shape[-1] > 1:
                                similarities = similarities[0]
                                num_layers = similarities.shape[-1]
                                _padm = 250
                                for i in range(1, num_layers):
                                    sim_map = similarities[:,:,i]
                                    sim_map[sim_map < 0] = 0.0
                                    new_img = np.zeros_like(sim_map)
                                    non_zero_ind = np.transpose(np.nonzero(sim_map))
                                    if len(non_zero_ind) > 0:
                                        # Get the bounding box of non-zero pixels
                                        x, y, w, h = np.min(non_zero_ind[:,1]), np.min(non_zero_ind[:,0]), np.max(non_zero_ind[:,1]), np.max(non_zero_ind[:,0])
                                        # Add some margin (optional)
                                        margin = 10
                                        x = max(0, x - margin)
                                        y = max(0, y - margin)
                                        w = min(sim_map.shape[1] - x, w + 2 * margin)
                                        h = min(sim_map.shape[0] - y, h + 2 * margin)
                                        cropped_image = sim_map[y:y+h, x:x+w]
                                        mid = new_img.shape[0] // 2
                                        start_h = mid - (sim_map.shape[0]//2)
                                        start_w = mid - (sim_map.shape[1]//2)
                                        new_img[start_h: start_h + sim_map.shape[0], start_w: start_w + sim_map.shape[1]] = sim_map
                                    sim_map_img = np.flip(monochannel_to_inferno_rgb(np.flip(new_img,axis=-1).transpose((1, 0))),axis=-1)
                                    x,y,z = np.where(sim_map_img==0)  # re-color background to white
                                    sim_map_img[x,y] = [255,255,255]
                                    # cv2.imwrite('test.jpg', sim_map_img)
                                    sim_map_layers.append(sim_map_img)
                            else:
                                similarities = (similarities + 1.0) / 2.0
                                final_sim = np.flip(monochannel_to_inferno_rgb(np.flip(similarities[0],axis=-1).transpose((1, 0))),axis=-1)

                            traversable_map = self.actor.mapper.one_map.navigable_map.astype(np.float32)
                            traversable_map[pose_m[0]-_pad:pose_m[0]+_pad, pose_m[1]-_pad:pose_m[1]+_pad] = 0.5

                            # show GT goals
                            # for _goal in current_obj["goal_object"]:
                            #     bbox = _goal["aabb"]
                            #     center = _goal["centroid"]
                            #     center_m = self.actor.mapper.one_map.metric_to_px(
                            #                 -center[2], -center[0]
                            #             )
                            #     traversable_map[min(center_m[0], traversable_map.shape[0]-1), min(center_m[1], traversable_map.shape[1]-1)] = 0.7
                            #     for p in _goal["navigable_points"]:
                            #         pm = self.actor.mapper.one_map.metric_to_px(
                            #                 -float(p[2]), -float(p[0])
                            #             )
                            #         traversable_map[min(pm[0], traversable_map.shape[0]-1), min(pm[1], traversable_map.shape[1]-1)] = 0.7

                            # mark selected goal
                            # goal_pos = self.actor.mapper.chosen_detection
                            # if goal_pos is not None:
                            #     traversable_map[goal_pos[0]-_pad:goal_pos[0]+_pad, goal_pos[1]-_pad:goal_pos[1]+_pad] = 0.8
                            # else:
                            #     # mark frontier goals
                            #     pts = np.array([nav_goal.get_descr_point() for nav_goal in self.actor.mapper.nav_goals])
                            #     if len(pts) > 0:
                            #         pt = pts[0]
                            #         traversable_map[pt[0]-_pad:pt[0]+_pad, pt[1]-_pad:pt[1]+_pad] = 0.8

                            # # mark path to the goal
                            # if self.actor.mapper.path is not None and len(self.actor.mapper.path) > 0:
                            #     for pth in self.actor.mapper.path:
                            #         try:
                            #             traversable_map[pth[0], pth[1]] = 0.3
                            #         except:
                            #             pass

                            traversable_map = np.flip(traversable_map,axis=-1).transpose((1, 0))
                            traversable_map = 1 - traversable_map
                            non_zero_pixels = cv2.findNonZero(traversable_map)
                            # Get the bounding box of non-zero pixels
                            x, y, w, h = cv2.boundingRect(non_zero_pixels)
                            # Add some margin (optional)
                            margin = 10
                            x = max(0, x - margin)
                            y = max(0, y - margin)
                            w = min(traversable_map.shape[1] - x, w + 2 * margin)
                            h = min(traversable_map.shape[0] - y, h + 2 * margin)

                            # Crop the image
                            cropped_image = traversable_map[y:y+h, x:x+w]
                            cropped_image = 1-cropped_image
                            traversable_map = np.flip(monochannel_to_gray(cropped_image),axis=-1)
                            x,y,z = np.where(traversable_map==127)  # re-color agent
                            traversable_map[x,y] = [250,0,20]

                            # obstcl_map_layers = []
                            # if self.actor.mapper.one_map.layered:
                            #     obstacles_layered = self.actor.mapper.one_map.obstacle_map_layered.cpu().numpy()
                            #     if len(obstacles_layered.shape) > 2 and obstacles_layered.shape[-1] > 1:
                            #         num_layers = obstacles_layered.shape[-1]
                            #         for i in range(num_layers):
                            #             obstcl_map_img = np.flip(monochannel_to_inferno_rgb(np.flip(obstacles_layered[:,:,i],axis=-1).transpose((1, 0))),axis=-1)
                            #             obstcl_map_layers.append(obstcl_map_img)

                            dist = self.get_closest_dist(
                                self.sim.get_agent(0).get_state().position,
                                current_obj["goal_object"],
                            )
                            text_to_append = f"Goal#{sequence_id+1} '{current_obj['language_instruction']}'\nStep count: {steps}. Distance to closest viewpoint:{np.round(dist,2)} m."
                            is_success = False
                            if called_found and dist < self.max_dist:
                                is_success = True
                            img_frame = add_sim_maps_to_image_paper(
                                observation=observations,
                                maps={
                                    "sim_map": final_sim,
                                    "traversable_map": traversable_map,
                                    "object_detected": self.actor.mapper.object_detected,
                                    "predictions": self.actor.mapper.detections,
                                    "sim_map_layers": sim_map_layers,
                                    "called_found": called_found,
                                    "is_success": is_success,
                                    "ran_out_of_time": steps > self.max_steps,
                                    # "obstcl_map_layers": obstcl_map_layers if self.actor.mapper.one_map.layered else None,
                                },
                                text_to_append=text_to_append,
                            )
                            # if steps > self.max_explore_steps and not called_found:
                            #     rect_color = (0,0,255)
                            #     start_y, end_y = 0, img_frame.shape[0]
                            #     start_x, end_x = 0, img_frame.shape[1]
                            #     img_frame = cv2.rectangle(img_frame, (start_x, start_y), (end_x, end_y), rect_color, thickness=10)
                            #     font_size = 1.0
                            #     font_thickness = 2
                            #     font = cv2.FONT_HERSHEY_SIMPLEX
                            #     cv2.putText(
                            #         img_frame,
                            #         "exploit-only mode activated",
                            #         (310, 900),
                            #         font,
                            #         font_size,
                            #         (0, 0, 0),
                            #         font_thickness,
                            #         lineType=cv2.LINE_AA,
                            #     )

                            rgb_frames.append(img_frame)
                            if called_found or steps >= self.max_steps:
                                font_size = 2.0
                                font_thickness = 2
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_color = (255,0,0)
                                if is_success:
                                    font_color = (0,255,0)
                                cv2.putText(
                                    img_frame,
                                    "Success" if is_success else "Failure",
                                    (900, 700),
                                    font,
                                    font_size,
                                    font_color,
                                    font_thickness,
                                    lineType=cv2.LINE_AA,
                                )
                                for _ in range(10):
                                    rgb_frames.append(img_frame)

                        if steps % 100 == 0:
                            dist = self.get_closest_dist(
                                self.sim.get_agent(0).get_state().position,
                                current_obj["goal_object"],
                            )
                            if dist == float("inf"):
                                with open(
                                    f"{self.results_path}/exceptions/{episode.episode_id}_dist_inf.txt", "w"
                                ) as f:
                                    f.write(f"Step {steps}, current query: {goal_query}, episode_id: {episode.episode_id}, distance to closest viewpoint: {dist}")
                                break
                            print(
                                f"Step {steps}, current query: {goal_query}, episode_id: {episode.episode_id}, distance to closest viewpoint: {dist}"
                            )
                        steps += 1

                        if steps % self.max_explore_steps == 0 and not called_found:
                            # check every max_explore_steps number of steps
                            check_found, _ = self.actor.check_if_goal_found()
                            if check_found:
                                self.actor.set_exploit()
                                running_exploit = True

                        if called_found or steps >= self.max_steps:
                            running = False
                            result = Result.FAILURE_OOT
                            # reset the relation graph
                            self.actor.mapper.one_map.rel_graph.reset_graph()
                            self.actor.mapper.one_map.reset_checked_image_map()
                            # We will now compute the closest distance to the bounding box of the object
                            if called_found:
                                dist = self.get_closest_dist(
                                    self.sim.get_agent(0).get_state().position,
                                    current_obj["goal_object"],
                                )
                                if dist < self.max_dist:
                                    result = Result.NO_FAILURE
                                    print("Object found!")
                                elif self.actor.mapper.chosen_detection is not None:
                                    pos = self.actor.mapper.chosen_detection
                                    pos_metric = self.actor.mapper.one_map.px_to_metric(
                                        pos[0], pos[1]
                                    )
                                    dist_detect = self.get_closest_dist(
                                        [-pos_metric[1], self.sim.get_agent(0).get_state().position[1], -pos_metric[0]],
                                        current_obj["goal_object"],
                                    )
                                    if dist_detect < self.max_dist:
                                        result = Result.FAILURE_NOT_REACHED
                                    elif running_exploit:
                                        result = Result.FAILURE_MISDETECT_ON_MAP
                                    elif self.actor.mapper.use_detector:
                                        result = Result.FAILURE_MISDETECT
                                    not_failed = False
                                    print(
                                        f"Object not found! Dist {dist}, detect dist: {dist_detect}."
                                    )
                            else:
                                not_failed = False
                                if (
                                    result == Result.FAILURE_OOT
                                    and np.linalg.norm(poses[-1][:2] - poses[-10][:2])
                                    < 0.05
                                ):
                                    result = Result.FAILURE_STUCK
                                num_frontiers = len(self.actor.mapper.nav_goals)
                                if (
                                    result == Result.FAILURE_STUCK
                                    or result == Result.FAILURE_OOT
                                ) and num_frontiers == 0:
                                    result = Result.FAILURE_ALL_EXPLORED
                            results[-1].add_sequence(np.array(poses), result, current_obj)

                            if self.save_maps:
                                # final_sim = (self.actor.mapper.get_map() + 1.0) / 2.0
                                confs = (
                                    (self.actor.mapper.one_map.confidence_map > 0)
                                    .cpu()
                                    .squeeze()
                                    .numpy()
                                )
                                # nav_map = self.actor.mapper.one_map.navigable_map.astype(bool)
                                feature_map = (
                                    self.actor.mapper.one_map.feature_map.cpu()
                                    .numpy()
                                    .astype(float)
                                )
                                # final_sim = final_sim[0]
                                # final_sim = monochannel_to_inferno_rgb(final_sim)

                                # final_sim[~confs, :] = [0, 0, 0]
                                # # final_sim[(~nav_map) & confs, :] = [0, 0, 0]
                                # min_x = np.min(np.where(confs)[0])
                                # max_x = np.max(np.where(confs)[0])
                                # min_y = np.min(np.where(confs)[1])
                                # max_y = np.max(np.where(confs)[1])
                                # final_sim = final_sim[min_x:max_x, min_y:max_y]
                                # final_sim = final_sim.transpose((1, 0, 2))
                                # final_sim = np.flip(
                                #     final_sim, axis=0
                                # )  # get min and max x and y of confs

                                # gt_goal_objects = []
                                # for _goal in current_obj["goal_object"]:
                                #     bbox = _goal["aabb"]
                                #     center = _goal["centroid"]
                                #     nav_points_map = [
                                #         self.actor.mapper.one_map.metric_to_px(
                                #             -float(p[2]), -float(p[0])
                                #         )
                                #         for p in _goal["navigable_points"]
                                #     ]
                                #     gt_goal_objects.append(
                                #         {
                                #             "object_category": current_obj["object_category"],
                                #             "instruction": current_obj["language_instruction"],
                                #             "object_extras": current_obj["extras"],
                                #             "center": center,
                                #             "aabb": bbox,
                                #             "center_map": self.actor.mapper.one_map.metric_to_px(
                                #                 -center[2], -center[0]
                                #             ),
                                #             "nav_points_map": nav_points_map,
                                #         }
                                #     )
                                np.savez_compressed(
                                    f"{self.results_path}/saved_maps/{episode.episode_id}_{sequence_id}.npz",
                                    # nav_map=nav_map,
                                    feature_map=feature_map,
                                    confidence_map=confs,
                                    query=goal_query,
                                    # final_sim_img=final_sim,
                                    pose_observations=map_poses_and_obs,
                                    # gt_goal_objects=gt_goal_objects,
                                )

                                # cv2.imwrite(
                                #     f"{self.results_path}/similarities/final_sim_{episode.episode_id}_{sequence_id}.png",
                                #     final_sim,
                                # )
                                # # Create the plot
                                # plt.figure(figsize=(10, 10))
                                # poses_ = np.array(
                                #     [
                                #         self.actor.mapper.one_map.metric_to_px(*pos[:2])
                                #         for pos in poses
                                #     ]
                                # )
                                # poses_[:, 0] -= min_x
                                # poses_[:, 1] -= min_y
                                # plt.imshow(
                                #     final_sim[:, :, ::-1],
                                #     interpolation="nearest",
                                #     aspect="equal",
                                #     extent=(0, final_sim.shape[1], 0, final_sim.shape[0]),
                                # )

                                # plt.plot(
                                #     poses_[:, 0], poses_[:, 1], "b-o"
                                # )  # 'b-o' means blue line with circle markers

                                # # Set equal aspect ratio to ensure accurate positions
                                # plt.axis("equal")

                                # # Add labels and title
                                # plt.xlabel("X position")
                                # plt.ylabel("Y position")
                                # plt.title("Path of Poses")

                                # # Add grid for better readability
                                # plt.grid(True)

                                # # Save the plot as SVG
                                # plt.savefig(
                                #     f"{self.results_path}/similarities/path_{episode.episode_id}_{sequence_id}.svg",
                                #     format="svg",
                                #     dpi=300,
                                #     bbox_inches="tight",
                                # )

                                # # Display the plot (optional, comment out if not needed)
                                # # plt.show()

                            sequence_id += 1
                            if sequence_id < len(episode.goals):
                                current_obj = episode.goals[sequence_id]
                                if self.config.goal_query_type == "coarse":
                                    goal_query = "a " + " ".join(current_obj["object_category"].split('_'))
                                elif self.config.goal_query_type == "fine":
                                    goal_query = "a " + " ".join(current_obj["extras"]["object_category"].split("_"))
                                else:
                                    if self.config.goal_query_processing == "extract" or self.config.goal_query_processing == "extract_and_split_support":
                                        goal_query = current_obj['language_instruction'].split('Find ')[-1].split('Go to ')[-1].split('.')[0]
                                    elif self.config.goal_query_processing == "extract_no_support":
                                        goal_query = current_obj['language_instruction'].split('Find ')[-1].split('Go to ')[-1].split('.')[0].split(' on the ')[0]
                                    else:
                                        goal_query = current_obj['language_instruction']

                                if self.config.goal_query_processing == "extract_graph":
                                    full_query = goal_query
                                    text_graph = extract_graph_from_text(goal_query)
                                    text_graph_queries = text_graph["nodes"]
                                    if len(text_graph_queries) == 0:
                                        text_graph_queries = full_query.split('Find ')[-1].split('Go to ')[-1].split('.')[0]
                                    elif len(text_graph["edges"]) > 0:
                                        text_graph_queries.extend([e["relation"] for e in text_graph["edges"]])
                                    self.actor.set_queries(text_graph_queries, full_query)
                                elif self.config.goal_query_processing == "mix":
                                    goal_query = current_obj['language_instruction']
                                    full_query = goal_query
                                    if "on " in goal_query:
                                        goal_query = goal_query.split(' on ')[::-1]
                                        self.actor.set_queries(goal_query, full_query)
                                    else:
                                        self.actor.set_query(goal_query)
                                elif self.config.goal_query_type == "detailed" and self.config.goal_query_processing == "extract_and_split_support" and "on " in goal_query:
                                    full_query = goal_query
                                    goal_query = goal_query.split(' on ')[::-1]
                                    self.actor.set_queries(goal_query, full_query)
                                else:
                                    self.actor.set_query(goal_query)

                    except Exception as e:
                        print(str(e))
                        not_failed = False
                        running = False
                        result = Result.FAILURE_EXCEPTION
                        results[-1].add_sequence(np.array(poses), result, current_obj)
                        with open(
                            f"{self.results_path}/exceptions/{episode.episode_id}.txt", "w"
                        ) as f:
                            f.write(str(e))
                        break

            for seq_id, seq in enumerate(results[n_ep].sequence_poses):
                np.savetxt(
                    f"{self.results_path}/trajectories/poses_{episode.episode_id}_{seq_id}.csv",
                    seq,
                    delimiter=",",
                )
                # write goal object visible in frame
                # if len(is_object_in_frame) > 0:
                #     with open(
                #         f"{self.results_path}/extras/goal_visible_{episode.episode_id}_{seq_id}.txt", "w"
                #     ) as f:
                #         f.write(
                #             ",".join(
                #                 str(_vis)
                #                 for _vis in is_object_in_frame[seq_id])
                #             )
                    
                # write object detection details
                # if len(all_object_detections) > 0:
                #     np.save(
                #         f"{self.results_path}/extras/detections_{episode.episode_id}_{seq_id}",
                #         all_object_detections[seq_id],
                #     )

            # save final sim to image file
            if self.save_video:
                generate_video(video_dir=f"{self.results_path}/videos", video_name=f"{episode.episode_id}__{result.name}", images=rgb_frames)
                rgb_frames = []

            print(
                f"Overall progress: {sum([m.get_progress(self.num_seq) for m in results]) / (n_eps)}, per object: "
            )
            # for obj in success_per_obj.keys():
            #     print(f"{obj}: {success_per_obj[obj] / obj_count[obj]}")
            # print(
            #     f"Result distribution: successes: {results.count(Result.NO_FAILURE)}, misdetects: {results.count(Result.FAILURE_MISDETECT)}, OOT: {results.count(Result.FAILURE_OOT)}, stuck: {results.count(Result.FAILURE_STUCK)}, not reached: {results.count(Result.FAILURE_NOT_REACHED)}, all explored: {results.count(Result.FAILURE_ALL_EXPLORED)}")
            # Write result to file
            with open(
                f"{self.results_path}/state/state_{episode.episode_id}.txt", "w"
            ) as f:
                f.write(
                    ",".join(
                        str(results[n_ep].sequence_results[i].value)
                        for i in range(len(results[n_ep].sequence_results))
                    )
                )
            pbar.update()
            # Free up memory to avoid OOM
            torch.cuda.empty_cache()
        pbar.close()

    def validate(self):
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
        from habitat.sims.habitat_simulator.actions import HabitatSimActions

        n_eps = 0
        results = []
        self.actions = [
            HabitatSimActions.move_forward,
            HabitatSimActions.turn_left,
            HabitatSimActions.turn_right,
        ]
        pbar = tqdm.tqdm(total=len(self.episodes))
        for n_ep, episode in enumerate(self.episodes):
            poses = []
            map_poses_and_obs = []
            metric = Metrics(episode.episode_id)
            results.append(metric)
            if len(self.include_ids) > 0 and episode.episode_id not in self.include_ids:
                pbar.update()
                continue
            if episode.episode_id in self.exclude_ids:
                pbar.update()
                continue
            # if '105515379_173104395' in episode.scene_id or '105515184_173104128' in episode.scene_id or '104348028_171512877' in episode.scene_id or '103997895_171031182' in episode.scene_id or '103997586_171030669' in episode.scene_id or '102816036' in episode.scene_id or '102344529' in episode.scene_id or '102815835' in episode.scene_id or '102817140' in episode.scene_id or '104348010_171512832' in episode.scene_id:
            #     continue
            n_eps += 1
            if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                self.load_scene(episode.scene_id)

            self.sim.initialize_agent(
                0,
                habitat_sim.AgentState(episode.start_position, episode.start_rotation),
            )
            self.actor.reset()
            self.follower = ShortestPathFollower(
            self.sim, goal_radius=0.25, return_one_hot=False,
                stop_on_error=True # False for debugging
            )

            sequence_id = 0
            current_obj = episode.goals[sequence_id]
            if self.config.goal_query_type == "coarse":
                goal_query = "a " + " ".join(current_obj["object_category"].split('_'))
            elif self.config.goal_query_type == "fine":
                goal_query = "a " + " ".join(current_obj["extras"]["object_category"].split("_"))
            else:
                if self.config.goal_query_processing == "extract" or self.config.goal_query_processing == "extract_and_split_support":
                    goal_query = current_obj['language_instruction'].split('Find ')[-1].split('Go to ')[-1].split('.')[0]
                elif self.config.goal_query_processing == "extract_no_support":
                    goal_query = current_obj['language_instruction'].split('Find ')[-1].split('Go to ')[-1].split('.')[0].split(' on the ')[0]
                else:
                    goal_query = current_obj['language_instruction']

            if self.config.goal_query_processing == "extract_graph":
                full_query = goal_query
                text_graph = extract_graph_from_text(goal_query)
                text_graph_queries = text_graph["nodes"]
                text_graph_queries.extend([e["relation"] for e in text_graph["edges"]])
                self.actor.set_queries(text_graph_queries, full_query)
            elif self.config.goal_query_processing == "mix":
                goal_query = current_obj['language_instruction']
                full_query = goal_query
                if "on " in goal_query:
                    goal_query = goal_query.split(' on ')[::-1]
                    self.actor.set_queries(goal_query, full_query)
                else:
                    self.actor.set_query(goal_query)
            elif self.config.goal_query_type == "detailed" and self.config.goal_query_processing == "extract_and_split_support" and " on " in goal_query:
                full_query = goal_query
                goal_query = goal_query.split(' on ')[::-1]
                self.actor.set_queries(goal_query, full_query)
            elif self.config.goal_query_type == "detailed" and self.config.goal_query_processing == "extract_and_split_support" and "above " in goal_query:
                full_query = goal_query
                goal_query = goal_query.split(' above ')[::-1]
                self.actor.set_queries(goal_query, full_query)
            elif self.config.goal_query_type == "detailed" and self.config.goal_query_processing == "extract_and_split_support" and "below " in goal_query:
                full_query = goal_query
                goal_query = goal_query.split(' below ')
                self.actor.set_queries(goal_query, full_query)
            # elif self.config.goal_query_type == "detailed" and self.config.goal_query_processing == "extract_and_split_support" and "next to " in goal_query:
            #     full_query = goal_query
            #     goal_query = goal_query.split(' next to ')
            #     # form kernel
            #     goal_query_mod = [goal_query[1], goal_query[1], goal_query[1],
            #                       goal_query[1], goal_query[0], goal_query[1],
            #                       goal_query[1], goal_query[1], goal_query[1]]
            #     self.actor.set_queries(goal_query_mod, full_query)
            else:
                self.actor.set_query(goal_query)
            
            rgb_frames = []
            is_object_in_frame = {}
            all_object_detections = {}
            while sequence_id < len(episode.goals):
                steps = 0
                not_failed = True
                running = True
                running_exploit = False
                map_poses_and_obs = []
                while steps < self.max_steps and running:
                    try:
                        observations = self.sim.get_sensor_observations()
                        # observations['depth'] = fill_depth_holes(observations['depth'])
                        observations["state"] = self.sim.get_agent(0).get_state()
                        pose = np.zeros((4,))
                        pose[0] = -observations["state"].position[2]
                        pose[1] = -observations["state"].position[0]
                        pose[2] = observations["state"].position[1]
                        # yaw
                        orientation = observations["state"].rotation
                        q0 = orientation.x
                        q1 = orientation.y
                        q2 = orientation.z
                        q3 = orientation.w
                        r = R.from_quat([q0, q1, q2, q3])
                        # r to euler
                        yaw, _, _1 = r.as_euler("yxz")
                        pose[3] = yaw

                        poses.append(pose)
                        if self.save_maps:
                            map_poses_and_obs.append(
                                {
                                    "pose_xyzyaw": pose,
                                    "pose_map": self.actor.mapper.one_map.metric_to_px(
                                        pose[0], pose[1]
                                    ),
                                    "obs_from_pose": observations,
                                }
                            )
                        if self.log_rerun:
                            rr.log("logs", rr.TextLog(f"\"{current_obj['language_instruction']}\""))
                            cam_x = -self.sim.get_agent(0).get_state().position[2]
                            cam_y = -self.sim.get_agent(0).get_state().position[0]
                            rr.log(
                                "camera/rgb",
                                rr.Image(observations["rgb"]).compress(jpeg_quality=50),
                            )
                            rr.log(
                                "camera/depth",
                                rr.Image(
                                    (observations["depth"] - observations["depth"].min())
                                    / (
                                        observations["depth"].max()
                                        - observations["depth"].min()
                                    )
                                ),
                            )
                            self.logger.log_pos(cam_x, cam_y)

                        action, called_found = self.actor.act(observations)
                        self.execute_action(action)

                        if steps % 100 == 0:
                            dist = self.get_closest_dist(
                                self.sim.get_agent(0).get_state().position,
                                current_obj["goal_object"],
                            )
                            print(
                                f"Step {steps}, current query: {goal_query}, episode_id: {episode.episode_id}, distance to closest viewpoint: {dist}"
                            )
                        steps += 1

                        if called_found or steps >= self.max_steps:
                            running = False
                            result = Result.FAILURE_OOT
                            # reset the relation graph
                            self.actor.mapper.one_map.rel_graph.reset_graph()
                            self.actor.mapper.one_map.reset_checked_image_map()
                            # We will now compute the closest distance to the bounding box of the object
                            if called_found:
                                dist = self.get_closest_dist(
                                    self.sim.get_agent(0).get_state().position,
                                    current_obj["goal_object"],
                                )
                                if dist < self.max_dist:
                                    result = Result.NO_FAILURE
                                    print("Object found!")
                                else:
                                    pos = self.actor.mapper.chosen_detection
                                    pos_metric = self.actor.mapper.one_map.px_to_metric(
                                        pos[0], pos[1]
                                    )
                                    dist_detect = self.get_closest_dist(
                                        [-pos_metric[1], self.sim.get_agent(0).get_state().position[1], -pos_metric[0]],
                                        current_obj["goal_object"],
                                    )
                                    if dist_detect < self.max_dist:
                                        result = Result.FAILURE_NOT_REACHED
                                    elif running_exploit:
                                        result = Result.FAILURE_MISDETECT_ON_MAP
                                    elif self.actor.mapper.use_detector:
                                        result = Result.FAILURE_MISDETECT
                                    not_failed = False
                                    print(
                                        f"Object not found! Dist {dist}, detect dist: {dist_detect}."
                                    )
                            else:
                                not_failed = False
                                if (
                                    result == Result.FAILURE_OOT
                                    and np.linalg.norm(poses[-1][:2] - poses[-10][:2])
                                    < 0.05
                                ):
                                    result = Result.FAILURE_STUCK
                                num_frontiers = len(self.actor.mapper.nav_goals)
                                if (
                                    result == Result.FAILURE_STUCK
                                    or result == Result.FAILURE_OOT
                                ) and num_frontiers == 0:
                                    result = Result.FAILURE_ALL_EXPLORED
                            results[-1].add_sequence(np.array(poses), result, current_obj)

                            sequence_id += 1
                            if sequence_id < len(episode.goals):
                                current_obj = episode.goals[sequence_id]
                                if self.config.goal_query_type == "coarse":
                                    goal_query = "a " + " ".join(current_obj["object_category"].split('_'))
                                elif self.config.goal_query_type == "fine":
                                    goal_query = "a " + " ".join(current_obj["extras"]["object_category"].split("_"))
                                else:
                                    if self.config.goal_query_processing == "extract" or self.config.goal_query_processing == "extract_and_split_support":
                                        goal_query = current_obj['language_instruction'].split('Find ')[-1].split('Go to ')[-1].split('.')[0]
                                    elif self.config.goal_query_processing == "extract_no_support":
                                        goal_query = current_obj['language_instruction'].split('Find ')[-1].split('Go to ')[-1].split('.')[0].split(' on the ')[0]
                                    else:
                                        goal_query = current_obj['language_instruction']

                                if self.config.goal_query_processing == "extract_graph":
                                    full_query = goal_query
                                    text_graph = extract_graph_from_text(goal_query)
                                    text_graph_queries = text_graph["nodes"]
                                    text_graph_queries.extend([e["relation"] for e in text_graph["edges"]])
                                    self.actor.set_queries(text_graph_queries, full_query)
                                elif self.config.goal_query_processing == "mix":
                                    goal_query = current_obj['language_instruction']
                                    full_query = goal_query
                                    if "on " in goal_query:
                                        goal_query = goal_query.split(' on ')[::-1]
                                        self.actor.set_queries(goal_query, full_query)
                                    else:
                                        self.actor.set_query(goal_query)
                                elif self.config.goal_query_type == "detailed" and self.config.goal_query_processing == "extract_and_split_support" and "on " in goal_query:
                                    full_query = goal_query
                                    goal_query = goal_query.split(' on ')[::-1]
                                    self.actor.set_queries(goal_query, full_query)
                                else:
                                    self.actor.set_query(goal_query)

                    except Exception as e:
                        print(str(e))
                        not_failed = False
                        running = False
                        result = Result.FAILURE_EXCEPTION
                        results[-1].add_sequence(np.array(poses), result, current_obj)
                        with open(
                            f"{self.results_path}/exceptions/{episode.episode_id}.txt", "w"
                        ) as f:
                            f.write(str(e))
                        break

            for seq_id, seq in enumerate(results[n_ep].sequence_poses):
                np.savetxt(
                    f"{self.results_path}/trajectories/poses_{episode.episode_id}_{seq_id}.csv",
                    seq,
                    delimiter=",",
                )
            # save final sim to image file
            if self.save_video:
                generate_video(video_dir=f"{self.results_path}/videos", video_name=f"{episode.episode_id}__{result.name}", images=rgb_frames)
                rgb_frames = []

            print(
                f"Overall progress: {sum([m.get_progress(self.num_seq) for m in results]) / (n_eps)}, per object: "
            )
            # Write result to file
            with open(
                f"{self.results_path}/state/state_{episode.episode_id}.txt", "w"
            ) as f:
                f.write(
                    ",".join(
                        str(results[n_ep].sequence_results[i].value)
                        for i in range(len(results[n_ep].sequence_results))
                    )
                )
            pbar.update()
            # Free up memory to avoid OOM
            torch.cuda.empty_cache()
        pbar.close()