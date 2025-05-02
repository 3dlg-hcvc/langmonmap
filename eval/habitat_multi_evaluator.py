# eval utils
import csv
import gc

import torch
from eval import within_fov_cone
from eval import get_closest_dist, FMMPlanner
from eval.actor import Actor
from eval.dataset_utils.gibson_dataset import load_gibson_episodes
from mapping import rerun_logger
from config import EvalConf
from onemap_utils import monochannel_to_inferno_rgb, generate_video, add_sim_maps_to_image
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

SEQ_LEN = 3

class Result(enum.Enum):
    SUCCESS = 1
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

    def add_sequence(self, sequence: np.ndarray, result: Result, target_object: str) -> None:
        start_id = 0
        if len(self.sequence_poses) > 0:
            start_id = sum([len(seq) for seq in self.sequence_poses])
        seq_poses = sequence[start_id:, :]
        self.sequence_poses.append(seq_poses)
        length = np.linalg.norm(seq_poses[1:, :2] - seq_poses[:-1, :2])
        self.sequence_results.append(result)
        self.sequence_lengths.append(length)
        self.sequence_object.append(target_object)

    def get_progress(self):
        return self.sequence_results.count(Result.SUCCESS) /SEQ_LEN


class HabitatMultiEvaluator:
    def __init__(self,
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
        self.object_nav_path = config.object_nav_path
        self.scene_path = config.scene_path
        self.scene_data = {}
        self.episodes = []
        self.exclude_ids = []
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
        self.num_seq = SEQ_LEN
        self.square = config.square_im

        if self.multi_object:
            self.episodes, self.scene_data = HM3DMultiDataset.load_hm3d_multi_episodes(self.episodes,
                                                                                       self.scene_data,
                                                                                       self.object_nav_path)
        else:
            raise RuntimeError("You are running the multi object evaluation with a single object config.")
        if self.actor is not None:
            self.logger = rerun_logger.RerunLogger(self.actor.mapper, False, "") if self.log_rerun else None
        self.results_path = config.results_path if len(config.results_path) > 0 else "results_multi/"
        state_dir = os.path.join(self.results_path, "state")
        os.makedirs(state_dir, exist_ok=True)
        self.exclude_ids = [p.split('state_')[-1].split('.txt')[0] for p in os.listdir(state_dir)]
        self.include_ids = []
        os.makedirs(os.path.join(self.results_path, 'trajectories'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'similarities'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'saved_maps'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'exceptions'), exist_ok=True)
        os.makedirs(os.path.join(self.results_path, 'extras'), exist_ok=True)

    def load_scene(self, scene_id: str):
        if self.sim is not None:
            self.sim.close()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path + scene_id

        backend_cfg.scene_dataset_config_file = self.scene_path + "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        backend_cfg.navmesh_settings = habitat_sim.nav.NavMeshSettings()
        backend_cfg.navmesh_settings.set_defaults()
        backend_cfg.navmesh_settings.agent_radius = 0.2
        backend_cfg.navmesh_settings.agent_height = 0.88

        self.hfov = 90 if self.square else 79
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.hfov = self.hfov
        rgb.position = np.array([0, 0.88, 0])
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        res_x = 640
        res_y = 640 if self.square else 480
        rgb.resolution = [res_y, res_x]

        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth"
        depth.hfov = self.hfov
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.position = np.array([0, 0.88, 0])
        depth.resolution = [res_y, res_x]
        agent_cfg = habitat_sim.agent.AgentConfiguration(
            height=0.88,
            radius=0.2,
            action_space=dict(
            move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
            turn_left=ActionSpec("turn_left", ActuationSpec(amount=5.0)),
            turn_right=ActionSpec("turn_right", ActuationSpec(amount=5.0)),
        ))
        agent_cfg.sensor_specifications = [rgb, depth]
        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(sim_cfg)
        if self.scene_data[scene_id].objects_loaded:
            return
        self.scene_data = HM3DDataset.load_hm3d_objects(self.scene_data, self.sim.semantic_scene.objects, scene_id)

    def execute_action(self, action: Dict
                       ):
        if 'discrete' in action.keys():
            # We have a discrete actor
            self.sim.step(action['discrete'])

        elif 'continuous' in action.keys():
            # We have a continuous actor
            self.vel_control.angular_velocity = action['continuous']['angular']
            self.vel_control.linear_velocity = action['continuous']['linear']
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
            agent_state.rotation = utils.quat_from_magnum(
                target_rigid_state.rotation
            )
            self.sim.get_agent(0).set_state(agent_state)
            self.sim.step_physics(self.time_step)

    def read_results(self, path, sort_by, data_pkl=None):
        from eval.dataset_utils import gen_multiobject_dataset
        from eval.dataset_utils.object_nav_utils import object_nav_gen
        state_dir = os.path.join(path, 'state')
        state_results = {}

        # Check if the state directory exists
        if not os.path.isdir(state_dir):
            print(f"Error: {state_dir} is not a valid directory")
            return state_results
        pose_dir = os.path.join(os.path.abspath(os.path.join(state_dir, os.pardir)), "trajectories")

        # Iterate through all files in the state directory
        data = []
        sum_successes = 0
        if data_pkl is None:
            episodes = []
            scene_data = {}
            valid_start_positions = {}
            scene_floors = {}
            scenes = {}
            episodes, scene_data = HM3DDataset.load_hm3d_episodes(episodes, scene_data, gen_multiobject_dataset.path_to_hm3d_objectnav_v2)
            number_of_floors = gen_multiobject_dataset.load_scenes(episodes, scene_data, valid_start_positions, scene_floors, scenes)
            scene_loaded = {}
            sim = None
            for filename in sorted(os.listdir(state_dir)):
                if filename.startswith('state_') and filename.endswith('.txt'):
                    try:
                        # Extract the experiment number from the filename
                        experiment_num = int(filename[6:-4])  # removes 'state_' and '.txt'
                        # Read the content of the file
                        # if experiment_num > 10:
                        #     continue
                        with open(os.path.join(state_dir, filename), 'r') as file:
                            content = file.read().strip()

                        # Convert the content to a number (assuming it's a float)
                        state_values = content.split(',')
                        state_values = [int(val) for val in state_values]
                        # Store the result in the dictionary
                        # Create a row for each sequence in the experiment

                        for seq_num, value in enumerate(state_values):
                            spl = 0
                            map_size = 0
                            if value == 1:
                                if seq_num == 2:
                                    sum_successes += 1
                                poses = np.genfromtxt(os.path.join(pose_dir, "poses_" + str(experiment_num) + "_" +
                                                                   str(seq_num) + ".csv"), delimiter=",")
                                if len(poses.shape) == 1:
                                    poses = poses.reshape((1, 4))
                                path_length = np.linalg.norm(poses[1:, :3] - poses[:-1, :3], axis=1).sum()
                                # compute the optimal path length
                                if sim is None or not sim.curr_scene_name in self.episodes[experiment_num].scene_id:
                                    if sim is not None:
                                        sim.close()
                                        sim = None
                                    sim = gen_multiobject_dataset.build_sim(gen_multiobject_dataset.path_to_hm3d_v0_2, self.episodes[experiment_num].scene_id, gen_multiobject_dataset.start_poses_tilt_angle, True)
                                if self.episodes[experiment_num].scene_id not in scene_loaded:
                                    needs_save = gen_multiobject_dataset.load_all_scene_data(self.episodes[experiment_num].scene_id,
                                                                                scenes, scene_data,
                                                                                viewpoint_conf=object_nav_gen.VPConf(1.0, 0.1, 0.05), sim=sim)
                                    if needs_save:
                                        print(f"Storing viewpoints for scene {self.episodes[experiment_num].scene_id}")
                                        gen_multiobject_dataset.store_viewpoints(scenes, self.episodes[experiment_num].scene_id,"datasets/multi_object_data")
                                    scene_loaded[self.episodes[experiment_num].scene_id] = True

                                start_pos = poses[0, :3]
                                pos = np.array([-start_pos[1], start_pos[2], -start_pos[0]])
                                floor_data = scenes[self.episodes[experiment_num].scene_id].floors[self.episodes[experiment_num].floor_id]
                                possible_objs = floor_data.objects[self.episodes[experiment_num].obj_sequence[seq_num]]
                                min_dist = np.inf
                                obj_found = False
                                for obj in possible_objs:
                                    dist, next_start = object_nav_gen.get_geodesic(pos, sim, obj, correct_start=True)
                                    if dist is None:
                                        continue
                                    obj_found = True
                                    if dist < min_dist:
                                        min_dist = dist
                                best_dist = min_dist
                                if not obj_found:
                                    print(f"Warning: No object found for sequence {seq_num} in experiment {experiment_num}")
                                spl = min(1.0, 1 * (best_dist/ max(path_length, best_dist)))

                            top_down_map = maps.get_topdown_map(
                                            sim.pathfinder,
                                            height=self.episodes[experiment_num].start_position[1],
                                            map_resolution=512,
                                            draw_border=True,
                                        )
                            # Draw the start position
                            top_down_map = gen_multiobject_dataset.draw_point(
                                sim,
                                top_down_map,
                                np.array(self.episodes[experiment_num].start_position),
                                maps.MAP_SOURCE_POINT_INDICATOR,
                            )

                            # Draw the object goals
                            object_goals = self.episodes[experiment_num].obj_sequence
                            top_down_map = gen_multiobject_dataset.draw_point(
                                sim,
                                top_down_map,
                                np.array(self.episodes[experiment_num].best_dist[seq_num][1]),
                                maps.MAP_TARGET_POINT_INDICATOR,
                            )

                            # Colorize and save the map
                            top_down_map = maps.colorize_topdown_map(top_down_map)
                            map_size = top_down_map.shape[0] * top_down_map.shape[1]

                            optimal_total_path_length = sum([d[0] for d in self.episodes[experiment_num].best_dist])
                            data.append({
                                'experiment': experiment_num,
                                'sequence': seq_num,
                                'state': value,
                                'spl': spl / self.num_seq,
                                'map_size': map_size,
                                'opt_path': optimal_total_path_length,
                                'object': self.episodes[experiment_num].obj_sequence[seq_num],
                                'scene': self.episodes[experiment_num].scene_id
                            })

                            np.savez_compressed(
                                f"{self.results_path}/saved_maps_gt/{self.episodes[experiment_num].episode_id}_{seq_num}.npz",
                                gt_topdown_map=top_down_map,
                                gt_object_goals=object_goals,
                                experiment_result=data
                            )

                        # deltas = poses[1:, :3] - poses[:-1, :3]
                        # distance_traveled = np.linalg.norm(deltas, axis=1).sum()
                        # if state_value == 1:
                        #     spl[experiment_num] = self.episodes[experiment_num].best_dist / max(
                        #         self.episodes[experiment_num].best_dist, distance_traveled)
                        # else:
                        #     spl[experiment_num] = 0
                        if self.episodes[experiment_num].episode_id != experiment_num:
                            print(
                                f"Warning, experiment_num {experiment_num} does not correctly resolve to episode_id {self.episodes[experiment_num].episode_id}")
                    except ValueError:
                        print(f"Warning: Skipping {filename} due to invalid format")
                    # except Exception as e:
                    #     print(f"Error reading {filename}: {str(e)}")
            data = pd.DataFrame(data)
        else:
            with open(data_pkl, 'rb') as f:
                data = pickle.load(f)
        # data = data[data['experiment'] < 88]
        states = data["state"].unique()
        # print(sum_successes/236)
        def has_success(group, seq_id):
            return group[(group['sequence'] == seq_id) & (group['state'] == 1)].shape[0] > 0
        def calc_prog_per_episode(group):
            successes = group.groupby('experiment').apply(lambda x: (x['state'] == 1).sum())
            progress = successes / self.num_seq
            return progress

        def calc_spl_per_episode(group):
            spls_per_exp = group.groupby('experiment')['spl'].sum()
            return spls_per_exp

        def calculate_percentages(group):
            total = len(group)
            result = pd.Series({Result(state).name: (group['state'] == state).sum() / total for state in states})
            progress = calc_prog_per_episode(group)
            spl = calc_spl_per_episode(group)
            s = progress[progress == 1]
            result['Progress'] = progress.mean()
            result['SPL'] = spl.mean()
            result['opt_PL'] = group['opt_path'].mean()
            result['Map Size'] = group['map_size'].mean() / 100
            result['s'] = s.sum() / len(progress)
            result['s_spl'] = spl[progress == 1].sum()/len(progress)

            # Calculate average SPL and multiply by 100
            # avg_spl = group['spl'].mean()
            # result['Average SPL'] = avg_spl

            return result

        # Per-object results
        object_results = data.groupby('object').apply(calculate_percentages).reset_index()
        object_results = object_results.rename(columns={'object': 'Object'})

        # Per-scene results
        scene_results = data.groupby('scene').apply(calculate_percentages).reset_index()
        scene_results = scene_results.rename(columns={'scene': 'Scene'})

        # Overall results
        overall_percentages = calculate_percentages(data)
        overall_row = pd.DataFrame([{'Object': 'Overall'} | overall_percentages.to_dict()])
        object_results = pd.concat([overall_row, object_results], ignore_index=True)

        overall_row = pd.DataFrame([{'Scene': 'Overall'} | overall_percentages.to_dict()])
        scene_results = pd.concat([overall_row, scene_results], ignore_index=True)

        # Sorting
        object_results = object_results.sort_values(by=sort_by, ascending=False)
        scene_results = scene_results.sort_values(by=sort_by, ascending=False)

        # Function to format percentages
        def format_percentages(val):
            return f"{val:.2%}" if isinstance(val, float) else val

        # Apply formatting to all columns except the first one (Object/Scene)
        object_table = object_results.iloc[:, 0].to_frame().join(
            object_results.iloc[:, 1:].applymap(format_percentages))
        scene_table = scene_results.iloc[:, 0].to_frame().join(
            scene_results.iloc[:, 1:].applymap(format_percentages))

        print(f"Results by Object (sorted by {sort_by} rate, descending):")
        print(tabulate(object_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        print(f"\nResults by Scene (sorted by {sort_by} rate, descending):")
        print(tabulate(scene_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        data_per_scene = data.groupby('scene')
        sr_per_scene = []
        spl_per_scene = []
        for scene, scene_data in data_per_scene:
            print(f"\nScene: {scene}")
            success_rates = []
            spl_values = []
            seq_numbers = []
            for i in range(self.num_seq):
                sequences = scene_data[scene_data['sequence'] == i]
                if len(sequences) > 0:
                    successful_experiments = sequences[sequences['state'] == 1]
                    spl = sequences['spl'].mean() * SEQ_LEN
                    success_rate = len(successful_experiments) / len(sequences)

                    success_rates.append(success_rate)
                    spl_values.append(spl)
                    seq_numbers.append(i)
                    print(f"  Sequence {i}:")
                    print(f"    Num of experiments: {len(sequences)}")
                    print(f"    Overall SPL: {spl:.4f}")
                    print(f"    Fraction of successful experiments: {success_rate:.2%}")
                else:
                    print(f"  Sequence {i}: No data")
                    success_rates.append(0)
                    spl_values.append(0)
            sr_per_scene.append(success_rates)
            spl_per_scene.append(spl_values)
        sr_per_scene = np.array(sr_per_scene)
        spl_per_scene = np.array(spl_per_scene)
        sr_per_scene = np.mean(sr_per_scene, axis=0)
        spl_per_scene = np.mean(spl_per_scene, axis=0)
        print(f"SPL: {spl_per_scene}, Success Rate: {sr_per_scene}")
        # Plot Success Rate
        ax1.plot(np.arange(self.num_seq), sr_per_scene, label=scene, marker='o')

        # Plot SPL
        ax2.plot(np.arange(self.num_seq), spl_per_scene, label=scene, marker='o')
        # Set up Success Rate subplot
        ax1.set_xlabel('Sequence Number')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate per Sequence')
        # ax1.legend()
        ax1.grid(True)

        # Set up SPL subplot
        ax2.set_xlabel('Sequence Number')
        ax2.set_ylabel('SPL')
        ax2.set_title('SPL per Sequence')
        # ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('output_plot.png')

        plt.show()
        # selected_experiment_ids = successful_experiments['experiment'].unique()
        # experiments_with_second_success = successful_experiments.groupby('experiment').filter(
        #     lambda x: has_success(x, 1))
        # successful_second_ids = experiments_with_second_success['experiment'].unique()
        # fraction_successful = len(successful_second_ids) / len(selected_experiment_ids) if len(
        #     selected_experiment_ids) > 0 else 0
        #
        # # Calculate conditional SPL for each experiment
        # second_sequences = data[(data['state'] == 1) & (data['sequence'] == 1)]
        # conditional_spl = second_sequences['spl'].mean()
        # print(f"\nOverall Conditional SPL (second sequence, given first success): {conditional_spl:.4f}")
        #
        # print(f"Fraction of successful first experiments: {len(selected_experiment_ids)/len(all_ids):.2%}")
        # print(f"Fraction of successful second, conditioned on first: {fraction_successful:.2%}")
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
            if str(episode.episode_id) in self.exclude_ids:
                pbar.update()
                continue
            n_eps += 1
            if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                self.load_scene(episode.scene_id)

            self.sim.initialize_agent(0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
            self.actor.reset()
            sequence_id = 0
            current_obj = episode.obj_sequence[sequence_id]
            self.actor.set_query("the "+current_obj)
            if self.log_rerun:
                pts = []
                for obj in self.scene_data[episode.scene_id].object_locations[current_obj]:
                    if not self.is_gibson:
                        pt = obj.bbox.center[[0, 2]]
                        pt = (-pt[1], -pt[0])
                        pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                    else:
                        for pt_ in obj:
                            pt = (pt_[0], pt_[1])
                            pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                pts = np.array(pts)
                rr.log("map/ground_truth", rr.Points2D(pts, colors=[[255, 255, 0]], radii=[1]))
            not_failed = True
            rgb_frames = []
            while not_failed and sequence_id < len(episode.obj_sequence):
                steps = 0
                running = True
                running_exploit = False
                map_poses_and_obs = []
                # gt_goal_objects = []
                # for aabb in self.scene_data[episode.scene_id].object_locations[current_obj]:
                #     bbox = aabb.bbox
                #     center = bbox.center[[0, 2]]
                #     size = bbox.sizes[[0, 2]]
                #     gt_goal_objects.append({
                #         "object_category": current_obj,
                #         "center": center,
                #         "size": size,
                #         "center_map": self.actor.mapper.one_map.metric_to_px(-center[1], -center[0]),
                #     })
                while steps < self.max_steps and running:
                    observations = self.sim.get_sensor_observations()
                    # observations['depth'] = fill_depth_holes(observations['depth'])
                    observations['state'] = self.sim.get_agent(0).get_state()
                    pose = np.zeros((4,))
                    pose[0] = -observations['state'].position[2]
                    pose[1] = -observations['state'].position[0]
                    pose[2] = observations['state'].position[1]
                    # yaw
                    orientation = observations['state'].rotation
                    q0 = orientation.x
                    q1 = orientation.y
                    q2 = orientation.z
                    q3 = orientation.w
                    r = R.from_quat([q0, q1, q2, q3])
                    # r to euler
                    yaw, _, _1 = r.as_euler("yxz")
                    pose[3] = yaw

                    poses.append(pose)
                    # map_poses_and_obs.append({"pose_xyzyaw": pose,
                    #                           "pose_map": self.actor.mapper.one_map.metric_to_px(pose[0], pose[1]),
                    #                           "obs_from_pose": observations
                    #                           })
                    if self.log_rerun:
                        cam_x = -self.sim.get_agent(0).get_state().position[2]
                        cam_y = -self.sim.get_agent(0).get_state().position[0]
                        rr.log("camera/rgb", rr.Image(observations["rgb"]).compress(jpeg_quality=50))
                        rr.log("camera/depth", rr.Image((observations["depth"] - observations["depth"].min()) / (
                                observations["depth"].max() - observations["depth"].min())))
                        self.logger.log_pos(cam_x, cam_y)
                    action, called_found = self.actor.act(observations)
                    self.execute_action(action)
                    if self.log_rerun:
                        self.logger.log_map()
                    
                    if self.save_video:
                        # _pad = 3
                        similarities = self.actor.mapper.get_map()
                        sim_map_layers = []
                        final_sim = None
                        if len(similarities.shape) > 3 and similarities.shape[-1] > 1:
                            similarities = similarities[0]
                            num_layers = similarities.shape[-1]
                            for i in range(num_layers):
                                sim_map = similarities[:,:,i] # + 1.0) / 2.0
                                sim_map_img = np.flip(monochannel_to_inferno_rgb(np.flip(sim_map,axis=-1).transpose((1, 0))),axis=-1)
                                sim_map_layers.append(sim_map_img)
                        else:
                            similarities = (similarities + 1.0) / 2.0
                            final_sim = np.flip(monochannel_to_inferno_rgb(np.flip(similarities[0],axis=-1).transpose((1, 0))),axis=-1)

                        # traversable_map = self.actor.mapper.one_map.navigable_map.astype(np.float32)
                        # pose_m = self.actor.mapper.one_map.metric_to_px(
                        #     -observations["state"].position[2], -observations["state"].position[0]
                        # )
                        # traversable_map[pose_m[0]-_pad:pose_m[0]+_pad, pose_m[1]-_pad:pose_m[1]+_pad] = 0.5

                        # show GT goals
                        # for _goal in current_obj["goal_object"]:
                            # bbox = _goal["aabb"]
                            # center = _goal["centroid"]
                            # center_m = self.actor.mapper.one_map.metric_to_px(
                            #             -center[2], -center[0]
                            #         )
                            # traversable_map[min(center_m[0], traversable_map.shape[0]-1), min(center_m[1], traversable_map.shape[1]-1)] = 0.7
                            # for p in _goal["navigable_points"]:
                            #     pm = self.actor.mapper.one_map.metric_to_px(
                            #             -float(p[2]), -float(p[0])
                            #         )
                            #     traversable_map[min(pm[0], traversable_map.shape[0]-1), min(pm[1], traversable_map.shape[1]-1)] = 0.7

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

                        # mark path to the goal
                        # if self.actor.mapper.path is not None and len(self.actor.mapper.path) > 0:
                        #     for pth in self.actor.mapper.path:
                        #         try:
                        #             traversable_map[pth[0], pth[1]] = 0.3
                        #         except:
                        #             pass

                        # traversable_map = np.flip(monochannel_to_inferno_rgb(np.flip(traversable_map,axis=-1).transpose((1, 0))),axis=-1)

                        # obstcl_map_layers = []
                        # if self.actor.mapper.one_map.layered:
                        #     obstacles_layered = self.actor.mapper.one_map.obstacle_map_layered.cpu().numpy()
                        #     if len(obstacles_layered.shape) > 2 and obstacles_layered.shape[-1] > 1:
                        #         num_layers = obstacles_layered.shape[-1]
                        #         for i in range(num_layers):
                        #             obstcl_map_img = np.flip(monochannel_to_inferno_rgb(np.flip(obstacles_layered[:,:,i],axis=-1).transpose((1, 0))),axis=-1)
                        #             obstcl_map_layers.append(obstcl_map_img)

                        dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                                self.scene_data[episode.scene_id].object_locations[current_obj],
                                                self.is_gibson)
                        text_to_append = f"steps={steps}, query={current_obj}, dist_to closest_goal={np.round(dist,2)}, object_detected={self.actor.mapper.object_detected}, running_exploit={running_exploit}, called_found={called_found}"
                        img_frame = add_sim_maps_to_image(
                            observation=observations,
                            maps={
                                "sim_map": final_sim,
                                # "traversable_map": traversable_map,
                                "object_detected": self.actor.mapper.object_detected,
                                "predictions": self.actor.mapper.detections,
                                "sim_map_layers": sim_map_layers,
                                # "obstcl_map_layers": obstcl_map_layers if self.actor.mapper.one_map.layered else None,
                            },
                            text_to_append=text_to_append,
                        )
                        rgb_frames.append(img_frame)

                    if steps % 100 == 0:
                        dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                                self.scene_data[episode.scene_id].object_locations[current_obj],
                                                self.is_gibson)
                        print(
                            f"Step {steps}, current object: {current_obj}, episode_id: {episode.episode_id}, distance to closest object: {dist}")
                    steps += 1

                    if steps > self.max_explore_steps and not called_found:
                        self.actor.set_exploit()
                        running_exploit = True

                    if called_found or steps >= self.max_steps:
                        running = False
                        result = Result.FAILURE_OOT
                        # We will now compute the closest distance to the bounding box of the object
                        if called_found:
                            dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                                    self.scene_data[episode.scene_id].object_locations[current_obj],
                                                    self.is_gibson)
                            if dist < self.max_dist:
                                result = Result.SUCCESS
                                print("Object found!")
                            else:
                                pos = self.actor.mapper.chosen_detection
                                pos_metric = self.actor.mapper.one_map.px_to_metric(pos[0], pos[1])
                                dist_detect = get_closest_dist([-pos_metric[1], -pos_metric[0]],
                                                               self.scene_data[episode.scene_id].object_locations[current_obj],
                                                               self.is_gibson)
                                if dist_detect < self.max_dist:
                                    result = Result.FAILURE_NOT_REACHED
                                elif running_exploit:
                                    result = Result.FAILURE_MISDETECT_ON_MAP
                                else:
                                    result = Result.FAILURE_MISDETECT
                                not_failed = False
                                print(f"Object not found! Dist {dist}, detect dist: {dist_detect}.")
                        else:
                            not_failed = False
                            if result == Result.FAILURE_OOT and np.linalg.norm(poses[-1][:2] - poses[-10][:2]) < 0.05:
                                result = Result.FAILURE_STUCK
                            num_frontiers = len(self.actor.mapper.nav_goals)
                            if (result == Result.FAILURE_STUCK or result == Result.FAILURE_OOT) and num_frontiers == 0:
                                result = Result.FAILURE_ALL_EXPLORED
                        results[-1].add_sequence(np.array(poses), result, current_obj)
                        
                        # final_sim = (self.actor.mapper.get_map() + 1.0) / 2.0
                        # confs = (self.actor.mapper.one_map.confidence_map > 0).cpu().squeeze().numpy()
                        # nav_map = self.actor.mapper.one_map.navigable_map.astype(bool)
                        # feature_map = self.actor.mapper.one_map.feature_map.cpu().numpy().astype(float)
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
                        # final_sim = np.flip(final_sim, axis=0)                        # get min and max x and y of confs

                        # np.savez_compressed(
                        #     f"{self.results_path}/saved_maps/{episode.episode_id}_{sequence_id}.npz",
                        #     nav_map=nav_map,
                        #     feature_map=feature_map,
                        #     confidence_map=confs,
                        #     query=current_obj,
                        #     final_sim_img=final_sim,
                        #     pose_observations=map_poses_and_obs,
                        #     gt_goal_objects=gt_goal_objects
                        # )

                        # cv2.imwrite(f"{self.results_path}/similarities/final_sim_{episode.episode_id}_{sequence_id}.png", final_sim)
                        # # Create the plot
                        # plt.figure(figsize=(10, 10))
                        # poses_ = np.array([self.actor.mapper.one_map.metric_to_px(*pos[:2]) for pos in poses])
                        # poses_[:, 0] -= min_x
                        # poses_[:, 1] -= min_y
                        # plt.imshow(final_sim[:, :, ::-1], interpolation='nearest', aspect='equal',
                        #           extent=(0, final_sim.shape[1], 0, final_sim.shape[0]))

                        # plt.plot(poses_[:, 0], poses_[:, 1], 'b-o')  # 'b-o' means blue line with circle markers

                        # # Set equal aspect ratio to ensure accurate positions
                        # plt.axis('equal')

                        # # Add labels and title
                        # plt.xlabel('X position')
                        # plt.ylabel('Y position')
                        # plt.title('Path of Poses')

                        # # Add grid for better readability
                        # plt.grid(True)

                        # # Save the plot as SVG
                        # plt.savefig(f"{self.results_path}/similarities/path_{episode.episode_id}_{sequence_id}.svg", format='svg', dpi=300, bbox_inches='tight')

                        # # Display the plot (optional, comment out if not needed)
                        # plt.show()

                        sequence_id += 1
                        if sequence_id < len(episode.obj_sequence):
                            current_obj = episode.obj_sequence[sequence_id]
                            self.actor.set_query("the "+current_obj)
                            if self.log_rerun:
                                pts = []
                                for obj in self.scene_data[episode.scene_id].object_locations[current_obj]:
                                    if not self.is_gibson:
                                        pt = obj.bbox.center[[0, 2]]
                                        pt = (-pt[1], -pt[0])
                                        pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                                    else:
                                        for pt_ in obj:
                                            pt = (pt_[0], pt_[1])
                                            pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                                pts = np.array(pts)
                                rr.log("map/ground_truth", rr.Points2D(pts, colors=[[255, 255, 0]], radii=[1]))

            for seq_id, seq in enumerate(results[n_ep].sequence_poses):
                np.savetxt(f"{self.results_path}/trajectories/poses_{episode.episode_id}_{seq_id}.csv", seq, delimiter=",")
            # save final sim to image file

            # save final sim to image file
            if self.save_video:
                generate_video(video_dir=f"{self.results_path}/videos", video_name=f"{episode.episode_id}_{seq_id}__{result.name}", images=rgb_frames)
                rgb_frames = []

            print(f"Overall progress: {sum([m.get_progress() for m in results]) / (n_eps)}, per object: ")
            # for obj in success_per_obj.keys():
            #     print(f"{obj}: {success_per_obj[obj] / obj_count[obj]}")
            # print(
            #     f"Result distribution: successes: {results.count(Result.SUCCESS)}, misdetects: {results.count(Result.FAILURE_MISDETECT)}, OOT: {results.count(Result.FAILURE_OOT)}, stuck: {results.count(Result.FAILURE_STUCK)}, not reached: {results.count(Result.FAILURE_NOT_REACHED)}, all explored: {results.count(Result.FAILURE_ALL_EXPLORED)}")
            # Write result to file
            with open(f"{self.results_path}/state/state_{episode.episode_id}.txt", 'w') as f:
                f.write(','.join(
                    str(results[n_ep].sequence_results[i].value) for i in range(len(results[n_ep].sequence_results))))
            pbar.update()
        pbar.close()
