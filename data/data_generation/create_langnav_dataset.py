# Followed code from https://github.com/3dlg-hcvc/hssd/

import argparse
import glob
import gzip
import itertools
import json
import lzma
import multiprocessing
import os
import os.path as osp
import pickle
import traceback
from collections import defaultdict
import random

import cv2
import GPUtil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import trimesh
import yaml
from multion_episode_generator import (
    build_goal,
    generate_multion_episode,
    ISLAND_RADIUS_LIMIT
)
from sklearn.cluster import AgglomerativeClustering

import habitat
import habitat_sim
from utils import (
    COLOR_PALETTE,
    get_topdown_map,
)
from habitat.config import read_write
from habitat.config.default import get_config
from habitat.datasets.multi_object_nav import multi_object_nav_dataset
# from habitat.datasets.pointnav.pointnav_generator import ISLAND_RADIUS_LIMIT
from habitat.tasks.rearrange.utils import get_aabb

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"

COMPRESSION = ".gz"
SCENE_DATASET_VERSION_ID = "hssd-hab"
EPISODE_DATASET_VERSION_ID = "hssd"
GOAL_CATEGORIES_FILENAME = "goal_categories.yaml"
OBJECT_ON_SAME_FLOOR = True  
NUM_EPISODES = 50000
MIN_OBJECT_DISTANCE = 1.0
MAX_OBJECT_DISTANCE = 30.0
MAX_NUM_GOALS = 3

NUM_GPUS = len(GPUtil.getAvailable(limit=256))
TASKS_PER_GPU = 1
deviceIds = GPUtil.getAvailable(order="memory")

scenes_root_path = (
    f"data/scene_datasets/{SCENE_DATASET_VERSION_ID}"
)
goal_categories_path = os.path.join(
    "data/scene_datasets/goals/", GOAL_CATEGORIES_FILENAME
)
semantic_id_mapping_path = os.path.join(
    scenes_root_path, "semantics", "object_semantic_id_mapping.json"
)
scene_splits_path = os.path.join(
    scenes_root_path, "scene_splits.yaml"
)

scene_instance_json = os.path.join(
    scenes_root_path, "scenes-multion", "{}.scene_instance.json"
)

obj_metadata_path = os.path.join(
    scenes_root_path, "semantics", "objects.csv"
)

obj_metadata = pd.read_csv(obj_metadata_path).fillna("")

with open(scene_splits_path, "r") as f:
    scene_splits = yaml.safe_load(f)

with open(goal_categories_path, "r") as f:
    all_goal_categories = yaml.safe_load(f)

with open(semantic_id_mapping_path, "r") as f:
    semantic_id_mapping = json.load(f)
    
output_dataset_folder = (
    f"data/datasets/langnav/{EPISODE_DATASET_VERSION_ID}"
)
episode_dataset_viz_folder = os.path.join(
    output_dataset_folder, "viz", "episodes"
)
goal_distances_viz_folder = os.path.join(
    output_dataset_folder, "viz", "goal_distances"
)
failure_viz_folder = os.path.join(
    output_dataset_folder, "viz", "failure_cases"
)

def get_objnav_config(i, scene):

    TASK_CFG = "habitat-lab/habitat/config/benchmark/nav/multion/multion_fp_custom.yaml"
    SCENE_DATASET_CFG = os.path.join(
        scenes_root_path, "hssd-hab-mon.scene_dataset_config.json"
    )

    objnav_config = get_config(TASK_CFG)  # .clone()

    deviceIds = GPUtil.getAvailable(
        order="memory", limit=1, maxLoad=1.0, maxMemory=1.0
    )

    if i < NUM_GPUS * TASKS_PER_GPU or len(deviceIds) == 0:
        deviceId = i % NUM_GPUS
    else:
        deviceId = deviceIds[0]

    with read_write(objnav_config):

        FOV = 90
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = (
            FOV
        )
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = (
            FOV
        )
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.hfov = (
            FOV
        )
        # TODO: confirm the width and height
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width = (
            640
        )
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.height = (
            640
        )
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = (
            640
        )
        objnav_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = (
            640
        )

        objnav_config.habitat.simulator.habitat_sim_v0.gpu_device_id = deviceId
        objnav_config.habitat.simulator.scene = scene
        objnav_config.habitat.simulator.scene_dataset = SCENE_DATASET_CFG
        objnav_config.habitat.simulator.habitat_sim_v0.enable_physics = True

        objnav_config.habitat.task.measurements = {}

    return objnav_config


def get_simulator(objnav_config):
    sim = habitat.sims.make_sim(
        "Sim-v0", config=objnav_config.habitat.simulator
    )
    # no need to recompute navmesh because forked hab-sim installation includes static objs by default
    sim.compute_navmesh_island_classifications()
    for island_idx in sim.indoor_islands:
        if sim.pathfinder.island_area(island_idx) > ISLAND_RADIUS_LIMIT:
            return sim
    return None  # scene has no valid indoor islands


def dense_sampling_trimesh(triangles, density=25.0, max_points=200000):
    # Create trimesh mesh from triangles
    t_vertices = triangles.reshape(-1, 3)
    t_faces = np.arange(0, t_vertices.shape[0]).reshape(-1, 3)
    t_mesh = trimesh.Trimesh(vertices=t_vertices, faces=t_faces)
    surface_area = t_mesh.area
    n_points = min(int(surface_area * density), max_points)
    t_pts, _ = trimesh.sample.sample_surface_even(t_mesh, n_points)
    return t_pts


def get_scene_key(glb_path):
    return osp.basename(glb_path).split(".")[0]


def get_file_opener(fname):
    ext = os.path.splitext(fname)[-1]

    if ext == ".gz":
        file_opener = gzip.open
    elif ext == ".xz":
        file_opener = lzma.open
    else:
        print(ext)
        assert False
    return file_opener


def save_dataset(dset: habitat.Dataset, fname: str):
    file_opener = get_file_opener(fname)
    if (
        os.path.basename(os.path.dirname(fname)) == "content"
        and len(dset.episodes) == 0
    ):
        print("WARNING UNEXPECTED EMPTY EPISODES: %s" % fname)
        return
    with file_opener(fname, "wt") as f:
        if len(dset.episodes) == 0:
            print("WARNING EMPTY EPISODES: %s" % fname)
            f.write(
                json.dumps(
                    {
                        "episodes": [],
                        "category_to_task_category_id": dset.category_to_scene_annotation_category_id,
                        "category_to_scene_annotation_category_id": dset.category_to_scene_annotation_category_id,
                    }
                )
            )
        else:
            dset_str = dset.to_json()
            f.write(dset_str)

def generate_object_instruction(scene, object_id, category_name):
    
    obj_description = {}

    obj_semantic_detail = obj_metadata[obj_metadata["id"] == object_id]
    obj_description["obj_name"] = obj_semantic_detail["name"].tolist()[0] if len(obj_semantic_detail["name"].tolist()) > 0 else category_name
    obj_description["obj_main_cat"] = obj_semantic_detail["main_category"].tolist()[0] if len(obj_semantic_detail["main_category"].tolist()) > 0 else ''
    obj_description["obj_super_cat"] = obj_semantic_detail["super_category"].tolist()[0] if len(obj_semantic_detail["super_category"].tolist()) > 0 else ''
    obj_description["obj_wnsynsetkey"] = obj_semantic_detail["wnsynsetkey"].tolist()[0] if len(obj_semantic_detail["wnsynsetkey"].tolist()) > 0 else ''
    
    # parent information
    with open(scene_instance_json.format(scene), "r") as f:
        scene_inst_json = json.load(f)
        
    all_objects = scene_inst_json["object_instances"]
    
    object_detail = list(filter(lambda x: x["template_name"] == object_id, all_objects))
    if len(object_detail) <= 0:
        return obj_description
    
    object_detail = object_detail[0]
    object_parent = object_detail["stk_mapping"]["parent"] if "parent" in object_detail["stk_mapping"] else []
    
    if len(object_parent) > 0:
        object_parent_detail = list(filter(lambda x: x["instance_id"] == object_parent["instance_id"], all_objects))[0]
        object_parent_id = object_parent_detail["template_name"]
    
        parent_semantic_detail = obj_metadata[obj_metadata["id"] == object_parent_id]
        obj_description["parent_name"] = parent_semantic_detail["name"].tolist()[0] if len(parent_semantic_detail["name"].tolist()) > 0 else ''
        obj_description["parent_main_cat"] = parent_semantic_detail["main_category"].tolist()[0] if len(parent_semantic_detail["main_category"].tolist()) > 0 else ''
        obj_description["parent_super_cat"] = parent_semantic_detail["super_category"].tolist()[0] if len(parent_semantic_detail["super_category"].tolist()) > 0 else ''
        obj_description["parent_wnsynsetkey"] = parent_semantic_detail["wnsynsetkey"].tolist()[0] if len(parent_semantic_detail["wnsynsetkey"].tolist()) > 0 else ''
        
    else:
        obj_description["parent_name"] = ''
        obj_description["parent_main_cat"] = ''
        obj_description["parent_super_cat"] = ''
        obj_description["parent_wnsynsetkey"] = ''
        
    return obj_description

def generate_scene(args):
    i, scene, split, goal_categories = args
    objnav_config = get_objnav_config(i, scene)

    sim = get_simulator(objnav_config)

    if sim is None:
        print(f"Scene {scene} has no valid indoor islands.")
        return scene, 0, defaultdict(list), None, 0

    total_objects = sim.get_rigid_object_manager().get_num_objects()

    # Check there exists a navigable point
    test_point = sim.sample_navigable_point()
    if total_objects == 0 or not sim.is_navigable(np.array(test_point)):
        print(f"Scene {scene} has no objects / is not navigable: %s" % scene)
        sim.close()
        return scene, total_objects, defaultdict(list), None, 0

    objects = []
    objects_grouped_by_cat = {}

    rgm = sim.get_rigid_object_manager()
    obj_ids = [int(x.split(",")[5]) for x in rgm.get_objects_info()[1:]]

    # recording (bboxes, category, etc.) info about all objects in scene
    for obj_id in tqdm.tqdm(
        obj_ids,
        desc="Generating object data",
    ):
        source_obj = rgm.get_object_by_id(obj_id)
        semantic_id = source_obj.semantic_id
        
        category_name = None
        for cat, cat_id in semantic_id_mapping.items():
            if cat_id == semantic_id:
                category_name = cat
                if cat not in objects_grouped_by_cat:
                    objects_grouped_by_cat[cat] = 0
                objects_grouped_by_cat[cat] += 1
                
                break

        # CRUCIAL: replacing semantic id with object id to fetch instance maps
        for node in source_obj.visual_scene_nodes:
            node.semantic_id = obj_id

        if (
            category_name is None or category_name not in goal_categories
        ):  # non-goal category
            continue

        if source_obj is None:
            print("=====> Source object is None. Skipping.")
            continue

        aabb = get_aabb(obj_id, sim, transformed=True)
        center = np.array(source_obj.translation)
        sizes = np.array(source_obj.root_scene_node.cumulative_bb.size())
        rotation = source_obj.rotation

        obb = habitat_sim.geo.OBB(center, sizes, rotation)
        object_name = rgm.get_object_handle_by_id(obj_id)
        obj_description = generate_object_instruction(scene, object_name.split('_:')[0], category_name)
        obj_instance_name = obj_description["obj_name"]
        
        obj = {
            "center": source_obj.translation,
            "id": obj_id,
            "object_name": object_name,
            "obb": obb,
            "aabb": aabb,
            "category_id": semantic_id,
            "category_name": category_name,
            "category_desc": json.dumps(obj_description),
            "obj_instance_name": obj_instance_name
        }
        objects.append(obj)

    print("Scene loaded.")
    fname_obj = (
        f"{output_dataset_folder}/{split}/scene_goals/{scene}_goal_objs.pkl"
    )
    fname = (
        f"{output_dataset_folder}/{split}/content/{scene}.json{COMPRESSION}"
    )

    ############################################################################
    # Pre-compute goals
    ############################################################################
    obj_file_exists = os.path.exists(fname_obj)

    if obj_file_exists:
        with open(fname_obj, "rb") as f:
            goals_by_category = pickle.load(f)
        total_objects_by_cat = {
            k: len(v["goals"]) for k, v in goals_by_category.items()
        }
    else:
        # goals_by_category = defaultdict(list)
        goals_by_category = {}
        cell_size = (
            objnav_config.habitat.simulator.agents.main_agent.radius / 2.0
        )
        categories_to_counts = {}

        for obj in tqdm.tqdm(objects, desc="Objects for %s:" % scene):
            instance_name = obj['obj_instance_name']
            goal_cat_key = osp.basename(scene) + "_" + obj["category_name"]
            all_goal_names = ([o.obj_instance_name for o in goals_by_category[
                goal_cat_key
            ]["goals"]] if goal_cat_key in goals_by_category else [])
            if instance_name in all_goal_names:
                continue
            
            if obj["category_name"] not in categories_to_counts:
                categories_to_counts[obj["category_name"]] = [0, 0]
            categories_to_counts[obj["category_name"]][1] += 1
            print(
                obj["category_name"], obj["category_id"], obj["category_name"]
            )

            goal, topdown_map = build_goal(
                sim,
                object_id=obj["id"],
                object_name_id=obj["object_name"],
                object_category_name=obj["category_name"],
                object_category_id=obj["category_id"],
                object_category_desc=obj["category_desc"],
                obj_instance_name=obj['obj_instance_name'],
                object_position=obj["center"],
                grid_radius=1.5,
            )

            if goal == None:
                # os.makedirs(
                #     os.path.join(failure_viz_folder, scene), exist_ok=True
                # )
                # fail_case_path = os.path.join(
                #     failure_viz_folder,
                #     scene,
                #     f'{obj["category_name"]}_{obj["object_name"]}.jpg',
                # )
                # cv2.imwrite(fail_case_path, topdown_map[:, :, ::-1])
                continue

            categories_to_counts[obj["category_name"]][0] += 1
            if (
                osp.basename(scene) + "_" + obj["category_name"]
                not in goals_by_category.keys()
            ):
                goals_by_category[
                    osp.basename(scene) + "_" + obj["category_name"]
                ] = {"goals": [], "topdown_maps": {}}

            goals_by_category[
                osp.basename(scene) + "_" + obj["category_name"]
            ]["goals"].append(goal)
            goals_by_category[
                osp.basename(scene) + "_" + obj["category_name"]
            ]["topdown_maps"][obj["id"]] = topdown_map

        for obj_cat in sorted(list(categories_to_counts.keys())):
            nvalid, ntotal = categories_to_counts[obj_cat]
            print(
                f"Category: {obj_cat:<15s} | {nvalid:03d}/{ntotal:03d} instances"
            )
        os.makedirs(osp.dirname(fname_obj), exist_ok=True)
        total_objects_by_cat = {
            k: len(v["goals"]) for k, v in goals_by_category.items()
        }
        with open(fname_obj, "wb") as f:
            pickle.dump(goals_by_category, f)

    ############################################################################
    # Compute ObjectNav episodes
    ############################################################################
    if os.path.exists(fname):
        print("Scene already generated. Skipping")
        sim.close(destroy=True)
        return scene, total_objects, total_objects_by_cat, None, None

    total_valid_cats = len(total_objects_by_cat)
    dset = habitat.datasets.make_dataset("MultiObjectNav-v2")

    dset.category_to_task_category_id = semantic_id_mapping
    dset.category_to_scene_annotation_category_id = semantic_id_mapping
    dset_goals_by_category = {
        k: {"goals": v["goals"]} for k, v in goals_by_category.items()
    }
    # dset_goals_by_category = {
    #     k: {v["goals"]} for k, v in goals_by_category.items()
    # }
    dset.goals_by_category = dset_goals_by_category
    scene_dataset_config = objnav_config.habitat.simulator.scene_dataset

    eps_generated = 0
    pbar = tqdm.tqdm(total = NUM_EPISODES, desc=scene)
    max_tries = 1000
    _tries = 0
    while eps_generated < NUM_EPISODES and _tries < max_tries:
        
        goal_cat_selected = random.sample(goals_by_category.keys(), MAX_NUM_GOALS)
        goals = [random.choice(goals_by_category[c]['goals']) for c in goal_cat_selected]
        
        ep = generate_multion_episode(
            sim,
            goals,
            closest_dist_limit=MIN_OBJECT_DISTANCE,
            furthest_dist_limit=MAX_OBJECT_DISTANCE,
            scene_dataset_config=scene_dataset_config,
            same_floor_flag=OBJECT_ON_SAME_FLOOR,
            eps_generated=eps_generated,
        )
        
        _tries += 1
        if ep is None:
            continue
        
        eps_generated += 1
        dset.episodes.append(ep)
        pbar.update()
        
    os.makedirs(osp.dirname(fname), exist_ok=True)
    save_dataset(dset, fname)
    sim.close(destroy=True)
    return (
        scene,
        total_objects,
        total_objects_by_cat,
        fname,
        len(dset.episodes),
    )


def read_dset(json_fname):
    dset2 = habitat.datasets.make_dataset("MultiObjectNav-v2")
    file_opener = get_file_opener(json_fname)

    with file_opener(json_fname, "rt") as f:
        dset2.from_json(f.read())
    return dset2


def prepare_inputs(split):
    scenes = scene_splits[split]
    goal_categories = all_goal_categories[split]
    return [(i, scene, split, goal_categories) for i, scene in enumerate(scenes)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "minival", "*"],
        required=True,
        type=str,
    )
    args = parser.parse_args()

    mp_ctx = multiprocessing.get_context("forkserver")

    np.random.seed(1234)
    if args.split == "*":
        inputs = []
        for split in ["train", "val", "test", "minival"]:
            inputs += prepare_inputs(split)
    else:
        inputs = prepare_inputs(args.split)

    GPU_THREADS = NUM_GPUS * TASKS_PER_GPU
    print("GPU threads:", GPU_THREADS)
    print("*" * 50)

    # # [DEBUG]:
    # total_all = 0
    # subtotals = []
    # for inp in tqdm.tqdm(inputs):
    #     if inp[1] != '102344022':
    #         continue
    #     scene, subtotal, subtotal_by_cat, fname = generate_scene(inp)
    #     total_all += subtotal
    #     subtotals.append(subtotal_by_cat)

    # Generate episodes for all scenes
    os.makedirs(output_dataset_folder, exist_ok=True)
    no_episode_scenes = []

    # Create split outer files
    if args.split == "*":
        splits = ["train", "val", "test", "minival"]
    else:
        splits = [args.split]
    for split in splits:
        dset = habitat.datasets.make_dataset("MultiObjectNav-v2")
        dset.category_to_task_category_id = semantic_id_mapping
        dset.category_to_scene_annotation_category_id = semantic_id_mapping
        global_dset = (
            f"{output_dataset_folder}/{split}/{split}.json{COMPRESSION}"
        )
        if os.path.exists(global_dset):
            os.remove(global_dset)
        if not os.path.exists(os.path.dirname(global_dset)):
            os.mkdir(os.path.dirname(global_dset))
        jsons_gz = glob.glob(
            f"{output_dataset_folder}/{split}/content/*.json{COMPRESSION}"
        )

        save_dataset(dset, global_dset)

    with mp_ctx.Pool(GPU_THREADS, maxtasksperchild=2) as pool, tqdm.tqdm(
        total=len(inputs)
    ) as pbar, open(
        os.path.join(output_dataset_folder, "train_subtotals.json"), "w"
    ) as f:
        total_all = 0
        subtotals = []
        for (
            scene,
            subtotal,
            subtotal_by_cat,
            fname,
            num_episodes,
        ) in pool.imap_unordered(generate_scene, inputs):
            pbar.update()
            total_all += subtotal
            subtotals.append(subtotal_by_cat)
            if num_episodes == 0:
                no_episode_scenes.append(scene)
            print("Scene with no episodes:", no_episode_scenes)
        print(total_all)
        print(subtotals)

        json.dump({"total_objects:": total_all, "subtotal": subtotals}, f)
