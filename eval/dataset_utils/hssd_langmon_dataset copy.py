import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
import numpy as np
from eval.dataset_utils import LangMonEpisode, SceneData

# typing
from typing import Dict, List

# fs utils
import os
from os import listdir
import gzip
import json


def load_hm3d_multi_episodes(
    episodes: List[LangMonEpisode],
    scene_data: Dict[str, SceneData],
    object_nav_path: str,
    scene_path: str,
    square: bool,
):
    """
    Loads LangMON episodes
    """
    i = 0
    langmon_dataset_path = os.path.join(object_nav_path, "content")
    files = listdir(langmon_dataset_path)
    files = sorted(files, key=str.casefold)
    sim = None
    for file in files:
        if file.endswith(".json.gz"):
            with gzip.open(os.path.join(langmon_dataset_path, file), "r") as f:
                json_data = json.load(f)

            if len(json_data["episodes"]) == 0:
                continue
            scene_id = json_data["episodes"][0]["scene_id"]
            if scene_id not in scene_data:
                scene_data_ = SceneData(scene_id, {}, {})
            else:
                scene_data_ = scene_data[scene_id]

            if sim is None or not sim.curr_scene_name in scene_id:
                sim = load_scene(
                    sim=sim,
                    scene_path=scene_path,
                    scene_id=scene_id,
                    square=square,
                )
            is_updated = False
            for ep in json_data["episodes"]:
                if "shortest_dists" not in ep or len(ep["shortest_dists"]) == 0:
                    start_pos = ep["start_position"]
                    for i, _g in enumerate(ep["goals"]):
                        nearest_nav_points = []
                        shortest_dists = []
                        for _obj in _g["goal_object"]:
                            nearest_nav_points.append([float(_obj["nearest_nav_point"][0]), float(_obj["nearest_nav_point"][1]), float(_obj["nearest_nav_point"][2])])
                            shortest_path = habitat_sim.nav.ShortestPath()
                            shortest_path.requested_start = start_pos
                            shortest_path.requested_end = nearest_nav_points[-1]
                            sim.pathfinder.find_path(shortest_path)
                            shortest_dists.append(shortest_path.geodesic_distance)
                        shortest_dists_index = np.argmin(np.array(shortest_dists))
                        episode.shortest_dists.append(shortest_dists[shortest_dists_index])
                        start_pos = nearest_nav_points[shortest_dists_index]

                        is_updated = True
                        
                episode = LangMonEpisode(
                    scene_id=ep["scene_id"],
                    scene_dataset_config=ep["scene_dataset_config"],
                    episode_id=ep["scene_id"] + "__" + ep["episode_id"],
                    start_position=ep["start_position"],
                    start_rotation=ep["start_rotation"],
                    goals=ep["goals"],
                    shortest_paths=ep["shortest_paths"],
                    object_category=ep["object_category"],
                )
                episodes.append(episode)
                # for obj in ep['goals']:
                #     if obj not in scene_data_.object_locations.keys():
                #         scene_data_.object_locations[obj] = []
                #         scene_data_.object_ids[obj] = []
                i += 1
                scene_data[scene_id] = scene_data_

                if is_updated:
                    with gzip.open(os.path.join(langmon_dataset_path, file), 'w') as fout:
                        fout.write(json_data.encode('utf-8'))

    return episodes, scene_data


def load_scene(sim, scene_path: str, scene_id: str, square: bool = True):
    if sim is not None:
        sim.close()
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    backend_cfg.override_scene_light_defaults = True
    backend_cfg.pbr_image_based_lighting = True
    backend_cfg.scene_id = scene_id

    backend_cfg.scene_dataset_config_file = (
        scene_path + "hssd-hab-mon.scene_dataset_config.json"
    )

    hfov = 90 if square else 79
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.hfov = hfov
    rgb.position = np.array([0, 1.5, 0])
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    res_x = 640
    res_y = 640 if square else 480
    rgb.resolution = [res_y, res_x]

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.hfov = hfov
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.position = np.array([0, 1.5, 0])
    depth.resolution = [res_y, res_x]
    agent_cfg = habitat_sim.agent.AgentConfiguration(
        action_space=dict(
            move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
            turn_left=ActionSpec("turn_left", ActuationSpec(amount=30.0)),
            turn_right=ActionSpec("turn_right", ActuationSpec(amount=30.0)),
        )
    )
    agent_cfg.sensor_specifications = [rgb, depth]
    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)
    return sim


if __name__ == "__main__":
    eps, scene_data = load_hm3d_multi_episodes([], {}, "datasets/multiobject_episodes/")
    print(f"Found {len(eps)} episodes")
    scene_dist = {}
    for ep in eps:
        if ep.scene_id not in scene_dist:
            scene_dist[ep.scene_id] = 1
        else:
            scene_dist[ep.scene_id] += 1

    for sc in scene_dist:
        print(f"Scene {sc}, number of eps {scene_dist[sc]}")

    obj_counts = {}
    for ep in eps:
        for obj in ep.obj_sequence:
            if obj not in obj_counts:
                obj_counts[obj] = 1
            else:
                obj_counts[obj] += 1
    total = sum([obj_counts[obj] for obj in obj_counts])
    for obj in obj_counts:
        print(
            f"Object {obj}, count {obj_counts[obj]}, percentage {obj_counts[obj] / total}"
        )
