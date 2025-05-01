from eval.dataset_utils import LangMonEpisode, SceneData, SemanticObject

# typing
from typing import Dict, List

# fs utils
import os
from os import listdir
import gzip
import json


def load_hm3d_multi_episodes(episodes: List[LangMonEpisode], scene_data: Dict[str, SceneData], object_nav_path: str):
    """
    Loads LangMON episodes
    """
    i = 0
    langmon_dataset_path = os.path.join(object_nav_path, "content")
    files = listdir(langmon_dataset_path)
    files = sorted(files, key=str.casefold)
    for file in files:
        if file.endswith(".json.gz"):
            with gzip.open(os.path.join(langmon_dataset_path, file), 'r') as f:
                json_data = json.load(f)
                if len(json_data['episodes']) == 0:
                    continue
                scene_id = json_data['episodes'][0]['scene_id']
                if scene_id not in scene_data:
                    scene_data_ = SceneData(scene_id, {}, {})
                else:
                    scene_data_ = scene_data[scene_id]
                for ep in json_data['episodes']:
                    episode = LangMonEpisode(
                        scene_id=ep['scene_id'],
                        scene_dataset_config=ep['scene_dataset_config'],
                        episode_id=ep['scene_id']+'__'+ep['episode_id'],
                        start_position=ep['start_position'],
                        start_rotation=ep['start_rotation'],
                        goals=ep['goals'],
                        shortest_paths=ep['shortest_paths'],
                        object_category=ep['object_category']
                    )
                    episodes.append(episode)
                    # for obj in ep['goals']:
                    #     if obj not in scene_data_.object_locations.keys():
                    #         scene_data_.object_locations[obj] = []
                    #         scene_data_.object_ids[obj] = []
                    i += 1
                scene_data[scene_id] = scene_data_

    return episodes, scene_data


if __name__ == '__main__':
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
        print(f"Object {obj}, count {obj_counts[obj]}, percentage {obj_counts[obj] / total}")