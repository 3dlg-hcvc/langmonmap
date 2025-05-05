from eval.dataset_utils import LanguageNavEpisode, SceneData, SemanticObject

# typing
from typing import Dict, List

# fs utils
import os
from os import listdir
import gzip
import json


def load_goat_bench_episodes(episodes: List[LanguageNavEpisode], scene_data: Dict[str, SceneData], object_nav_path: str):
    """
    Loads Goat-bench episodes
    """
    i = 0
    dataset_path = os.path.join(object_nav_path, "content")
    files = listdir(dataset_path)
    files = sorted(files, key=str.casefold)
    for file in files:
        if file.endswith(".json.gz"):
            with gzip.open(os.path.join(dataset_path, file), 'r') as f:
                json_data = json.load(f)
                if len(json_data['episodes']) == 0:
                    continue
                scene_id = file.split(".json.gz")[0]
                if scene_id not in scene_data:
                    scene_data_ = SceneData(scene_id, {}, {})
                else:
                    scene_data_ = scene_data[scene_id]
                
                goals = json_data["goals"]

                for ep in json_data['episodes']:
                    episode = LanguageNavEpisode(
                        scene_id=ep['scene_id'],
                        scene_dataset_config=ep['scene_dataset_config'],
                        episode_id=scene_id+'__'+str(ep['episode_id']),
                        start_position=ep['start_position'],
                        start_rotation=ep['start_rotation'],
                    )
                    episode.goals = []

                    filtered_tasks = []
                    for _goal in ep['tasks']:
                        ## only consider language goals for now
                        goal_category = _goal[0]
                        goal_type = _goal[1]
                        goal_inst_id = _goal[2]
                        if goal_type not in ['description']:
                            continue
                        
                        dset_same_cat_goals = [
                            x
                            for x in goals.values()
                            if x[0]["object_category"] == goal_category
                        ]

                        if goal_type == "description":
                            goal_inst = [
                                x
                                for x in dset_same_cat_goals[0]
                                if x["object_id"] == goal_inst_id
                            ]
                            if len(goal_inst[0]["lang_desc"].split(" ")) <= 55:
                                filtered_tasks.append(_goal)
                            else:
                                num_filtered_eps += 1
                        else:
                            filtered_tasks.append(_goal)
                    
                    for goal in filtered_tasks:
                        goal_type = goal[1]
                        goal_category = goal[0]
                        goal_inst_id = goal[2]

                        dset_same_cat_goals = [
                            x
                            for x in goals.values()
                            if x[0]["object_category"] == goal_category
                        ]
                        children_categories = dset_same_cat_goals[0][0][
                            "children_object_categories"
                        ]
                        for child_category in children_categories:
                            goal_key = "{}_{}".format(
                                scene_id,
                                child_category,
                            )
                            if goal_key not in goals:
                                continue
                            dset_same_cat_goals[0].extend(goals[goal_key])

                        assert (
                            len(dset_same_cat_goals) == 1
                        ), f"more than 1 goal categories for {goal_category}"

                        if goal_type == "object":
                            episode.goals.append(dset_same_cat_goals[0])
                        else:
                            goal_inst = [
                                x
                                for x in dset_same_cat_goals[0]
                                if x["object_id"] == goal_inst_id
                            ]
                            episode.goals.append(goal_inst)

                    episodes.append(episode)
                    # for obj in ep['goals']:
                    #     if obj not in scene_data_.object_locations.keys():
                    #         scene_data_.object_locations[obj] = []
                    #         scene_data_.object_ids[obj] = []
                    i += 1
                scene_data[scene_id] = scene_data_

    return episodes, scene_data


if __name__ == '__main__':
    eps, scene_data = load_goat_bench_episodes([], {}, "datasets/multiobject_episodes/")
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