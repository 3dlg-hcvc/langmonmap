from dataclasses import dataclass

# typing
import os
from typing import List, Dict, Union, Optional
import numpy as np

@dataclass
class SemanticObject:
    object_id: str  # unique identifier
    object_category: str
    bbox: List[float]
    semantic_id: Optional = None
    view_pts: Union[List, None] = None

    def __eq__(self, other):
        if isinstance(other, SemanticObject):
            return self.object_id == other.object_id
        elif isinstance(other, str):
            return self.object_id == other
        return False


@dataclass
class SceneData:
    scene_id: str
    object_locations: Dict[str, List[SemanticObject]]
    object_ids: Dict[str, List[str]]
    objects_loaded: bool = False


@dataclass
class Episode:
    scene_id: str
    episode_id: int
    start_position: List[float]
    start_rotation: List[float]
    obj_sequence: List[str]
    best_dist: Union[float, List[float]]
    floor_id: Union[int, None] = None

@dataclass
class GibsonEpisode(Episode):
    object_id: int = 0
    floor_id: int = 0

@dataclass
class LangMonEpisode():
    r"""LangMon Episode
    """

    scene_id: str
    scene_dataset_config: str
    episode_id: str
    start_position: List[float]
    start_rotation: List[float]
    goals: list = None
    shortest_paths: Optional[List[List]] = None
    object_category: Optional[List[str]] = None
    object_index: Optional[int] = 0
    current_goal_index: Optional[int] = 0
    shortest_dists: Optional[list] = None

@dataclass
class LanguageNavEpisode():
    r"""Goat-bench Episode
    """
    scene_id: str
    scene_dataset_config: str
    episode_id: str
    start_position: List[float]
    start_rotation: List[float]
    goals: list = None
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List]] = None
    shortest_dists: Optional[list] = None
    object_category: Optional[str] = None
    object_instance_id: Optional[int] = None
    instructions: Optional[List[str]] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_instance_id}"
