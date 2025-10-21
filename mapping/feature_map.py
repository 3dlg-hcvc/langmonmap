"""
This is the core mapping module, which contains the OneMap class.
"""
from transforms3d.derivations.angle_axes import point

from mapping import (precompute_gaussian_kernel_components,
                     precompute_gaussian_sum_els, gaussian_kernel_sum,
                     compute_gaussian_kernel_components,
                     detect_frontiers,
                     relation_graph
                     )
from mapping.mlfm_utils import *
from config import MappingConf

from onemap_utils import ceildiv

import time
import collections

# enum
from enum import Enum

# NumPy
import numpy as np

# typing
from typing import Tuple, List, Optional

# rerun
import rerun as rr

# torch
import torch

# warnings
import warnings

# cv2
import cv2
import open3d as o3d

# functools
from functools import wraps

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

def rotate_pcl(
        pointcloud: torch.Tensor,
        tf_camera_to_episodic: torch.Tensor,
) -> torch.Tensor:
    # TODO We might be interested in a complete 3d rotation if the camera is not perfectly horizontal
    rotation_matrix = tf_camera_to_episodic[:3, :3]

    yaw = torch.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    # print(yaw)
    r = torch.tensor([[torch.cos(yaw), -torch.sin(yaw)], [torch.sin(yaw), torch.cos(yaw)]], dtype=torch.float32).to("cuda")
    pointcloud[:, :2] = (r @ pointcloud[:, :2].T).T
    return pointcloud

def print_memory_stats(label):
    print(f"\n--- Memory Stats for {label} ---")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")

class DenseProjectionType(Enum):
    INTERPOLATE = "interpolate"
    SUBSAMPLE = "subsample"


class FusionType(Enum):
    EMA = "EMA"
    SPATIAL = "Spatial"


class OneMap:
    feature_map: torch.Tensor  # map where first dimension is x direction, second dimension is y, and last direction is
    # feature_dim
    obstacle_map: torch.Tensor  # map where first dimension is x direction, second dimension is y, and last direction is
    obstacle_map_layered: torch.Tensor
    # obstacle likelihood
    navigable_map: np.ndarray  # binary traversability map where first dimension is x direction, second dimension is y
    # navigable likelihood
    fully_explored_map: np.ndarray  # binary explored map where first dimension is x direction, second dimension is y
    checked_map: np.ndarray  # binary checked map where first dimension is x direction, second dimension is y,
    # can be reset
    confidence_map_feats: torch.Tensor
    confidence_map: torch.Tensor
    checked_conf_map: torch.Tensor
    updated_mask: torch.Tensor  # tracks which cells have been updated, for lazy similarity computation

    def __init__(self,
                 feature_dim: int,
                 config: MappingConf,
                 dense_projection: DenseProjectionType = DenseProjectionType.INTERPOLATE,
                 fusion_type: FusionType = FusionType.EMA,
                 map_device: str = "cuda",
                 ) -> None:
        """

        :param feature_dim: The dimension of the feature space
        :param n_cells: The number of cells in the x and y direction respectively
        :param size: The size of the map in meters
        :param dense_projection: The type of dense projection to use, must be one of DenseProjectionType
        :param fusion_type: The type of fusion to use, must be one of FusionType
        """
        assert isinstance(dense_projection,
                          DenseProjectionType), "Invalid dense_projection. It should be one of DenseProjection."
        assert isinstance(fusion_type, FusionType), "Invalid fusion_type. It should be one of FusionType."

        self.dense_projection = dense_projection
        self.fusion_type = fusion_type if config.probabilistic_fusion else FusionType.SPATIAL
        self.map_device = map_device

        self.n_cells = config.n_points

        self.obstacle_min = config.obstacle_min
        self.obstacle_max = config.obstacle_max

        self.layered = config.layered
        self.z_bins_lower, self.z_bins_upper, self.z_bins_step = config.z_bins_lower, config.z_bins_upper, config.z_bins_step
        self.z_bins = torch.arange(self.z_bins_lower, self.z_bins_upper, config.z_bins_step).to("cuda")
        if self.layered:
            self.n_layers = len(self.z_bins) + 1
        else:
            self.n_layers = 0
        self.map_center_cells = self.map_center_cells = torch.tensor([self.n_cells // 2, self.n_cells // 2],
                                                                     dtype=torch.int32).to("cuda")
        self.size = config.size
        self.cell_size = self.size / self.n_cells
        self.feature_dim = feature_dim
        if self.layered:
            self.feature_map = torch.zeros((self.n_cells, self.n_cells, self.n_layers, feature_dim), dtype=torch.float32)
        else:
            self.feature_map = torch.zeros((self.n_cells, self.n_cells, feature_dim), dtype=torch.float32)
        self.feature_map = self.feature_map.to(self.map_device)

        self.obstacle_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32)
        if self.layered:
            self.obstacle_map_layered = torch.zeros((self.n_cells, self.n_cells, self.n_layers), dtype=torch.float32)
        self.agent_radius = config.agent_radius
        col_kernel_size = self.n_cells / self.size * self.agent_radius
        col_kernel_size = int(col_kernel_size) + (int(col_kernel_size) % 2 == 0)
        self.navigable_map = np.ones((self.n_cells, self.n_cells), dtype=bool)
        self.occluded_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        self.navigable_kernel = np.ones((col_kernel_size, col_kernel_size), np.uint8)

        self.fully_explored_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        self.checked_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        if self.layered:
            self.confidence_map_feats = torch.zeros((self.n_cells, self.n_cells, self.n_layers), dtype=torch.float32).to(self.map_device)
        self.confidence_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32)
        self.confidence_map = self.confidence_map.to(self.map_device)
        self.checked_conf_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32)
        self.checked_conf_map = self.checked_conf_map.to(self.map_device)

        if self.layered:
            self.updated_mask = torch.zeros((self.n_cells, self.n_cells, self.n_layers), dtype=torch.bool).to(self.map_device)
        else:
            self.updated_mask = torch.zeros((self.n_cells, self.n_cells), dtype=torch.bool).to(self.map_device)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_initialized = False
        self.agent_height_0 = None

        self.kernel_half = int(np.round(config.blur_kernel_size / self.cell_size))
        self.kernel_size = self.kernel_half * 2 + 1
        self.kernel_components_sum = precompute_gaussian_sum_els(self.kernel_size).to("cuda")
        self.kernel_components = precompute_gaussian_kernel_components(self.kernel_size).to("cuda")
        self.kernel_ids = torch.arange(-self.kernel_half, self.kernel_half + 1).to("cuda")
        self.kernel_ids_x, self.kernel_ids_y = torch.meshgrid(self.kernel_ids, self.kernel_ids)
        self.kernel_ids_x = self.kernel_ids_x.unsqueeze(0)
        self.kernel_ids_y = self.kernel_ids_y.unsqueeze(0)
        print("ValueMap initialized. The map contains {} cells, each storing {} features. The resulting"
              " size is {} Mb".format(self.n_cells ** 2, feature_dim, self.feature_map.element_size() *
                                      self.feature_map.nelement() / 1024 / 1024))

        self.obstacle_map_threshold = config.obstacle_map_threshold
        self.fully_explored_threshold = config.fully_explored_threshold
        self.checked_map_threshold = config.checked_map_threshold
        self.depth_factor = config.depth_factor
        self.gradient_factor = config.gradient_factor
        self.optimal_object_distance = config.optimal_object_distance
        self.optimal_object_factor = config.optimal_object_factor
        self.filter_stairs = config.filter_stairs
        self.floor_threshold = config.floor_threshold
        self.floor_level = config.floor_level

        self._iters = 0

        # initialize relation graph
        self.rel_graph = relation_graph.RelationGraph()
        self.relation_graph_conf_threshold = config.relation_graph_conf_threshold

        self.store_raw_images = config.store_raw_images
        self.images_to_store = config.images_to_store
        self.use_gpt = config.use_gpt
        if self.store_raw_images:
            self.image_map = collections.defaultdict(collections.deque)
            self.checked_image_map = []


    def reset(self):
        # Reset value map
        if self.layered:
            self.feature_map = torch.zeros((self.n_cells, self.n_cells, self.n_layers, self.feature_dim), dtype=torch.float32).to(
                self.map_device)
        else:
            self.feature_map = torch.zeros((self.n_cells, self.n_cells, self.feature_dim), dtype=torch.float32).to(
                self.map_device)

        # Reset obstacle map
        self.obstacle_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)
        if self.layered:
            self.obstacle_map_layered = torch.zeros((self.n_cells, self.n_cells, self.n_layers), dtype=torch.float32).to(self.map_device)

        # Reset navigable map
        self.navigable_map = np.ones((self.n_cells, self.n_cells), dtype=bool)
        self.occluded_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        # Reset fully explored map
        self.fully_explored_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        # Reset checked map
        self.checked_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)

        # Reset confidence map
        if self.layered:
            self.confidence_map_feats = torch.zeros((self.n_cells, self.n_cells, self.n_layers), dtype=torch.float32).to(self.map_device)
        self.confidence_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)

        # Reset checked confidence map
        self.checked_conf_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32).to(self.map_device)

        # Reset updated mask
        if self.layered:
            self.updated_mask = torch.zeros((self.n_cells, self.n_cells, self.n_layers), dtype=torch.bool).to(self.map_device)
        else:
            self.updated_mask = torch.zeros((self.n_cells, self.n_cells), dtype=torch.bool).to(self.map_device)

        # Reset iteration counter
        self._iters = 0
        self.agent_height_0 = None

        # reset relation graph
        self.rel_graph = relation_graph.RelationGraph()

        if self.store_raw_images:
            self.image_map = collections.defaultdict(collections.deque)
            self.checked_image_map = []

    def reset_updated_mask(self):
        if self.layered:
            self.updated_mask = torch.zeros((self.n_cells, self.n_cells, self.n_layers), dtype=torch.bool).to(self.map_device)
        else:
            self.updated_mask = torch.zeros((self.n_cells, self.n_cells), dtype=torch.bool).to(self.map_device)

    def reset_checked_map(self):
        self.checked_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        self.checked_conf_map = torch.zeros((self.n_cells, self.n_cells), dtype=torch.float32)

        # Reset navigable map
        # self.navigable_map = np.ones((self.n_cells, self.n_cells), dtype=bool)
        # self.occluded_map = np.zeros((self.n_cells, self.n_cells), dtype=bool)

    def reset_checked_image_map(self):
        if self.store_raw_images:
            self.checked_image_map = []

    def set_camera_matrix(self,
                          camera_matrix: np.ndarray
                          ) -> None:
        """
        Sets the camera matrix for the map
        :param camera_matrix: 3x3 numpy array representing the camera matrix
        :return:
        """
        self.camera_initialized = True
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

    def update(self,
               values: torch.Tensor,
               depth: np.ndarray,
               tf_camera_to_episodic: np.ndarray,
               artifical_obstacles: Optional[List[Tuple[float]]] = None,
               query_text_features: Optional[torch.Tensor] = None,
               agent_pos: Optional[list[int]] = None,
               raw_image: Optional[np.ndarray] = None,
               ) -> None:
        """
        Updates the map with values by projecting them into the map from depth
        :param values: torch tensor of values. Either a 3D array of shape (feature_dim, hf, wf)
                        or a 1D array of shape (feature_dim)
        :param depth:  numpy array of depth values of shape (h, w)
        :param tf_camera_to_episodic: 4x4 numpy array representing the transformation from camera to episodic
        """
        assert values.shape[0] == self.feature_dim
        if not self.camera_initialized:
            warnings.warn("Camera matrix must be set before updating the map")
            return
        if self.agent_height_0 is None:
            self.agent_height_0 = tf_camera_to_episodic[2, 3] / tf_camera_to_episodic[3, 3]
        if len(values.shape) == 1: #or (values.shape[-1] == 1 and values.shape[-2] == 1):
            confidences_mapped, values_mapped = self.project_single(values, depth,
                                                                    tf_camera_to_episodic, self.fx, self.fy,
                                                                    self.cx, self.cy)
            self.fuse_maps(confidences_mapped, values_mapped, obstacle_mapped, obstcl_confidence_mapped, artifical_obstacles)
        elif len(values.shape) == 3:
            values = values.permute(1, 2, 0)  # feature_dim last for convenience
            if self.layered:
                (confidences_mapped, values_mapped, 
                obstacle_mapped, obstcl_confidence_mapped) = self.project_dense_layered(values, torch.Tensor(depth).to("cuda"),
                                                                                torch.tensor(tf_camera_to_episodic),
                                                                                self.fx, self.fy,
                                                                                self.cx, self.cy, raw_image)
                self.fuse_maps_layered(confidences_mapped, values_mapped, obstacle_mapped, obstcl_confidence_mapped, artifical_obstacles, query_text_features, agent_pos)
            elif (values.shape[0] == 1 and values.shape[1] == 1):
                (confidences_mapped, values_mapped,
                obstacle_mapped, obstcl_confidence_mapped) = self.project_dense_vlfm(values, torch.Tensor(depth).to("cuda"),
                                                                                torch.tensor(tf_camera_to_episodic),
                                                                                self.fx, self.fy,
                                                                                self.cx, self.cy)
                self.fuse_maps(confidences_mapped, values_mapped, obstacle_mapped, obstcl_confidence_mapped, artifical_obstacles)
            else:
                (confidences_mapped, values_mapped,
                obstacle_mapped, obstcl_confidence_mapped) = self.project_dense(values, torch.Tensor(depth).to("cuda"),
                                                                                torch.tensor(tf_camera_to_episodic),
                                                                                self.fx, self.fy,
                                                                                self.cx, self.cy)
                self.fuse_maps(confidences_mapped, values_mapped, obstacle_mapped, obstcl_confidence_mapped, artifical_obstacles)
        else:
            raise Exception("Provided Value observation of unsupported format")
        
        if self.store_raw_images:
            # store image at the location from where it was observed
            # store only latest few images
            if (agent_pos[0], agent_pos[1]) in self.image_map and len(self.image_map[(agent_pos[0], agent_pos[1])]) >= self.images_to_store:
                self.image_map[(agent_pos[0], agent_pos[1])].popleft()
            self.image_map[(agent_pos[0], agent_pos[1])].append(raw_image)   # keep appending all images for now

    def fuse_maps_layered(self,
                  confidences_mapped: torch.Tensor,
                  values_mapped: torch.Tensor,
                  obstacle_mapped: torch.Tensor,
                  obstcl_confidence_mapped: torch.Tensor,
                  artifical_obstacles: Optional[List[Tuple[float]]] = None,
                  query_text_features: Optional[torch.Tensor] = None,
                  agent_pos: Optional[list[int]] = None,
                  ) -> None:
        """
        Fuses the mapped values into the value map using the confidence estimates and tracked confidences
        This function takes in sparse tensors of confidences and values, and fuses them into the map, only updating
        the cells that have been updated.
        :param confidences_mapped: torch: sparse COO tensor of confidences
        :param values_mapped: torchL sparse COO tensor of values
        :return:
        """
        if self.fusion_type == FusionType.EMA:
            indices = confidences_mapped.indices()
            indices = torch.clamp(indices, max=self.n_cells-1)
            indices_obstacle = obstacle_mapped.indices()
            confs_new = confidences_mapped.values().data.squeeze()
            confs_old = self.confidence_map_feats[indices[0], indices[1], indices[2]]

            confs_old_obs = self.confidence_map[indices_obstacle[0], indices_obstacle[1]]

            confidence_denominator = confs_new + confs_old
            weight_1 = torch.nan_to_num(confs_old / confidence_denominator)
            weight_2 = torch.nan_to_num(confs_new / confidence_denominator)

            self.updated_mask[indices_obstacle[0], indices_obstacle[1], :] = True

            self.feature_map[indices[0], indices[1], indices[2], :] = self.feature_map[indices[0], indices[1], indices[2], :] * weight_1.unsqueeze(-1) + \
                                                       values_mapped.values().data * weight_2.unsqueeze(-1)
            self.obstacle_map_layered[indices[0], indices[1], indices[2]] = 1

            self.confidence_map_feats[indices[0], indices[1], indices[2]] = confidence_denominator

            # we also need to update the checked confidence
            confs_old_checked = self.checked_conf_map[indices[0], indices[1]]
            confidence_denominator_checked = confs_new + confs_old_checked
            self.checked_conf_map[indices[0], indices[1]] = confidence_denominator_checked

            # Obstacle Map update
            confs_new = obstcl_confidence_mapped.values().data.squeeze()
            confidence_denominator = confs_new + confs_old_obs
            weight_1 = torch.nan_to_num(confs_old_obs / confidence_denominator)
            weight_2 = torch.nan_to_num(confs_new / confidence_denominator)

            self.obstacle_map[indices_obstacle[0], indices_obstacle[1]] = self.obstacle_map[
                                                                              indices_obstacle[0], indices_obstacle[
                                                                                  1]] * weight_1 + \
                                                                          obstacle_mapped.values().data.squeeze() * weight_2

            self.confidence_map[indices_obstacle[0], indices_obstacle[1]] = confidence_denominator

            self.occluded_map = (self.obstacle_map > self.obstacle_map_threshold).cpu().numpy()
            if artifical_obstacles is not None:
                for obs in artifical_obstacles:
                    try:
                        self.occluded_map[obs[0], obs[1]] = True
                    except:
                        pass
            self.navigable_map = 1 - cv2.dilate((self.occluded_map).astype(np.uint8),
                                                self.navigable_kernel, iterations=1).astype(bool)

            self.fully_explored_map = (np.nan_to_num(1.0 / self.confidence_map.cpu().numpy())
                                       < self.fully_explored_threshold)

            self.checked_map = (np.nan_to_num(1.0 / self.checked_conf_map.cpu().numpy())
                                < self.checked_map_threshold)

            # update relation graph for the whole map
            self.update_relation_graph(indices.cpu().numpy(), query_text_features, agent_pos)

    def fuse_maps(self,
                  confidences_mapped: torch.Tensor,
                  values_mapped: torch.Tensor,
                  obstacle_mapped: torch.Tensor,
                  obstcl_confidence_mapped: torch.Tensor,
                  artifical_obstacles: Optional[List[Tuple[float]]] = None
                  ) -> None:
        """
        Fuses the mapped values into the value map using the confidence estimates and tracked confidences
        This function takes in sparse tensors of confidences and values, and fuses them into the map, only updating
        the cells that have been updated.
        :param confidences_mapped: torch: sparse COO tensor of confidences
        :param values_mapped: torchL sparse COO tensor of values
        :return:
        """
        if self.fusion_type == FusionType.EMA:
            indices = confidences_mapped.indices()
            indices_obstacle = obstacle_mapped.indices()
            confs_new = confidences_mapped.values().data.squeeze()
            confs_old = self.confidence_map[indices[0], indices[1]]

            confs_old_obs = self.confidence_map[indices_obstacle[0], indices_obstacle[1]]

            confidence_denominator = confs_new + confs_old
            weight_1 = torch.nan_to_num(confs_old / confidence_denominator)
            weight_2 = torch.nan_to_num(confs_new / confidence_denominator)

            self.updated_mask[indices[0], indices[1]] = True

            if self.layered:
                self.feature_map[indices[0], indices[1], :] = self.feature_map[indices[0], indices[1], :] * weight_1.unsqueeze(-1) + \
                                                       values_mapped.values().data * weight_2.unsqueeze(-1)
            else:
                self.feature_map[indices[0], indices[1]] = self.feature_map[indices[0], indices[1]] * weight_1.unsqueeze(-1) + \
                                                       values_mapped.values().data * weight_2.unsqueeze(-1)

            self.confidence_map[indices[0], indices[1]] = confidence_denominator

            # we also need to update the checked confidence
            confs_old_checked = self.checked_conf_map[indices[0], indices[1]]
            confidence_denominator_checked = confs_new + confs_old_checked
            self.checked_conf_map[indices[0], indices[1]] = confidence_denominator_checked

            # Obstacle Map update
            confs_new = obstcl_confidence_mapped.values().data.squeeze()
            confidence_denominator = confs_new + confs_old_obs
            weight_1 = torch.nan_to_num(confs_old_obs / confidence_denominator)
            weight_2 = torch.nan_to_num(confs_new / confidence_denominator)

            self.obstacle_map[indices_obstacle[0], indices_obstacle[1]] = self.obstacle_map[
                                                                              indices_obstacle[0], indices_obstacle[
                                                                                  1]] * weight_1 + \
                                                                          obstacle_mapped.values().data.squeeze() * weight_2

            self.occluded_map = (self.obstacle_map > self.obstacle_map_threshold).cpu().numpy()
            if artifical_obstacles is not None:
                for obs in artifical_obstacles:
                    self.occluded_map[obs[0], obs[1]] = True
            self.navigable_map = 1 - cv2.dilate((self.occluded_map).astype(np.uint8),
                                                self.navigable_kernel, iterations=1).astype(bool)

            self.fully_explored_map = (np.nan_to_num(1.0 / self.confidence_map.cpu().numpy())
                                       < self.fully_explored_threshold)

            self.checked_map = (np.nan_to_num(1.0 / self.checked_conf_map.cpu().numpy())
                                < self.checked_map_threshold)
        else:
            indices = confidences_mapped.indices()
            indices_obstacle = obstacle_mapped.indices()

            self.updated_mask[indices[0], indices[1]] = True
            self.feature_map[indices[0], indices[1]] = (self.feature_map[indices[0], indices[1]] + values_mapped.values().data)/2

            # Obstacle Map update
            obs_max = torch.maximum(self.obstacle_map[indices_obstacle[0], indices_obstacle[1]], obstacle_mapped.values().data.squeeze())
            if len(obs_max) > 0:
                self.obstacle_map[indices_obstacle[0], indices_obstacle[1]] = obs_max[0]

            self.occluded_map = (self.obstacle_map > self.obstacle_map_threshold).cpu().numpy()
            if artifical_obstacles is not None:
                for obs in artifical_obstacles:
                    self.occluded_map[obs[0], obs[1]] = True
            self.navigable_map = 1 - cv2.dilate((self.occluded_map).astype(np.uint8),
                                                self.navigable_kernel, iterations=1).astype(bool)

            self.fully_explored_map = (self.obstacle_map > self.fully_explored_threshold).cpu().numpy()
            self.checked_map = (self.obstacle_map > self.checked_map_threshold).cpu().numpy()

    def neighbors_from_bounds(self, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi,
                          center: list = None, drop_center=True):

        # clamp to grid
        xs = np.arange(max(0, x_lo), min(self.n_cells-1, x_hi) + 1)
        ys = np.arange(max(0, y_lo), min(self.n_cells-1, y_hi) + 1)
        zs = np.arange(max(0, z_lo), min(self.n_layers-1, z_hi) + 1)

        # Cartesian product of indices
        nbrs = np.array(np.meshgrid(xs, ys, zs, indexing='ij')).reshape(3, -1).T

        # Remove the center only if it’s uniquely defined (odd-length spans)
        if drop_center:
            x, y, z = center
            mask_center = ~((nbrs[:,0]==x) & (nbrs[:,1]==y) & (nbrs[:,2]==z))
            nbrs = nbrs[mask_center]

        return nbrs

    def get_neighbor_indices(self, x, y, z, _boundaries):

        xs = np.arange(max(x-_boundaries[0], 0), min(x+_boundaries[0], self.n_cells-1)+1)
        ys = np.arange(max(y-_boundaries[2], 0), min(y+_boundaries[1], self.n_cells-1)+1)
        zs = np.arange(max(z-_boundaries[2], 0), min(z+_boundaries[2], self.n_layers-1)+1)

        # all 8 neighbors in the same layer + center
        neighbors = np.array(np.meshgrid(xs, ys, zs, indexing='ij')).reshape(3, -1).T

        # remove the center (x,y,z)
        mask_center = ~((neighbors[:,0]==x) & (neighbors[:,1]==y) & (neighbors[:,2]==z))
        neighbors = neighbors[mask_center]
        return neighbors

    def update_relation_graph(
        self,
        updated_indices: np.ndarray,
        query_text_features: Optional[torch.Tensor] = None,
        agent_pos: Optional[list[int]] = None,
    ):

        if query_text_features.shape[0] <= 2 or updated_indices.shape[-1] == 0:
            return

        agent_pos = agent_pos.astype(np.int64)
        node_query_features = query_text_features[:2, :]

        # update only for the updated indices
        x_dim, y_dim, z_dim = updated_indices[0], updated_indices[1], updated_indices[2]

        # compute similarity scores and check with threhold
        similarity_map = torch.einsum('hwlc, nc -> hwln', self.feature_map, node_query_features)
        similarity_map = (similarity_map + 1.0) / 2.0   # map to [0-1]

        # find most promising areas in the map
        for i in range(2):
            # we do binary relations

            # find topk similarity score values
            similarity = similarity_map[x_dim, y_dim, z_dim, i].topk(k=2)
            for sim in similarity[0]:
                # find x,y,z
                s = similarity_map[:, :, :, i] == sim
                x, y, z = torch.nonzero(s)[0]
                x, y, z = x.item(), y.item(), z.item()

                if sim.item() >= self.relation_graph_conf_threshold:
                    node_info = {
                        "vis_feats": self.feature_map[x,y,z].cpu(),
                        "map_location": (x,y,z),
                        "map_extent": [[x,y,z]]
                    }
                    created = False

                    # check existing nodes in the graph to get relation
                    other_nodes, has_nodes = self.rel_graph.has_nodes()
                    if not has_nodes:
                        parent_node_id, _ = self.rel_graph.add_landmark(node_info)
                        created = True
                        continue

                    xy_span = int(np.ceil(1.0 * 100 * self.cell_size))
                    for other in other_nodes:
                        # find relation based on map location
                        other_node_id = other[0]
                        other_map_loc = other[1]["info"]["map_location"]
                        xy_dist = np.linalg.norm(other_map_loc[:2] - np.array([x,y]))
                        z_dist = abs(z - other_map_loc[2])
                        if xy_dist > xy_span or z_dist > 1:
                            continue

                        # check if it's the same object
                        sim_other_node = torch.einsum('c, c -> ', self.feature_map[x,y,z].cpu(), other[1]["info"]["vis_feats"])
                        sim_other_node = (sim_other_node + 1.0) / 2.0
                        if sim_other_node.item() > 0.8:
                            # update the other object with map_extent
                            self.rel_graph.update_landmark_extent(other_node_id, map_extent=[x,y,z], vis_feats=self.feature_map[x,y,z].cpu())
                            created = True
                            # merge nodes into one if they are very similar - mark the expanse as an indicator of size?
                            # self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"self": True})
                            break

                    if not created:
                        parent_node_id, _ = self.rel_graph.add_landmark(node_info)
                        created = True

                        for other in other_nodes:
                            # find relation based on map location
                            other_node_id = other[0]
                            other_map_loc = other[1]["info"]["map_location"]
                            xy_dist = np.linalg.norm(other_map_loc[:2] - np.array([x,y]))
                            z_dist = abs(z - other_map_loc[2])
                            if xy_dist > xy_span or z_dist > 1:
                                continue

                            # objects are nearby - figure out the relation
                            self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"near": True})
                            if z > other_map_loc[2]:
                                self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"below": True})
                            elif z < other_map_loc[2]:
                                self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"above": True})
                                self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"on": True}) 
                            else:
                                # same z
                                xy_span_strict = int(np.ceil(0.5 * 100 * self.cell_size))
                                if xy_dist <= xy_span_strict:
                                    self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"next to": True})

                                # figure out ego left/right
                                curr_obj_extent = np.array(self.rel_graph.get_landmark_extent(parent_node_id))
                                other_obj_extent = np.array(self.rel_graph.get_landmark_extent(other_node_id))
                                # find locations relative to the agent pos
                                curr_obj_extent[:, :2] -= agent_pos[:2]
                                A_min, A_max = curr_obj_extent[:,:2].min(axis=0), curr_obj_extent[:,:2].max(axis=0)
                                other_obj_extent[:, :2] -= agent_pos[:2]
                                B_min, B_max = other_obj_extent[:,:2].min(axis=0), other_obj_extent[:,:2].max(axis=0)
                                
                                eps = 0.0
                                if A_min[0] == A_max[0]:
                                    if (A_min[0] + eps) < B_min[0]:
                                        self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"left": True})
                                    else:
                                        self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"right": True})
                                else:
                                    cxB = (B_min[0] + B_max[0]) * 0.5
                                    left_part = max(0.0, min(cxB, A_max[0]) - A_min[0])
                                    wA = max(1e-8, A_max[0] - A_min[0])
                                    if (left_part / wA) >= 0.7:
                                        self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"left": True})
                                    else:
                                        right_part = max(0.0, A_max[0] - max(cxB, A_min[0]))
                                        wA = max(1e-8, A_max[0] - A_min[0])
                                        if (right_part / wA) >= 0.7:
                                            self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"right": True})

                                if ((A_min[0] >= B_min[0] - eps) and (A_min[1] >= B_min[1] - eps) and (A_max[0] <= B_max[0] + eps) and (A_max[1] <= B_max[1] + eps)):
                                    self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"inside": True})
                                    
                                    # when projected on a topdown, on might look like inside
                                    self.rel_graph.add_relation(from_node_id=parent_node_id, to_node_id=other_node_id, relation_info={"on": True})  

    def rows_not_in(self, a: np.ndarray, b: np.ndarray, return_index: bool = False):
        """
        Return rows of `a` that are NOT present as rows in `b`.
        Works for integer or exact-equality float comparisons.
        """
        if a.shape[1] != b.shape[1]:
            raise ValueError("a and b must have the same number of columns")

        def _asvoid(x):
            x = np.ascontiguousarray(x)
            return x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1]))).ravel()

        av = _asvoid(a)
        bv = _asvoid(b)
        mask = ~np.isin(av, bv)
        if return_index:
            idx = np.nonzero(mask)[0]
            return a[idx], idx
        return a[mask]

    @torch.no_grad()
    # @torch.compile
    def project_dense_vlfm(self,
                      values: torch.Tensor,
                      depth: torch.Tensor,
                      tf_camera_to_episodic: torch.Tensor,
                      fx, fy, cx, cy
                      ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Projects the dense features into the map
        TODO We could get rid of sparse tensors entirely and instead use arrays of indices and values to reduce overhead
        :param values: torch tensor of values, shape (hf, wf, feature_dim)
        :param depth: torch tensor of depth values, shape (h, w)
        :param tf_camera_to_episodic:
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :return: (confidences_mapped, values_mapped, obstacle_mapped, obstcl_confidence_mapped), sparse COO tensor in map coordinates
        """
        # check if values is on cuda
        if not values.is_cuda:
            print("Warning: Provided value array is not on cuda, which it should be as an output of a model. Moving to "
                  "Cuda, which will slow things down.")
            values = values.to("cuda")
        if not depth.is_cuda:
            print(
                "Warning: Provided depth array is not on cuda, which it could be if is an output of a model. Moving to "
                "Cuda, which will slow things down.")
            depth = depth.to("cuda")

        if values.shape[0:2] == depth.shape[0:2]:
            # our values align with the depth pixels
            depth_aligned = depth
        else:
            # our values are to be considered "patch wise" where we need to project each patch, by averaging the
            # depth values within that patch
            if self.dense_projection == DenseProjectionType.SUBSAMPLE:
                nh = values.shape[0]
                nw = values.shape[1]
                h = depth.shape[0]
                w = depth.shape[1]
                # TODO: this is possibly inaccurate, the patch_size might not add up and introduce errors
                patch_size_h = ceildiv(h, nh)
                patch_size_w = ceildiv(w, nw)

                pad_h = patch_size_h * nh - h
                pad_w = patch_size_w * nw - w
                pad_h_before = pad_h // 2
                pad_h_after = pad_h - pad_h_before
                pad_w_before = pad_w // 2
                pad_w_after = pad_w - pad_w_before

                depth_padded = np.pad(depth, ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)))
                depth_aligned = depth_padded.reshape(nh, patch_size_h, nw, patch_size_w).mean(axis=(1, 3))
            elif self.dense_projection == DenseProjectionType.INTERPOLATE:
                values = torch.nn.functional.interpolate(values.permute(2, 0, 1).unsqueeze(0),
                                                         size=depth.shape,
                                                         mode='bilinear',
                                                         align_corners=False).squeeze(0).permute(1, 2, 0)
                depth_aligned = depth
            else:
                raise Exception("Unsupported Dense Projection Mode.")

        # TODO this will be wrong for sub-sampled as e.g. fx will be wrong
        depth_image_smoothed = depth_aligned

        mask = depth_image_smoothed == float('inf')
        depth_image_smoothed[mask] = depth_image_smoothed[~mask].max()
        kernel_size = 11
        pad = kernel_size // 2

        depth_image_smoothed = -torch.nn.functional.max_pool2d(-depth_image_smoothed.unsqueeze(0), kernel_size,
                                                               padding=pad,
                                                               stride=1).squeeze(0)
        # depth_image_smoothed = F.gaussian_blur(depth_image_smoothed, [31, 31], sigma=4.0)
        # TODO Gaussian Blur temporarily disabled
        dx = torch.gradient(depth_image_smoothed, dim=1)[0] / (fx / depth.shape[1])
        dy = torch.gradient(depth_image_smoothed, dim=0)[0] / (fy / depth.shape[0])
        gradient_magnitude = torch.sqrt(dx ** 2 + dy ** 2)
        gradient_magnitude = torch.nn.functional.max_pool2d(gradient_magnitude.unsqueeze(0), 11, stride=1,
                                                            padding=5).squeeze(0)
        scores = ((1 - torch.tanh(gradient_magnitude * self.gradient_factor)) *
                  torch.exp(-((self.optimal_object_distance - depth) / self.optimal_object_factor) ** 2 / 3.0))
        scores_aligned = scores.reshape(-1)

        projected_depth, hole_mask = self.project_depth_camera(depth_aligned, (depth.shape[0], depth.shape[1]), fx,
                                                    fy, cx, cy)

        rotated_pcl = rotate_pcl(projected_depth, tf_camera_to_episodic)
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
        rotated_pcl[:, :2] += torch.tensor([cam_x, cam_y], device='cuda')

        values_aligned = values.reshape((-1, values.shape[-1]))

        pcl_grid_ids = torch.floor(rotated_pcl[:, :2] / self.cell_size).to(torch.int32)
        pcl_grid_ids[:, 0] += self.map_center_cells[0]
        pcl_grid_ids[:, 1] += self.map_center_cells[1]

        # Filter valid updates
        mask = (depth_aligned.flatten() != float('inf')) & (depth_aligned.flatten() != 0) & (pcl_grid_ids[:, 0] >= self.kernel_half + 1) & (
                pcl_grid_ids[:, 0] < self.n_cells - self.kernel_half - 1) & (
                       pcl_grid_ids[:, 1] >= self.kernel_half + 1) & (
                       pcl_grid_ids[:, 1] < self.n_cells - self.kernel_half - 1)  # for value map
        if hole_mask.nelement() == 0:
            mask_obstacle = mask & (((rotated_pcl[:, 2]> self.obstacle_min) & (
                                         rotated_pcl[:, 2]  < self.obstacle_max)) )
        else:
            mask_obstacle = mask & (((rotated_pcl[:, 2] > self.obstacle_min) & (
                    rotated_pcl[:, 2] < self.obstacle_max)) | hole_mask)
        mask &= (scores_aligned > 1e-5)
        mask_obstacle_masked = mask_obstacle[mask]
        scores_masked = scores_aligned[mask]

        pcl_grid_ids_masked = pcl_grid_ids[mask].T
        values_to_add = values_aligned[mask] * scores_masked.unsqueeze(1)

        combined_data = torch.cat((
            values_to_add,
            mask_obstacle_masked.unsqueeze(1),
            torch.ones((values_to_add.shape[0], 1), dtype=torch.uint8, device="cuda"),
            scores_masked.unsqueeze(1)),
            dim=1)  # prepare to aggregate doubles (values pointing to the same grid cell)

        # define the map from unique ids to all ids
        pcl_grid_ids_masked_unique, pcl_mapping = pcl_grid_ids_masked.unique(dim=1, return_inverse=True)
        # coalesce the data
        coalesced_combined_data = torch.zeros((pcl_grid_ids_masked_unique.shape[1], combined_data.shape[-1]),
                                              dtype=torch.float32, device="cuda")
        coalesced_combined_data.index_add_(0, pcl_mapping, combined_data)

        # Extract the data
        data_dim = combined_data.shape[-1]
        obstacle_mapped = coalesced_combined_data[:, data_dim - 3]
        scores_mapped = coalesced_combined_data[:, data_dim - 1].unsqueeze(1)
        sums_per_cell = coalesced_combined_data[:, data_dim - 2].unsqueeze(1)
        new_map = coalesced_combined_data[:, :data_dim - 3]

        # Normalize (from sum to mean)
        new_map /= scores_mapped
        scores_mapped /= sums_per_cell
        obstcl_confidence_mapped = scores_mapped

        # Get all the ids that are affected by the kernel (depth noise blurring)
        ids = pcl_grid_ids_masked_unique
        all_ids_ = torch.zeros((2, ids.shape[1], self.kernel_size, self.kernel_size), device="cuda")
        all_ids_[0] = (ids[0].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_x)
        all_ids_[1] = (ids[1].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_y)
        all_ids, mapping = all_ids_.reshape(2, -1).unique(dim=1, return_inverse=True)

        coalesced_map_data = torch.zeros((all_ids.shape[1], self.feature_dim), dtype=torch.float32, device="cuda")
        coalesced_scores = torch.zeros((all_ids.shape[1], 1), dtype=torch.float32, device="cuda")
        # Compute the blurred map and blurred scores
        coalesced_map_data.index_add_(0, mapping, (new_map.unsqueeze(1).unsqueeze(1)).reshape(-1, self.feature_dim))
        coalesced_scores.index_add_(0, mapping, (scores_mapped.unsqueeze(1)).reshape(-1, 1))

        # Free up memory to avoid OOM
        torch.cuda.empty_cache()

        # Compute the obstacle map
        obstacle_mapped[:] = (obstacle_mapped > 0).to(torch.float32)

        obstacle_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, obstacle_mapped.unsqueeze(1), (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu()
        obstcl_confidence_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, obstcl_confidence_mapped, (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu()
        # print("Updating with sparse matrix of size {}x{} with {} non-zero elements, resulting size is {} Mb".format(
        #     self.n_cells, self.n_cells, new_map.values().shape[0] * self.feature_dim,
        #                                 new_map.element_size() * new_map.values().shape[
        #                                     0] * self.feature_dim / 1024 / 1024))
        return torch.sparse_coo_tensor(all_ids, coalesced_scores, (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu(), torch.sparse_coo_tensor(all_ids, coalesced_map_data, (self.n_cells, self.n_cells, self.feature_dim), is_coalesced=True).cpu(), obstacle_mapped.cpu(), obstcl_confidence_mapped.cpu()

    @torch.no_grad()
    # @torch.compile
    def project_dense(self,
                      values: torch.Tensor,
                      depth: torch.Tensor,
                      tf_camera_to_episodic: torch.Tensor,
                      fx, fy, cx, cy
                      ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Projects the dense features into the map
        TODO We could get rid of sparse tensors entirely and instead use arrays of indices and values to reduce overhead
        :param values: torch tensor of values, shape (hf, wf, feature_dim)
        :param depth: torch tensor of depth values, shape (h, w)
        :param tf_camera_to_episodic:
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :return: (confidences_mapped, values_mapped, obstacle_mapped, obstcl_confidence_mapped), sparse COO tensor in map coordinates
        """
        # check if values is on cuda
        if not values.is_cuda:
            print("Warning: Provided value array is not on cuda, which it should be as an output of a model. Moving to "
                  "Cuda, which will slow things down.")
            values = values.to("cuda")
        if not depth.is_cuda:
            print(
                "Warning: Provided depth array is not on cuda, which it could be if is an output of a model. Moving to "
                "Cuda, which will slow things down.")
            depth = depth.to("cuda")

        if values.shape[0:2] == depth.shape[0:2]:
            # our values align with the depth pixels
            depth_aligned = depth
        else:
            # our values are to be considered "patch wise" where we need to project each patch, by averaging the
            # depth values within that patch
            if self.dense_projection == DenseProjectionType.SUBSAMPLE:
                nh = values.shape[0]
                nw = values.shape[1]
                h = depth.shape[0]
                w = depth.shape[1]
                # TODO: this is possibly inaccurate, the patch_size might not add up and introduce errors
                patch_size_h = ceildiv(h, nh)
                patch_size_w = ceildiv(w, nw)

                pad_h = patch_size_h * nh - h
                pad_w = patch_size_w * nw - w
                pad_h_before = pad_h // 2
                pad_h_after = pad_h - pad_h_before
                pad_w_before = pad_w // 2
                pad_w_after = pad_w - pad_w_before

                depth_padded = np.pad(depth, ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)))
                depth_aligned = depth_padded.reshape(nh, patch_size_h, nw, patch_size_w).mean(axis=(1, 3))
            elif self.dense_projection == DenseProjectionType.INTERPOLATE:
                values = torch.nn.functional.interpolate(values.permute(2, 0, 1).unsqueeze(0),
                                                         size=depth.shape,
                                                         mode='bilinear',
                                                         align_corners=False).squeeze(0).permute(1, 2, 0)
                depth_aligned = depth
            else:
                raise Exception("Unsupported Dense Projection Mode.")

        # TODO this will be wrong for sub-sampled as e.g. fx will be wrong
        depth_image_smoothed = depth_aligned

        mask = depth_image_smoothed == float('inf')
        depth_image_smoothed[mask] = depth_image_smoothed[~mask].max()
        kernel_size = 11
        pad = kernel_size // 2

        depth_image_smoothed = -torch.nn.functional.max_pool2d(-depth_image_smoothed.unsqueeze(0), kernel_size,
                                                               padding=pad,
                                                               stride=1).squeeze(0)
        # depth_image_smoothed = F.gaussian_blur(depth_image_smoothed, [31, 31], sigma=4.0)
        # TODO Gaussian Blur temporarily disabled
        dx = torch.gradient(depth_image_smoothed, dim=1)[0] / (fx / depth.shape[1])
        dy = torch.gradient(depth_image_smoothed, dim=0)[0] / (fy / depth.shape[0])
        gradient_magnitude = torch.sqrt(dx ** 2 + dy ** 2)
        gradient_magnitude = torch.nn.functional.max_pool2d(gradient_magnitude.unsqueeze(0), 11, stride=1,
                                                            padding=5).squeeze(0)
        scores = ((1 - torch.tanh(gradient_magnitude * self.gradient_factor)) *
                  torch.exp(-((self.optimal_object_distance - depth) / self.optimal_object_factor) ** 2 / 3.0))
        scores_aligned = scores.reshape(-1)

        projected_depth, hole_mask = self.project_depth_camera(depth_aligned, (depth.shape[0], depth.shape[1]), fx,
                                                    fy, cx, cy)

        rotated_pcl = rotate_pcl(projected_depth, tf_camera_to_episodic)
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
        rotated_pcl[:, :2] += torch.tensor([cam_x, cam_y], device='cuda')

        values_aligned = values.reshape((-1, values.shape[-1]))

        pcl_grid_ids = torch.floor(rotated_pcl[:, :2] / self.cell_size).to(torch.int32)
        pcl_grid_ids[:, 0] += self.map_center_cells[0]
        pcl_grid_ids[:, 1] += self.map_center_cells[1]

        # Filter valid updates
        mask = (depth_aligned.flatten() != float('inf')) & (depth_aligned.flatten() != 0) & (pcl_grid_ids[:, 0] >= self.kernel_half + 1) & (
                pcl_grid_ids[:, 0] < self.n_cells - self.kernel_half - 1) & (
                       pcl_grid_ids[:, 1] >= self.kernel_half + 1) & (
                       pcl_grid_ids[:, 1] < self.n_cells - self.kernel_half - 1)  # for value map
        if hole_mask.nelement() == 0:
            mask_obstacle = mask & (((rotated_pcl[:, 2]> self.obstacle_min) & (
                                         rotated_pcl[:, 2]  < self.obstacle_max)) )
        else:
            mask_obstacle = mask & (((rotated_pcl[:, 2] > self.obstacle_min) & (
                    rotated_pcl[:, 2] < self.obstacle_max)) | hole_mask)
        mask &= (scores_aligned > 1e-5)
        mask_obstacle_masked = mask_obstacle[mask]
        scores_masked = scores_aligned[mask]

        pcl_grid_ids_masked = pcl_grid_ids[mask].T
        values_to_add = values_aligned[mask] * scores_masked.unsqueeze(1)

        combined_data = torch.cat((
            values_to_add,
            mask_obstacle_masked.unsqueeze(1),
            torch.ones((values_to_add.shape[0], 1), dtype=torch.uint8, device="cuda"),
            scores_masked.unsqueeze(1)),
            dim=1)  # prepare to aggregate doubles (values pointing to the same grid cell)

        # define the map from unique ids to all ids
        pcl_grid_ids_masked_unique, pcl_mapping = pcl_grid_ids_masked.unique(dim=1, return_inverse=True)
        # coalesce the data
        coalesced_combined_data = torch.zeros((pcl_grid_ids_masked_unique.shape[1], combined_data.shape[-1]),
                                              dtype=torch.float32, device="cuda")
        coalesced_combined_data.index_add_(0, pcl_mapping, combined_data)

        # Extract the data
        data_dim = combined_data.shape[-1]
        obstacle_mapped = coalesced_combined_data[:, data_dim - 3]
        scores_mapped = coalesced_combined_data[:, data_dim - 1].unsqueeze(1)
        sums_per_cell = coalesced_combined_data[:, data_dim - 2].unsqueeze(1)
        new_map = coalesced_combined_data[:, :data_dim - 3]

        # Normalize (from sum to mean)
        new_map /= scores_mapped
        scores_mapped /= sums_per_cell
        obstcl_confidence_mapped = scores_mapped

        # Get all the ids that are affected by the kernel (depth noise blurring)
        ids = pcl_grid_ids_masked_unique
        all_ids_ = torch.zeros((2, ids.shape[1], self.kernel_size, self.kernel_size), device="cuda")
        all_ids_[0] = (ids[0].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_x)
        all_ids_[1] = (ids[1].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_y)
        all_ids, mapping = all_ids_.reshape(2, -1).unique(dim=1, return_inverse=True)

        # Compute the corresponding depths
        depths = ((all_ids - self.map_center_cells.unsqueeze(1)) * self.cell_size - torch.tensor([cam_x, cam_y],
                                                                                 dtype=torch.float32, device="cuda")
                  .unsqueeze(1))

        # And the depth noise
        depth_noise = torch.sqrt(torch.sum(depths ** 2, dim=0)) * self.depth_factor / self.cell_size

        # Compute the sum for each kernel centered around a grid cell
        kernel_sums = gaussian_kernel_sum(self.kernel_components_sum, depth_noise).unsqueeze(-1)  # all unique ids

        # remap the depths to all the id's to kernels centered around the original points in ids and
        # compute the sparse inverse kernel elements
        kernels = compute_gaussian_kernel_components(self.kernel_components, depth_noise[mapping].reshape(-1,
                                                                                  self.kernel_size, self.kernel_size))

        coalesced_map_data = torch.zeros((all_ids.shape[1], self.feature_dim), dtype=torch.float32, device="cuda")
        coalesced_scores = torch.zeros((all_ids.shape[1], 1), dtype=torch.float32, device="cuda")
        # Compute the blurred map and blurred scores
        coalesced_map_data.index_add_(0, mapping, (kernels.unsqueeze(-1) *
                                                   new_map.unsqueeze(1).unsqueeze(1)).reshape(-1, self.feature_dim))
        coalesced_scores.index_add_(0, mapping, (kernels * scores_mapped.unsqueeze(1)).reshape(-1, 1))

        # Free up memory to avoid OOM
        torch.cuda.empty_cache()

        # Normalize the map and scores
        coalesced_map_data /= kernel_sums
        coalesced_scores /= kernel_sums

        # Compute the obstacle map
        obstacle_mapped[:] = (obstacle_mapped > 0).to(torch.float32)

        obstacle_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, obstacle_mapped.unsqueeze(1), (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu()
        obstcl_confidence_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, obstcl_confidence_mapped, (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu()
        # print("Updating with sparse matrix of size {}x{} with {} non-zero elements, resulting size is {} Mb".format(
        #     self.n_cells, self.n_cells, new_map.values().shape[0] * self.feature_dim,
        #                                 new_map.element_size() * new_map.values().shape[
        #                                     0] * self.feature_dim / 1024 / 1024))
        return torch.sparse_coo_tensor(all_ids, coalesced_scores, (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu(), torch.sparse_coo_tensor(all_ids, coalesced_map_data, (self.n_cells, self.n_cells, self.feature_dim), is_coalesced=True).cpu(), obstacle_mapped.cpu(), obstcl_confidence_mapped.cpu()
    
    @torch.no_grad()
    # @torch.compile
    def project_dense_layered(self,
                      values: torch.Tensor,
                      depth: torch.Tensor,
                      tf_camera_to_episodic: torch.Tensor,
                      fx, fy, cx, cy,
                      raw_image: Optional[np.ndarray] = None,
                      viz: Optional[bool] = False,
                      ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Projects the dense features into the map
        TODO We could get rid of sparse tensors entirely and instead use arrays of indices and values to reduce overhead
        :param values: torch tensor of values, shape (hf, wf, feature_dim)
        :param depth: torch tensor of depth values, shape (h, w)
        :param tf_camera_to_episodic:
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :return: (confidences_mapped, values_mapped, obstacle_mapped, obstcl_confidence_mapped), sparse COO tensor in map coordinates
        """
        # check if values is on cuda
        if not values.is_cuda:
            print("Warning: Provided value array is not on cuda, which it should be as an output of a model. Moving to "
                  "Cuda, which will slow things down.")
            values = values.to("cuda")
        if not depth.is_cuda:
            print(
                "Warning: Provided depth array is not on cuda, which it could be if is an output of a model. Moving to "
                "Cuda, which will slow things down.")
            depth = depth.to("cuda")

        if values.shape[0:2] == depth.shape[0:2]:
            # our values align with the depth pixels
            depth_aligned = depth
        else:
            # our values are to be considered "patch wise" where we need to project each patch, by averaging the
            # depth values within that patch
            if self.dense_projection == DenseProjectionType.SUBSAMPLE:
                nh = values.shape[0]
                nw = values.shape[1]
                h = depth.shape[0]
                w = depth.shape[1]
                # TODO: this is possibly inaccurate, the patch_size might not add up and introduce errors
                patch_size_h = ceildiv(h, nh)
                patch_size_w = ceildiv(w, nw)

                pad_h = patch_size_h * nh - h
                pad_w = patch_size_w * nw - w
                pad_h_before = pad_h // 2
                pad_h_after = pad_h - pad_h_before
                pad_w_before = pad_w // 2
                pad_w_after = pad_w - pad_w_before

                depth_padded = np.pad(depth, ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)))
                depth_aligned = depth_padded.reshape(nh, patch_size_h, nw, patch_size_w).mean(axis=(1, 3))
            elif self.dense_projection == DenseProjectionType.INTERPOLATE:
                values = torch.nn.functional.interpolate(values.permute(2, 0, 1).unsqueeze(0),
                                                         size=depth.shape,
                                                         mode='bilinear',
                                                         align_corners=False).squeeze(0).permute(1, 2, 0)
                depth_aligned = depth
            else:
                raise Exception("Unsupported Dense Projection Mode.")

        # TODO this will be wrong for sub-sampled as e.g. fx will be wrong
        depth_image_smoothed = depth_aligned

        mask = depth_image_smoothed == float('inf')
        depth_image_smoothed[mask] = depth_image_smoothed[~mask].max()
        kernel_size = 11
        pad = kernel_size // 2

        depth_image_smoothed = -torch.nn.functional.max_pool2d(-depth_image_smoothed.unsqueeze(0), kernel_size,
                                                               padding=pad,
                                                               stride=1).squeeze(0)
        # depth_image_smoothed = F.gaussian_blur(depth_image_smoothed, [31, 31], sigma=4.0)
        # TODO Gaussian Blur temporarily disabled
        dx = torch.gradient(depth_image_smoothed, dim=1)[0] / (fx / depth.shape[1])
        dy = torch.gradient(depth_image_smoothed, dim=0)[0] / (fy / depth.shape[0])
        gradient_magnitude = torch.sqrt(dx ** 2 + dy ** 2)
        gradient_magnitude = torch.nn.functional.max_pool2d(gradient_magnitude.unsqueeze(0), 11, stride=1,
                                                            padding=5).squeeze(0)
        scores = ((1 - torch.tanh(gradient_magnitude * self.gradient_factor)) *
                  torch.exp(-((self.optimal_object_distance - depth) / self.optimal_object_factor) ** 2 / 3.0))
        scores_aligned = scores.reshape(-1)

        projected_depth, hole_mask = self.project_depth_camera(depth_aligned, (depth.shape[0], depth.shape[1]), fx,
                                                    fy, cx, cy)

        rotated_pcl = rotate_pcl(projected_depth, tf_camera_to_episodic)
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
        rotated_pcl[:, :2] += torch.tensor([cam_x, cam_y], device='cuda')

        values_aligned = values.reshape((-1, values.shape[-1]))

        if viz:
            # save out point cloud
            H, W = depth.shape
            depth_m = depth.cpu().numpy().astype(np.float32) / float(1.0)
            jj, ii = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            Z = depth_m
            valid = Z > 0
            X = (jj - cx) * Z / fx
            Y = (ii - cy) * Z / fy
            pts_cam = np.stack([X, Y, Z], axis=-1)[valid]  # [N,3]
            points = pts_cam.astype(np.float32)
            colors = (raw_image.astype(np.float32) / 255.0)[valid]
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

            # Optional: downsample for smoother viz
            pc = pc.voxel_down_sample(voxel_size=0.01)  # meters; adjust as needed

            # Estimate normals (optional, improves shading in some viewers)
            pc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
            )

            _t = time.time()
            out_ply = f"cloud_{_t}.ply"
            o3d.io.write_point_cloud(out_ply, pc)
            print(f"Saved PLY: {out_ply}")

            Image.fromarray(raw_image).save(f"rgb_{_t}.png")

            obs_k *= 255.0/depth_m.max()
            obs_k = obs_k[..., np.newaxis]
            obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)
            Image.fromarray(obs_k).save(f"depth_{_t}.png")

            # o3d.visualization.draw_geometries([pc])

        pcl_grid_ids = torch.floor(rotated_pcl.clone()[:, :2] / self.cell_size).to(torch.int32)
        pcl_grid_ids[:, 0] += self.map_center_cells[0]
        pcl_grid_ids[:, 1] += self.map_center_cells[1]

        # layered grid
        layered_pcl_grid_ids = torch.zeros((pcl_grid_ids.shape[0], 3), dtype=pcl_grid_ids.dtype, device=pcl_grid_ids.device)
        layered_pcl_grid_ids[:,:2] = pcl_grid_ids.clone()
        layered_pcl_grid_ids[:, 2] = torch.bucketize(rotated_pcl.clone()[:, 2], boundaries=self.z_bins)

        # Filter valid updates
        mask = (depth_aligned.flatten() != float('inf')) & (depth_aligned.flatten() != 0) & (pcl_grid_ids[:, 0] >= self.kernel_half + 1) & (
                pcl_grid_ids[:, 0] < self.n_cells - self.kernel_half - 1) & (
                       pcl_grid_ids[:, 1] >= self.kernel_half + 1) & (
                       pcl_grid_ids[:, 1] < self.n_cells - self.kernel_half - 1)  # for value map
        if hole_mask.nelement() == 0:
            mask_obstacle = mask & (((rotated_pcl[:, 2]> self.obstacle_min) & (
                                         rotated_pcl[:, 2]  < self.obstacle_max)) )
        else:
            mask_obstacle = mask & (((rotated_pcl[:, 2] > self.obstacle_min) & (
                    rotated_pcl[:, 2] < self.obstacle_max)) | hole_mask)
        mask &= (scores_aligned > 1e-5)
        mask_obstacle_masked = mask_obstacle[mask]
        scores_masked = scores_aligned[mask]

        pcl_grid_ids_masked = pcl_grid_ids[mask].T
        values_to_add = values_aligned[mask] * scores_masked.unsqueeze(1)
        layered_pcl_grid_ids_masked = layered_pcl_grid_ids[mask].T

        combined_data = torch.cat((
            values_to_add,
            mask_obstacle_masked.unsqueeze(1),
            torch.ones((values_to_add.shape[0], 1), dtype=torch.uint8, device="cuda"),
            scores_masked.unsqueeze(1),
            ),
            dim=1)  # prepare to aggregate doubles (values pointing to the same grid cell)

        # define the map from unique ids to all ids
        pcl_grid_ids_masked_unique, pcl_mapping = pcl_grid_ids_masked.unique(dim=1, return_inverse=True)
        # coalesce the data
        coalesced_combined_data = torch.zeros((pcl_grid_ids_masked_unique.shape[1], combined_data.shape[-1]),
                                              dtype=torch.float32, device="cuda")
        coalesced_combined_data.index_add_(0, pcl_mapping, combined_data)

        # Extract the data
        data_dim = combined_data.shape[-1]
        obstacle_mapped = coalesced_combined_data[:, data_dim - 3]
        scores_mapped = coalesced_combined_data[:, data_dim - 1].unsqueeze(1)
        sums_per_cell = coalesced_combined_data[:, data_dim - 2].unsqueeze(1)
        new_map = coalesced_combined_data[:, :data_dim - 3]

        # Normalize (from sum to mean)
        new_map /= scores_mapped
        scores_mapped /= sums_per_cell
        obstcl_confidence_mapped = scores_mapped

        # Get all the ids that are affected by the kernel (depth noise blurring)
        ids = pcl_grid_ids_masked_unique
        all_ids_ = torch.zeros((2, ids.shape[1], self.kernel_size, self.kernel_size), device="cuda")
        all_ids_[0] = (ids[0].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_x)
        all_ids_[1] = (ids[1].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_y)
        all_ids, mapping = all_ids_.reshape(2, -1).unique(dim=1, return_inverse=True)

        ## layered ids
        pcl_grid_ids_masked_unique_layered, pcl_mapping_layered = layered_pcl_grid_ids_masked.unique(dim=1, return_inverse=True)
        coalesced_combined_data_layered = torch.zeros((pcl_grid_ids_masked_unique_layered.shape[1], combined_data.shape[-1]),
                                              dtype=torch.float32, device="cuda")
        coalesced_combined_data_layered.index_add_(0, pcl_mapping_layered, combined_data)

        # obstacle_mapped_layered = coalesced_combined_data_layered[:, data_dim - 3]
        scores_mapped_layered = coalesced_combined_data_layered[:, data_dim - 1].unsqueeze(1)
        sums_per_cell_layered = coalesced_combined_data_layered[:, data_dim - 2].unsqueeze(1)
        new_map_layered = coalesced_combined_data_layered[:, :data_dim - 3]

        new_map_layered /= scores_mapped_layered
        scores_mapped_layered /= sums_per_cell_layered

        ids_layered = pcl_grid_ids_masked_unique_layered
        all_ids_layered_ = torch.zeros((3, ids_layered.shape[1], self.kernel_size, self.kernel_size), device="cuda")
        all_ids_layered_[0] = (ids_layered[0].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_x)
        all_ids_layered_[1] = (ids_layered[1].unsqueeze(-1).unsqueeze(-1) + self.kernel_ids_y)
        all_ids_layered_[2] = ids_layered[2].unsqueeze(-1).unsqueeze(-1)
        all_ids_layered, mapping_layered = all_ids_layered_.reshape(3, -1).unique(dim=1, return_inverse=True)

        # Compute the corresponding depths
        depths = ((all_ids[:2] - self.map_center_cells.unsqueeze(1)) * self.cell_size - torch.tensor([cam_x, cam_y],
                                                                                 dtype=torch.float32, device="cuda")
                  .unsqueeze(1))

        # And the depth noise
        depth_noise = torch.sqrt(torch.sum(depths ** 2, dim=0)) * self.depth_factor / self.cell_size

        # Compute the sum for each kernel centered around a grid cell
        kernel_sums = gaussian_kernel_sum(self.kernel_components_sum, depth_noise).unsqueeze(-1)  # all unique ids

        # remap the depths to all the id's to kernels centered around the original points in ids and
        # compute the sparse inverse kernel elements
        kernels = compute_gaussian_kernel_components(self.kernel_components, depth_noise[mapping].reshape(-1,
                                                                                  self.kernel_size, self.kernel_size))

        coalesced_map_data = torch.zeros((all_ids.shape[1], self.feature_dim), dtype=torch.float32, device="cuda")
        coalesced_scores = torch.zeros((all_ids.shape[1], 1), dtype=torch.float32, device="cuda")
        # Compute the blurred map and blurred scores
        coalesced_map_data.index_add_(0, mapping, (kernels.unsqueeze(-1) *
                                                   new_map.unsqueeze(1).unsqueeze(1)).reshape(-1, self.feature_dim))
        coalesced_scores.index_add_(0, mapping, (kernels * scores_mapped.unsqueeze(1)).reshape(-1, 1))

        # Free up memory to avoid OOM
        torch.cuda.empty_cache()

        # Normalize the map and scores
        coalesced_map_data /= kernel_sums
        coalesced_scores /= kernel_sums

        # Compute the corresponding depths
        layered_centers = torch.tensor([self.map_center_cells[0], self.map_center_cells[1], self.n_layers//2], device="cuda", dtype=torch.uint32)
        depths = ((all_ids_layered - layered_centers.unsqueeze(1)) * self.cell_size - torch.tensor([cam_x, cam_y, 0.0],
                                                                                 dtype=torch.float32, device="cuda")
                  .unsqueeze(1))

        # And the depth noise
        depth_noise = torch.sqrt(torch.sum(depths ** 2, dim=0)) * self.depth_factor / self.cell_size

        # Compute the sum for each kernel centered around a grid cell
        kernel_sums = gaussian_kernel_sum(self.kernel_components_sum, depth_noise).unsqueeze(-1)  # all unique ids

        # remap the depths to all the id's to kernels centered around the original points in ids and
        # compute the sparse inverse kernel elements
        kernels = compute_gaussian_kernel_components(self.kernel_components, depth_noise[mapping_layered].reshape(-1,
                                                                                  self.kernel_size, self.kernel_size))

        ## layered map
        coalesced_map_data_layered = torch.zeros((all_ids_layered.shape[1], self.feature_dim), dtype=torch.float32, device="cuda")
        coalesced_scores_layered = torch.zeros((all_ids_layered.shape[1], 1), dtype=torch.float32, device="cuda")
        # Compute the blurred map and blurred scores
        coalesced_map_data_layered.index_add_(0, mapping_layered, (kernels.unsqueeze(-1) *
                                                   new_map_layered.unsqueeze(1).unsqueeze(1)).reshape(-1, self.feature_dim))
        coalesced_scores_layered.index_add_(0, mapping_layered, (kernels * scores_mapped_layered.unsqueeze(1)).reshape(-1, 1))

        # Free up memory to avoid OOM
        torch.cuda.empty_cache()

        # Normalize the map and scores
        coalesced_map_data_layered /= kernel_sums
        coalesced_scores_layered /= kernel_sums

        # Compute the obstacle map
        obstacle_mapped[:] = (obstacle_mapped > 0).to(torch.float32)

        obstacle_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, obstacle_mapped.unsqueeze(1), (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu()
        obstcl_confidence_mapped = torch.sparse_coo_tensor(pcl_grid_ids_masked_unique, obstcl_confidence_mapped, (self.n_cells, self.n_cells, 1), is_coalesced=True).cpu()

        return (
            torch.sparse_coo_tensor(
                all_ids_layered,
                coalesced_scores_layered,
                (self.n_cells, self.n_cells, self.n_layers, 1),
                is_coalesced=True,
            ).cpu(),
            torch.sparse_coo_tensor(
                all_ids_layered,
                coalesced_map_data_layered,
                (self.n_cells, self.n_cells, self.n_layers, self.feature_dim),
                is_coalesced=True,
            ).cpu(),
            obstacle_mapped.cpu(),
            obstcl_confidence_mapped.cpu(),
        )

    def project_single(self,
                       values: torch.Tensor,
                       depth: np.ndarray,
                       tf_camera_to_episodic,
                       fx, fy, cx, cy
                       ) -> (torch.Tensor, torch.Tensor):
        """
        Projects a single value observation into the map using a heuristic, similar to VLFM
        :param values:
        :param depth:
        :param tf_camera_to_episodic:
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :return:
        """
        projected_depth = self.project_depth_camera(depth, *(depth.shape[0:2]), fx, fy, cx, cy)
        # TODO needs to be implemented
        raise NotImplementedError

    def project_depth_camera(self,
                             depth: torch.Tensor,
                             camera_resolution: Tuple[int, int],
                             fx, fy, cx, cy
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects the depth into 3D pointcloud. Camera resolution is passed if the depth is subsampled,
        to match value array resolution.
        :param depth: torch Tensor of shape (h, w), not necessarily the same as camera resolution
        :param camera_resolution: tuple of original camera resolution to correct depth if necessary (w, h)
        :param fx:
        :param fy:
        :param cx:
        :param cy:
        :return: a point cloud of shape (h * w, 3), where x is depth (points into the image),
                                                          y is horizontal (points left),
                                                          z is vertical (points up)
        """
        # TODO are the "-1" necessary?
        x = torch.arange(0, depth.shape[1], device="cuda") * (camera_resolution[1] - 1) / (depth.shape[1] - 1)
        y = torch.arange(0, depth.shape[0], device="cuda") * (camera_resolution[0] - 1) / (depth.shape[0] - 1)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        xx = xx.flatten()
        yy = yy.flatten()
        zz = depth.flatten()
        x_world = (xx - cx) * zz / fx
        y_world = (yy - cy) * zz / fy
        z_world = zz
        point_cloud = torch.vstack((z_world, -x_world, -y_world)).T
        if self.filter_stairs:
            hole_mask = -y_world < self.floor_threshold # todo threshold parameter
            if hole_mask.any():
                scale_factor = self.floor_level / -y_world[hole_mask]
                point_cloud[hole_mask] *= scale_factor.unsqueeze(-1)
                return point_cloud, hole_mask

        return point_cloud, torch.empty((0,))

    def metric_to_px(self, x, y):
        epsilon = 1e-9  # Small value to account for floating-point imprecision

        return (
            int(x / self.cell_size + self.map_center_cells[0].item() + epsilon),
            int(y / self.cell_size + self.map_center_cells[1].item() + epsilon))

    def px_to_metric(self, px, py):
        return ((px - self.map_center_cells[0].item()) * self.cell_size,
                (py - self.map_center_cells[1].item()) * self.cell_size)

    # def hdbscan_cosine_grid_3d(
    #     self,
    #     feat_hwld: np.ndarray,
    #     min_cluster_size: int = 200,
    #     min_samples: int = None,
    #     pca_dim: int = None,
    #     spatial_weight: float = 0.0,
    #     coord_weights: tuple[float, float, float] = None,
    #     mask: np.ndarray = None,
    #     seed: int = 0,
    # ):
    #     """
    #     Cluster a 3D grid of CLIP features with HDBSCAN (cosine metric).

    #     Args:
    #         feat_hwld: (H, W, L, D) float array; L is vertical.
    #         min_cluster_size: HDBSCAN min cluster size (tune to your grid).
    #         min_samples: Optional HDBSCAN min_samples (None → heuristic).
    #         pca_dim: If set, PCA-reduce features to this dim (then re-normalize).
    #         spatial_weight: If >0, append standardized (x,y,z) coords * this weight.
    #         coord_weights: Per-axis weights for coords. Scalar or (wx, wy, wz).
    #         mask: Optional boolean (H, W, L); True = keep. Otherwise, keep finite rows.
    #         seed: Random state for PCA, not for HDBSCAN.

    #     Returns:
    #         labels: (H, W, L) int32; -1 denotes noise.
    #         model: the fitted HDBSCAN object (access probabilities_, outlier_scores_).
    #     """
    #     H, W, L, D = feat_hwld.shape

    #     # Flatten features
    #     X = feat_hwld.reshape(-1, D).astype(np.float32)

    #     # Valid points mask
    #     if mask is not None:
    #         valid = mask.reshape(-1)
    #     else:
    #         valid = np.all(np.isfinite(X), axis=1)

    #     Xv = X[valid]

    #     ### Umap
    #     umap_model = UMAP(n_components=5)
    #     reduced_embeddings = umap_model.fit_transform(Xv)

    #     clusterer = HDBSCAN(min_cluster_size=20)
    #     cluster_labels = clusterer.fit_predict(reduced_embeddings)

    #     labels = np.full(H*W*L, -1, dtype=np.int32)
    #     labels[valid] = cluster_labels
    #     labels = labels.reshape(H, W, L)

    #     return labels, clusterer

        # L2-normalize (cosine geometry)
        # Xv /= (np.linalg.norm(Xv, axis=1, keepdims=True) + 1e-8)

        # # Optional PCA for speed/memory; re-normalize afterward for cosine
        # if pca_dim is not None and pca_dim < Xv.shape[1]:
        #     pca = PCA(n_components=pca_dim, random_state=seed)
        #     Xv = pca.fit_transform(Xv).astype(np.float32)
        #     Xv /= (np.linalg.norm(Xv, axis=1, keepdims=True) + 1e-8)

        # Optional spatial regularization via (x,y,z)
        # if spatial_weight and spatial_weight > 0:
        #     # Coordinates aligned with flattening order (C-order)
        #     ii, jj, kk = np.indices((H, W, L))  # i=x(row), j=y(col), k=z(layer)
        #     coords = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(np.float32)
        #     coords = coords[valid]

        #     # Standardize then weight
        #     coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-8)
        #     if coord_weights is None:
        #         cw = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        #     else:
        #         cw = np.asarray(coord_weights, dtype=np.float32)
        #         if cw.size == 1:
        #             cw = np.repeat(cw, 3)
        #     Xv = np.hstack([Xv, spatial_weight * coords * cw]).astype(np.float32)

        # HDBSCAN with cosine distance (1 - cosine similarity)
        # clusterer = HDBSCAN(
        #     min_cluster_size=min_cluster_size,
        #     min_samples=min_samples,
        #     metric='cosine',
        # )
        # lab_v = clusterer.fit_predict(Xv)

        # # Scatter back to (H,W,L)
        # labels = np.full(H*W*L, -1, dtype=np.int32)
        # labels[valid] = lab_v
        # labels = labels.reshape(H, W, L)

        # # save_clusters_per_slice(labels, out_dir="cluster_images", top_k=50, seed=0, with_outlines=True, dpi=300)
        # save_clusters_grid_figure(labels, out_path="cluster_images/all_slices.png",
        #                   top_k=50, seed=0, with_outlines=True, dpi=200)

        # return labels, clusterer

if __name__ == "__main__":
    rr.init("rerun_example_points3d", spawn=False)
    rr.connect("127.0.0.1:1234")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
    rr.log(
        "world/xyz",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )
    from detectron2.data.detection_utils import read_image

    map = OneMap(1)
    depth = read_image('test_images/depth2.png', format="BGR") * (-1) + 255
    depth2 = read_image('test_images/depth.png', format="BGR")

    fac = 10
    x = torch.arange(0, depth.shape[1] / fac, dtype=torch.float32)
    y = torch.arange(0, depth.shape[0], dtype=torch.float32)
    xx, yy, = torch.meshgrid(x, y)

    values = torch.sin(xx / (50.0 / fac)).T.unsqueeze(0)

    # values[:, :depth.shape[1]//2] = 1.0
    start = time.time()
    # map.update(torch.zeros((depth.shape[0], depth.shape[1], 3)), depth, np.eye(4))
    map.update(values, depth[:, :, 0], np.eye(4))
    map.update(-values, depth2[:, :, 0], np.eye(4))
    print(time.time() - start)
