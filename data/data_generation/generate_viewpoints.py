import csv
import gzip
import json
import math
from pathlib import Path
from typing import List, Dict, Union
import os
import habitat_sim
import tqdm
import numpy as np
from scipy.spatial.transform import Rotation

from eval.dataset_utils import *
from habitat_sim import ActionSpec, ActuationSpec
from habitat_sim.utils import common as utils

object_metadata_file = Path("datasets/scene_datasets/fphab/semantics/objects.csv")
square = True
datasets_path = Path("datasets/langnav")
scene_path = "datasets/scene_datasets/fphab/"
max_radius_to_search_nav_points = 1.5

sensor_h, sensor_w = 0, 0
sensor_vert_pos = 0
hfov = 0
vfov_rad = 0
focus = 0
view_pt_valid_thresh = 0.1
boundary_pts_dist = 0.5
boundary_check_radius = 2
dist_btw_view_pts = 0.2
view_pts_max_radius = 1
delta_degrees = 20

def load_object_metadata():
    ## load object metadata
    object_metadata = {}
    with open(object_metadata_file, "r") as f:
        _csv_reader = csv.reader(f)
        headings = next(_csv_reader)
        for row in _csv_reader:
            _data = row[:]
            m_json = {}
            for _ind, _val in enumerate(_data):
                m_json[headings[_ind]] = _val
            object_metadata[_data[0]] = m_json

    return object_metadata


def load_scene(scene_id: str, sim=None):
    if sim is not None:
        sim.close()
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    backend_cfg.override_scene_light_defaults = True
    backend_cfg.pbr_image_based_lighting = True
    backend_cfg.scene_id = scene_id
    backend_cfg.enable_physics = True

    backend_cfg.scene_dataset_config_file = (
        scene_path + "hssd-hab-mon.scene_dataset_config.json"
    )
    backend_cfg.navmesh_settings = habitat_sim.nav.NavMeshSettings()
    backend_cfg.navmesh_settings.set_defaults()
    backend_cfg.navmesh_settings.agent_radius = 0.1
    backend_cfg.navmesh_settings.agent_height = 1.5
    backend_cfg.navmesh_settings.include_static_objects = True

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
        height=1.5,
        radius=0.1,
        action_space=dict(
            move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
            turn_left=ActionSpec("turn_left", ActuationSpec(amount=30.0)),
            turn_right=ActionSpec("turn_right", ActuationSpec(amount=30.0)),
        ),
    )
    agent_cfg.sensor_specifications = [rgb, depth]
    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)

    global sensor_h, sensor_w, sensor_vert_pos, vfov_rad, focus
    sensor_h, sensor_w = sim.config.agents[0].sensor_specifications[0].resolution
    sensor_vert_pos = sim.config.agents[0].sensor_specifications[0].position[1]
    vfov_rad = get_vfov(hfov)
    focus = sensor_w / (2 * np.tan(np.deg2rad(hfov)/2))

    return sim


def get_all_scene_objects(sim, object_metadata):

    # load all objects in the scene
    rgm = sim.get_rigid_object_manager()
    object_info = rgm.get_objects_info()
    scene_objects_headers = object_info[0].split(",")
    scene_objects = [x.split(",") for x in object_info[1:]]
    scene_objects_metadata = {
        obj[0]: {"scene_data": obj, "metadata": object_metadata[obj[0].split("_")[0]]}
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


def augment_dataset_with_viewpoints(object_nav_path, out_dataset_path):

    langnav_dataset_path = object_nav_path / "content"
    files = os.listdir(langnav_dataset_path)
    files = sorted(files, key=str.casefold, reverse=True)
    object_metadata = load_object_metadata()
    out_files = os.listdir(out_dataset_path)
    sim = None
    rgm = None

    count_total_goal_objects = 0
    count_not_generated = 0
    for file in tqdm.tqdm(files, desc="Scenes"):
        if file.endswith(".json.gz"):

            if file in out_files:
                continue

            with gzip.open(os.path.join(langnav_dataset_path, file), "r") as f:
                json_data = json.load(f)

            if len(json_data["episodes"]) == 0:
                continue

            episodes = json_data["episodes"]
            for episode in tqdm.tqdm(episodes, desc="Episodes"):
                scene_id = episode["scene_id"]
                episode_id = episode["episode_id"]
                if sim is None or not sim.curr_scene_name in scene_id:
                    sim = load_scene(scene_id, sim)
                    # load all objects in the scene
                    (
                        rgm,
                        scene_objects_headers,
                        scene_objects_metadata,
                        scene_objects_metadata_by_main_category,
                        scene_objects_metadata_by_wnsynsetkey,
                        scene_objects_metadata_by_name,
                    ) = get_all_scene_objects(sim=sim, object_metadata=object_metadata)

                agent_start_position = episode["start_position"]
                agent_start_height = agent_start_position[1]
                for goal in episode["goals"]:
                    # get possible goal objects in the episode
                    goal_objects = goal["goal_object"]

                    ## populate more navigable points for existing goals
                    for goal_obj in tqdm.tqdm(goal_objects, desc="goals"):
                        goal_id = goal_obj["object_id"]
                        semantic_id = int(scene_objects_metadata[goal_id]["scene_data"][5])
                        source_obj = rgm.get_object_by_id(semantic_id)
                        obj_pos = np.array(source_obj.translation)
                        obj_dims = np.array(
                            source_obj.root_scene_node.cumulative_bb.size()
                        )

                        try:
                            ## generate viewpoints
                            #Get boundary points (evenly-spaced)
                            success, spaced_boundary_pts = boundary_around_obj(sim=sim, obj_pos = obj_pos,
                                                                                        obj_dims = obj_dims,
                                                                                        pts_dist = boundary_pts_dist,
                                                                                        keep_final_pt = True,
                                                                                        _shoot_till = boundary_check_radius,
                                                                                        delta_degrees=delta_degrees)

                            if not success:
                                print(f'No boundary_around_obj for episode {scene_id}::{episode_id}, goal:{goal_id}.')
                                continue

                            #Generate Viewpoints around the Boundary 
                            ind = 0
                            view_pts = []
                            for bound_pt in tqdm.tqdm(spaced_boundary_pts, desc="Viewpoints"):

                                bound_view_pts = view_pts_around(sim=sim, ref_pt = bound_pt, 
                                                                        view_pts_dist = dist_btw_view_pts, 
                                                                        max_radius = view_pts_max_radius,
                                                                        obj_pos = obj_pos,
                                                                        obj_dims = obj_dims,
                                                                        prev_ref_pt = spaced_boundary_pts[ind - 1])
                                
                                view_pts.extend(bound_view_pts)
                                ind += 1

                            if len(view_pts) > 0:
                                nav_points = [[float(p[0][0]),float(p[0][1]),float(p[0][2])] for p in view_pts]
                                nav_points_rot = [[float(p[1][0]),float(p[1][1]),float(p[1][2]),float(p[1][3])] for p in view_pts]
                                goal_obj['navigable_points'] = nav_points
                                goal_obj['navigable_points_rotation'] = nav_points_rot
                                print(f'added {len(view_pts)} navigable points {scene_id}:{episode_id}.')
                            else:
                                print(f'No navigable points found {scene_id}:{episode_id}:')
                                print(f'---------Object {goal["object_category"]}; position: {obj_pos}; dimension: {obj_dims}')
                                count_not_generated += 1
                        except:
                            print(f'Exception in {scene_id}:{episode_id}:')
                            print(f'---------Object {goal["object_category"]}; position: {obj_pos}; dimension: {obj_dims}')
                            count_not_generated += 1
                        
                        count_total_goal_objects += 1

            with gzip.open(os.path.join(out_dataset_path, file), 'wt') as f:
                f.write(json.dumps(json_data))
            print(f"File saved: {file}")

    print(f"All files processed. No navigable points found for {count_not_generated} out of {count_total_goal_objects} goal objects.")

    return

def dist_btw_pts(pt_0, pt_1):
    if not isinstance(pt_0, np.ndarray): pt_0 = np.array(pt_0)
    if not isinstance(pt_1, np.ndarray): pt_1 = np.array(pt_1)
    return np.linalg.norm(pt_1 - pt_0)

def generate_pts_btw(pt_1, pt_2, num_pts):
    r"""
    Generate n-dim points between 
    """

    assert len(pt_1) == len(pt_2), "Both points should be of same size!"
    assert num_pts >= 2, "Number of points requested is less than 2!"
    num_dims = len(pt_1)

    next_coord = lambda param, dim: (1 - param) * pt_1[dim] + (param) * pt_2[dim] 
    param_vals = np.linspace(0, 1, num_pts)
    
    return [[next_coord(param, dim) for dim in range(num_dims)] 
            for param in param_vals]


def concat_arr(arr_main: np.ndarray, arr_add: np.ndarray):
    r""""
    Concatenates two arrays
    """
    if len(arr_add) == 0: return arr_main
    if len(arr_main) == 0: return np.array(arr_add)
    return np.concatenate((np.array(arr_main), np.array(arr_add)))


def quat_coeffs_to_yaw(quat_coeffs: List):
    r""""
    Converts Quatertion Coefficients to Yaw Angle.
    """

    quat = utils.quat_from_coeffs(quat_coeffs)

    rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
    rot_vec = rot.as_rotvec(degrees = True)
    return rot_vec[1]

def get_vfov(hfov):
    r"""
    Obtains Vertical Field of Vision (VFOV) using HFOV, Sensor Resolution
    """

    hfov_rad = np.deg2rad(hfov)
    vfov_rad = 2 * math.atan(math.tan(hfov_rad/2) * (sensor_h/sensor_w))

    # return np.rad2deg(vfov_rad)
    return vfov_rad

def rel_pos_to_px(rel_pos: List):
    r"""
    Projects 3D coordinate of object wrt agent position (rel_pos) onto the sensor frame.
    """

    delta_px_y = np.round( focus * rel_pos[1] / abs(rel_pos[2]) ).astype(int)
    delta_px_x = np.round( focus * rel_pos[0] / abs(rel_pos[2]) ).astype(int)

    px_y = (sensor_h//2) - delta_px_y
    px_x = (sensor_w//2) - delta_px_x

    return (px_y, px_x)

def boundary_around_nav_obj(sim, obj_pos: np.ndarray, obj_dims: np.ndarray,
                            pts_dist = 0.5, with_center = True):

    if obj_dims is None: return [obj_pos]
    
    bound_pts = []
    obj_corners = []
    floor_height = _get_floor_height(sim, obj_pos.copy())

    #Get object corners
    delta_vals = np.array( [[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]] )
    obj_corners = [perturb_along_dim(obj_pos, [0, 2], delta * obj_dims[[0, 2]]) 
                for delta in delta_vals]

    #Generate points around the object dimensions
    for ind in range(len(obj_corners)):

        corner_prev = obj_corners[ind-1]
        corner_next = obj_corners[ind]

        #Generate points in between the two corners
        dist = dist_btw_pts(corner_prev[[0, 2]], corner_next[[0, 2]])
        num_pts = math.ceil( dist / pts_dist )
        if num_pts > 2:
            pts_in_btw = generate_pts_btw(corner_prev, corner_next, num_pts)[:-1]
        else:
            pts_in_btw = [corner_prev]

        pts_in_btw = [pt for pt in pts_in_btw 
                    if sim.pathfinder.is_navigable([pt[0], floor_height, pt[2]])]
        if len(pts_in_btw) == 0: continue

        if len(bound_pts) == 0:
            bound_pts = np.array(pts_in_btw)
        else:
            bound_pts = np.concatenate((bound_pts, pts_in_btw))
    
    #Add center point (obj_pos) to boundary points
    if with_center:
        print(f"Adding center point")
        bound_pts = np.concatenate((obj_pos[np.newaxis, :], bound_pts))

    return bound_pts

## Generate Radial Navigable Samples around the Boundary
def sample_radial_nav_pts(sim,
                          ref_pt: np.ndarray,
                            max_radius: float = 1, 
                            pts_dist: float = 0.2):

    r"""
    Generate radial and navigable samples centered around a reference point. 
    These are generated at a fixed radius step and a fixed angular step.

    Args:
        - ref_pt: Reference point, used as the central reference to generate samples
        - max_radius: Maximum radius to generate points till
        - pts_dist: Distance between the generated samples.
    
    """

    radial_pts = []
    radius_info = []
    step_radius = pts_dist

    for radius in np.arange(step_radius, max_radius + step_radius, step_radius):
        
        #Get angular step value
        perimeter = 2 * np.pi * radius
        num_concentric_pts = perimeter / pts_dist
        step_degree = math.ceil( 360 / num_concentric_pts )

        for degree in np.arange(0, 360, step_degree):
            
            #Get the new sample point
            rad = np.deg2rad(degree)
            dx, dz = radius * np.cos(rad), radius * np.sin(rad)
            radial_pt = [ref_pt[0] + dx, ref_pt[1], ref_pt[2] + dz]

            #Check for navigability
            if sim.pathfinder.is_navigable(radial_pt):
                radial_pts.append(radial_pt)
                radius_info.append(radius)

    
    return np.array(radial_pts), np.array(radius_info)

def samples_from_boundary(
                            boundary_pts: np.ndarray, 
                            sample_max_radius: float = 1,
                            pts_dist: float = 0.2):
    
    r"""
    Generate Navigable samples around the boundary.

    Args:
        - boundary_pts: Points lying around the boundary
        - pts_dist: Distance between the generated sample points.
    
    """

    #Keeps track of active boundary points, useful for reducing overlap during sampling
    active_boundary_pts = [True] * len(boundary_pts)     
    ref_outer_pts = []
    sampled_pts = []

    while sum(active_boundary_pts) > 0:

        active_ind = active_boundary_pts.index(True)
        active_bound_pt = boundary_pts[active_ind]

        #Generate navigable radial samples around the current active boundary point
        radial_nav_samples, radius_info = sample_radial_nav_pts(active_bound_pt, 
                                                                max_radius = sample_max_radius,
                                                                pts_dist = pts_dist)
        
        if len(radial_nav_samples) == 0:
            active_boundary_pts[active_ind] = False
            continue

        #If previously sampled, use the previous outer valid points to filter current points based on minimum distance
        # A new sample should be at least pts_dist away from all the previous outer points
        if len(ref_outer_pts) > 0:
            is_valid_mask = [True] * len(radial_nav_samples)
            for ind, pt in enumerate(radial_nav_samples):
                is_valid = np.all([dist_btw_pts(pt, ref_pt) > pts_dist for ref_pt in ref_outer_pts])
                is_valid_mask[ind] = is_valid

            radial_nav_samples = radial_nav_samples[is_valid_mask]
            radius_info = radius_info[is_valid_mask]
        
        #Append the valid sampled points
        if len(sampled_pts) == 0:
            sampled_pts = radial_nav_samples
        else:
            sampled_pts = np.concatenate((sampled_pts, radial_nav_samples), axis = 0)

        #Update the outer points for reference
        ref_outer_pts = radial_nav_samples[radius_info == max(radius_info)]

        #Deactivate all boundary points within the range of the outer points
        active_boundary_pts[active_ind] = False
        if sum(active_boundary_pts) <= 0: break
        
        #Deactivate boundary Points before the current boundary point
        within_range = True
        left_ind = active_ind - 1
        while within_range and left_ind >= 0:
            
            within_range = np.any([dist_btw_pts(ref_pt, boundary_pts[left_ind]) <= pts_dist for ref_pt in ref_outer_pts])

            if within_range: active_boundary_pts[left_ind] = False
            left_ind -= 1
        if sum(active_boundary_pts) <= 0: break

        #Deactive boundary Points after the current boundary point
        within_range = True
        right_ind = active_ind + 1
        while within_range and (right_ind < len(boundary_pts)):
    
            within_range = np.any([dist_btw_pts(ref_pt, boundary_pts[right_ind]) <= pts_dist for ref_pt in ref_outer_pts])

            if within_range: active_boundary_pts[right_ind] = False
            right_ind += 1

    
    return sampled_pts
def _get_floor_height(sim, search_center: np.ndarray) -> float:
    """Reference: OVON pose_sampler.py"""

    point = np.asarray(search_center)[:, None]
    snapped = sim.pathfinder.snap_point(point)

    # the centroid should not be lower than the floor
    tries = 0
    while point[1, 0] < snapped[1]:
        point[1, 0] -= 0.05
        snapped = sim.pathfinder.snap_point(point)
        tries += 1
        if tries > 40:  # trace 2.0m down.
            break

    return snapped[1]

def boundary_around_obj(sim,
                        obj_pos: np.ndarray, 
                        obj_dims: np.ndarray = None,
                        pts_dist: float = 0.5, 
                        delta_degrees: float = 20, 
                        keep_final_pt: bool = True, 
                        _shoot_till: float = 2,
                        _max_tries: int = 100):

    r"""
    Args:
        - sim: Simulator object.
        - obj_pos: Object Position
        - obj_dims: Object Dimensions
        - pts_dist: Distance between two boundary points
        - delta_degrees: Step size for varying degree values.
        - keep_final_pt: If False, removes the last boundary point if its 
                        distance to the initial point is less than pts_dist
        - _shoot_till: Maximum distance to check for possible boundary point
    """

    pt = obj_pos.copy()
    pt[1] = _get_floor_height(sim, pt)
    if sim.pathfinder.is_navigable(pt):
        print(f"Object position is in a navigable region!")
        return True, boundary_around_nav_obj(sim, np.array(obj_pos), 
                                                    np.array(obj_dims), 
                                                    pts_dist, 
                                                    with_center = True)

    boundary_pts = []
    invalid_ranges = []
    start_degree = 0

    #Start Boundary Point, which will be used as reference for subsequent boundary points    
    start_pt, start_degree = next_valid_boundary_pt(sim, obj_pos, _shoot_till, 
                                                start_degree = start_degree,
                                                delta_degrees=10)
    if start_pt is None: return False, []
    boundary_pts.append(start_pt)
    degrees = start_degree

    #Iterates through 360 degrees, making sure all possible boundary points are attained
    count_tries = 0
    while degrees < (360 + start_degree) and count_tries < _max_tries:

        #Next boundary point, with reference to the last sampled boundary point 
        next_pt, degrees, invalid_ranges = next_boundary_pt_at_dist(sim, obj_pos, 
                                                            ref_pt = boundary_pts[-1],
                                                            pts_dist = pts_dist,
                                                            curr_degree = degrees,
                                                            start_degree = start_degree,
                                                            delta_degrees = delta_degrees,
                                                            _shoot_till = _shoot_till,
                                                            invalid_ranges = invalid_ranges)

        if next_pt is not None: boundary_pts.append(next_pt)
        count_tries += 1

    if (not keep_final_pt) and (dist_btw_pts(boundary_pts[0][[0, 2]], boundary_pts[-1][[0, 2]]) < pts_dist): boundary_pts = boundary_pts[:-1]
    return True, boundary_pts

def next_valid_boundary_pt(
                            sim,
                            obj_pos: np.ndarray, 
                            _shoot_till: float = 2, 
                            start_degree: float = 0, 
                            delta_degrees: float = 5):
    r"""
    Varies the degree starting from start_degree, and returns when we reach the first boundary point

    Args:
        - sim: Simulator Object
        - obj_pos: Object Position. Used to find the boundary with reference to this point.
        - _shoot_till: Distance to check for the boundary. 
        - start_degree: Starting degree value to search for boundary point from.
        - delta_degrees: Search step to vary degrees by.
    """
    degrees = start_degree - delta_degrees
    while degrees <= (360 + start_degree):
        degrees += delta_degrees
        found_pt, pt = shoot_point_to_boundary(sim=sim, obj_pos=obj_pos, degrees=degrees, _shoot_till=_shoot_till)
        if found_pt: return pt, degrees

    return None, degrees

def next_boundary_pt_at_dist(
                                sim,
                                obj_pos: np.ndarray, ref_pt: np.ndarray, pts_dist: float,
                                curr_degree: float, delta_degrees:float, start_degree = 0, 
                                _shoot_till: float = 2, 
                                invalid_ranges: float = []):
    r"""
    Binary Search for the next boundary pt at a specific distance to ref_pt.

    Args:
        - sim: Simulator object
        - obj_pos: Object Position.
        - ref_pt: Reference point from which the next boundary point is to be found.
        - pts_dist: Distance between the reference point and the next boundary point.
        - curr_degree: Current angle value in degrees, corresponding to the reference point.
        - delta_degrees: Step size for varying degrees.
        - start_degree: Upper limit offset for degrees.
        - _shoot_till: Distance to check for the boundary. (Look into shoot_point_to_boundary)
        - invalid_ranges: Invalid ranges of degree values, denoting non-boundary regions.
        
    """

    search_depth = 1
    skip_iter = False
    curr_degree += (delta_degrees/2)

    diff_resolution = 0.05
    max_search_depth = math.ceil(np.log2(1/diff_resolution))

    #Searches a full 360 to find the next point at specific distance
    while curr_degree < (360 + start_degree):
        
        #Find potential boundary point at current degree value
        found_pt, potential_pt = shoot_point_to_boundary(sim=sim, obj_pos=obj_pos, degrees=curr_degree, _shoot_till=_shoot_till)
        if not found_pt: 
            
            #If current degree is in an invalid range (region without a boundary),
            # then update current degree to end of invalid range
            for invalid_range in invalid_ranges:
                lower_lim, upper_lim = invalid_range[0], invalid_range[1]

                if lower_lim > upper_lim:
                    if (upper_lim < curr_degree < 360) or (0 < curr_degree < lower_lim):
                        curr_degree = lower_lim
                        search_depth += 1
                        skip_iter = True

                else:
                    if lower_lim < curr_degree < upper_lim:
                        curr_degree = invalid_range[1]
                        search_depth += 1
                        skip_iter = True

            if skip_iter: 
                skip_iter = False
                continue
            
            #If no invalid range is found, then this region is a first occurrence.
            # Scan through degrees and get the first boundary point. Update the invalid ranges.
            invalid_degree_start = curr_degree
            potential_pt, curr_degree = next_valid_boundary_pt(sim, obj_pos, _shoot_till, 
                                                    start_degree = curr_degree,
                                                    delta_degrees = 1)
            invalid_ranges.append((invalid_degree_start, curr_degree))

            # return potential_pt, curr_degree+(diff_resolution*10), invalid_ranges
            return potential_pt, curr_degree, invalid_ranges

        #Get the distance between the potential point and the previous boundary point (ref_pt)
        dist_from_prev = dist_btw_pts(potential_pt[[0, 2]], ref_pt[[0, 2]])

        #Binary Search Implementation
        # If the search goes on for more than 20 steps, then we force a solution to solve the indecision
        if (abs(dist_from_prev - pts_dist) < diff_resolution) or (search_depth > max_search_depth): 
            return potential_pt, curr_degree, invalid_ranges
        
        elif dist_from_prev < pts_dist: 
            if search_depth > 1: search_depth += 1
            curr_degree += delta_degrees/(2**search_depth)

        elif dist_from_prev > pts_dist:
            search_depth += 1
            curr_degree -= delta_degrees/(2**search_depth)
    
    return None, curr_degree, invalid_ranges

def shoot_point_to_boundary(
                            sim,
                            obj_pos: np.ndarray, 
                            degrees: float, 
                            _shoot_till: float = 2):
    r""""
    Args:
        - sim: Simulator
        - obj_pos: Object Position (should be a non-navigable point)
        - degrees: Angle (in degrees) to face at, measured with respect to the object position
        - _shoot_till: Generate points till this limit (in meters)
        - _max_tries: Maximum number of trials to find the boundary point
    Returns:
        - found_boundary_pt (bool)
        - Boundary Point
    """

    #Ground the point to the floor height
    pt = obj_pos.copy()
    pt[1] = _get_floor_height(sim, pt)
    assert not sim.pathfinder.is_navigable(pt), "Please provide a non-navigable position as obj_pos"

    #Generate points along the angle direction till the given limit (shoot_till)
    #Check navigability of points. If none are navigable, return 
    shoot_pts = shoot_pts_towards(start_pos = pt.copy(),
                                degrees = degrees,
                                shoot_till = _shoot_till)
    is_nav_pts = [sim.pathfinder.is_navigable(point) for point in shoot_pts]
    if np.all(np.invert(is_nav_pts)): return False, None
    
    #Get the transition points: One point is non-navigable while the other is navigable
    ind_transition = np.argwhere(is_nav_pts)[0].item()
    low_bound_pt = pt if ind_transition==0 else shoot_pts[ind_transition - 1]

    #Snap point
    return True, sim.pathfinder.snap_point(low_bound_pt)

def shoot_pts_towards(
                        start_pos: np.ndarray, 
                        degrees: float, 
                        shoot_till: float, 
                        resolution_fact: int = 1, 
                        _num_samples: int = 10):
    r""""
    Args:
        - start_pos: Starting position from which to shoot points
        - degrees: Orientation to face towards when shooting points
        - resolution_fact: Resolution Factor, results in smaller step size
        - num_samples: Number of evenly-spaced points to shoot from the start position
    """

    assert resolution_fact > 0

    #Step size from the start position to the final limit (shoot_till)
    _delta = shoot_till / (_num_samples * resolution_fact)

    #Direction to step towards
    rad = np.deg2rad(degrees)
    dx_unit, dz_unit = np.cos(rad), np.sin(rad)

    #Shoot evenly-spaced points towards the required direction
    shoot_pts = [perturb_along_dim(start_pos.copy(), 
                                [0, 2], 
                                i * _delta * np.array([dx_unit, dz_unit]) )
                for i in range(_num_samples)]
    
    return shoot_pts

def perturb_along_dim(pt: Union[np.ndarray, List], dim: Union[int, List], delta: Union[float, List]):
    r"""
    Args:
        - pt: Point to perturb. 
        - dim: Dimension to perturb along. Can be an integer of a list of integers.
        - delta: Amount to perturb along each dimension. 
    
    """

    point = pt.copy()

    #Perturb only along one dimension
    if isinstance(dim, int):
        point[dim] += delta
        return point
    
    #Perturb along one or more dimensions
    assert len(delta) == len(dim), "Length of Delta values should match length of dimensions to perturb."
    for curr_dim, curr_delta in zip(dim, delta):
        point[curr_dim] += curr_delta
    
    return point

def group_angles(angles: List, step_size: float):
    r""""
    For a given list of angles, creates groups of angles differing only by step_size.
    
    """

    groups = []
    curr_group = [0, 0]

    #Checks if the current angle is <step_size> away from the previous angle
    #If so, the current group is updated. This is done till a greater jump is observed.
    for ind in range(len(angles[1:])):

        diff = (angles[ind+1] - angles[ind]) % 360      #Difference of angles

        if diff <= step_size:
            curr_group[1] = ind + 1                     #Update Current group
        else:
            if curr_group[0] != curr_group[1]: 
                groups.append(curr_group)               #Add current group 
            
            curr_group = [ind+1, ind+1]                 #Reinitialize Current group

    if curr_group[0] != curr_group[1]: 
        groups.append(curr_group)

    if len(groups) == 0: return []

    #Check if the difference between first and last angles is valid
    #If so, a new group extending forward and backward is created
    diff = (angles[0] - angles[-1]) % 360

    if diff <= step_size:

        curr_group = [len(angles) - 1, 0]

        if groups[-1][1] == len(angles)-1:      
            curr_group[0] = groups[-1][0] - len(angles)
            del groups[-1]
        
        if (len(groups) > 1) and (groups[0][0] == 0):
            curr_group[1] = groups[0][1]
            del groups[0]
        
        groups.append(curr_group)
    

    #Return groups with angles
    grouped_angles = [[angles[group[0]], angles[group[1]]] for group in groups]
    return np.array(grouped_angles)

def view_pts_around(sim, ref_pt: np.ndarray, view_pts_dist: float, max_radius: float,
                        obj_pos: np.ndarray, obj_dims: np.ndarray = None,
                        prev_ref_pt = None,
                        ):

    r""""
    Generates View Points around a given (boundary) reference point (ref_pt).

    Args:
        - ref_pt: Boundary Point.
        - view_pts_dist: Distance between viewpoints.
        - max_radius: Maximum radius to generate viewpoints.
        - obj_pos: Position of object
        - prev_ref_pt: Previous Boundary Point used as reference to avoid redundant viewpoint generation.
    """
    
    view_pts = []
    step_radius = view_pts_dist
    valid_view_zones = []
    valid_degrees = []
    
    #Iterates from the Maximum possible radius down to the minimum distance
    for radius in np.arange(max_radius, 0, -1 * step_radius):
        
        #Get angular step value
        perimeter = 2 * np.pi * radius
        num_concentric_pts = perimeter / view_pts_dist
        step_degree = math.ceil( 360 / num_concentric_pts)

        for degree in np.arange(0, 360, step_degree):

            #Potential Point
            rad = np.deg2rad(degree)
            dx, dz = radius * np.cos(rad), radius * np.sin(rad)
            radial_pt = [ref_pt[0] + dx, ref_pt[1], ref_pt[2] + dz]        

            #Skip if point is close to the previous boundary point
            if prev_ref_pt is not None:
                dist_to_prev_pt = dist_btw_pts(np.array(radial_pt)[[0, 2]], prev_ref_pt[[0, 2]])
                if dist_to_prev_pt <= (max_radius * 1.0): continue

            if radius == max_radius:

                #Check if this is a viewpoint
                is_valid, view_pt_rot = is_a_viewpoint(sim, np.array(radial_pt), 
                                                            np.array(obj_pos),
                                                            obj_dims)

                if is_valid:
                    valid_degrees.append(degree)


            #Skip if the point is not navigable
            if not sim.pathfinder.is_navigable(radial_pt): continue

            if radius < max_radius:

                #If the point is inside valid zone, append and continue
                skip_iter = False
                for low_lim, high_lim in valid_view_zones:

                    if high_lim < low_lim:

                        if (low_lim <= degree <= 360) or (0 <= degree <= high_lim):
                            view_pts.append((radial_pt, view_pt_rot))
                            skip_iter = True
                            break
                    
                    else:

                        if low_lim <= degree <= high_lim: 
                            view_pts.append((radial_pt, view_pt_rot))
                            skip_iter = True
                            break

                if skip_iter: continue

                #Check if point is a viewpoint
                is_valid, view_pt_rot = is_a_viewpoint(sim, np.array(radial_pt), 
                                                            np.array(obj_pos),
                                                            obj_dims)



            if not is_valid: continue
            
            view_pts.append((radial_pt, view_pt_rot))    

        if (radius == max_radius) and (len(valid_degrees) > 1):

            valid_view_zones = group_angles(valid_degrees, step_degree)

        
    if len(view_pts) > 1:

        range_of_angles = [face_object(view_pt[0], np.array(obj_pos), return_yaw=True) for view_pt in view_pts]
        max_angle, min_angle = max(range_of_angles), min(range_of_angles)
        mean_angle = np.mean(range_of_angles)

        if min_angle <= mean_angle <= max_angle:
            clockwise = True
        else:
            clockwise = False

        #Check if this is a viewpoint
        _, ref_pt_rot = is_a_viewpoint(sim, np.array(ref_pt), 
                                            np.array(obj_pos),
                                            obj_dims)

        angle_obj_to_bound = face_object(ref_pt, obj_pos, return_yaw=True)

        if clockwise:
            if min_angle <= angle_obj_to_bound <= max_angle:
                view_pts.append((ref_pt, ref_pt_rot))
        else:
            if (-180 < angle_obj_to_bound <= min_angle) or (max_angle <= angle_obj_to_bound <= 180):
                view_pts.append((ref_pt, ref_pt_rot))

    return view_pts #np.array(view_pts, dtype=object)

def is_a_viewpoint(sim, pt: np.ndarray, 
                    obj_pos: np.ndarray,
                    obj_dims: np.ndarray = None,
                    display: bool = None):
    
    r""""
    Check if the give point is a viewpoint of the object at obj_pos.

    Args:
        - pt: Possible viewpoint position.
        - obj_pos: Object position.
    
    Returns:
        - is_viewpoint (bool)
        - Viewpoint rotation
    """

    #Rotates the agent to face the object
    view_pt_rot = face_object(obj_pos, pt)

    #Agent height is enforced
    pt[1] = 0

    #Gets depth observation
    obs = obs_at_pose(sim, pos = pt, rot = view_pt_rot)
    depth = obs["depth"]
    
    #Pixel locations corresponding to the positions around the object dimensions
    obj_dim_pixels, obj_dim_depth = get_obj_pixels(np.array(pt), np.array(obj_pos), obj_dims)
    # if len(obj_dim_pixels) == 0: return False, view_pt_rot

    #If any of these pixels are visible, return True
    for px, target_depth in zip(obj_dim_pixels, obj_dim_depth):

        observed_depth = depth[px[0], px[1]]

        # if display: 
        #     display_sample(obs["color_sensor"], 
        #                 obs["semantic_sensor"], 
        #                 obs["depth"], 
        #                 plot_pts = [px[::-1]])

        if (observed_depth < (target_depth - 0.1)): continue
        else: return True, view_pt_rot
    return False, view_pt_rot 

def get_obj_pixels(agent_pos: np.ndarray, obj_pos: np.ndarray, obj_dims: np.ndarray = None):
    r""""
    Obtain the pixels corresponding to points along the object dimensions.
    """

    #Set agent height
    agent_pos[1] = sensor_vert_pos

    #Get Rotation Matrix
    quat_coeffs = face_object(obj_pos, agent_pos)
    quat = utils.quat_from_coeffs(quat_coeffs)
    rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
    rot_mat = rot.as_matrix()

    if obj_dims is not None:
        obj_px = []
        obj_pt_depth = []

        #Gets positions of points around the object dimensions
        obj_dim_pts = obj_bound_box_pts(obj_pos, obj_dims, agent_pos)  

        for pos in obj_dim_pts:

                rel_pos = pos - agent_pos           #Position of object wrt agent
                rel_pos = rel_pos @ rot_mat         #Rotates to face the object

                px = rel_pos_to_px(rel_pos)    #Pixel location from the position

                if (0 <= px[0] < sensor_h) and (0 <= px[1] < sensor_w):
                    obj_px.append(px)     
                    obj_pt_depth.append( abs(rel_pos[2]) ) 

        return np.array(obj_px), np.array(obj_pt_depth)


    #If obj_dims is None, only calculate for object position
    rel_pos = obj_pos - agent_pos
    rel_pos = rel_pos @ rot_mat

    obj_px = rel_pos_to_px(rel_pos)
    obj_pt_depth = abs(rel_pos[2])

    if (0 <= obj_px[0] < sensor_h) and (0 <= obj_px[1] < sensor_w):
        return np.array(obj_px)[np.newaxis, :], np.array([obj_pt_depth])
    
    return [], []

def obj_bound_box_pts(obj_pos: np.ndarray, obj_dims: np.ndarray, agent_pos: np.ndarray):
    r"""
    Generates points covering the dimensions of the object, with respect to the central object position
    """
    
    if obj_dims[2] < 0.1: 
        #Centered on the object, not at the bottom.
        y_perturbs = [-0.5, 0.5]
    else: 
        #Not Centered. Situated at the bottom of the object.
        y_perturbs = [0.5, 1]

    #Perturb along the xz plane: Five point (2 along each dimension, and one center point)
    xz_perturb_pts = np.array([perturb_along_dim(obj_pos, dim, delta * obj_dims[dim]) for dim in [0, 2] for delta in [-0.5, 0.5]])
    xz_perturb_pts = np.concatenate((np.array(obj_pos)[np.newaxis, :], xz_perturb_pts))
    
    #Perturb along the y axis and only keep the valid points within frame. 
    y_perturb_pts = np.array([perturb_along_dim(obj_pos, 1, i * obj_dims[1]) for i in y_perturbs])
    y_perturb_pts = np.concatenate((np.array(obj_pos)[np.newaxis, :], y_perturb_pts))
    y_perturb_valid = [is_obj_in_frame(pt, agent_pos) for pt in y_perturb_pts]
    y_valid_pts = [y_perturb_pts[ind][1] for ind in range(len(y_perturb_pts)) if y_perturb_valid[ind]]

    #All points along the dimensions of the object
    obj_pts = [[xz_pt[0], y_pt, xz_pt[2]] for y_pt in y_valid_pts for xz_pt in xz_perturb_pts]

    return obj_pts

def face_object(object_position: np.ndarray, point: np.ndarray, return_yaw: bool = False):
    r"""
    Faces the agent (at point) towards the object position.
    """
    EPS_ARRAY = np.array([1e-8, 0.0, 1e-8])
    cam_normal = (object_position - point) + EPS_ARRAY
    cam_normal[1] = 0
    cam_normal = cam_normal / np.linalg.norm(cam_normal)
    q = utils.quat_from_two_vectors(habitat_sim.geo.FRONT, cam_normal)
    quat_coeffs = utils.quat_to_coeffs(q)

    if return_yaw: return quat_coeffs_to_yaw(quat_coeffs)
    return utils.quat_to_coeffs(q)

def obs_at_pose(sim, pos: np.ndarray, rot: float = None,):
    r"""
    Get the RGB and Depth observations at position.
    """

    agent_state = habitat_sim.AgentState()
    agent_state.position = pos
    if rot is not None: agent_state.rotation = rot  

    sim.agents[0].set_state(agent_state, infer_sensor_states=False)
    return sim.get_sensor_observations()

def is_obj_in_frame(obj_pos: np.ndarray, agent_pos: np.ndarray):
    r"""
    Checks if object height is viewable from agent's perspective.
    """

    #Half of VFOV
    half_vfov_rad = vfov_rad / 2

    #Depth value: Distance from agent to object
    depth_val = dist_btw_pts(obj_pos[[0, 2]], agent_pos[[0, 2]])

    #Get upper and lower maximum view heights of agent at min_viewable_depth
    half_height = depth_val * np.tan(half_vfov_rad)
    upper_view_height = sensor_vert_pos + half_height
    lower_view_height = sensor_vert_pos - half_height

    if (lower_view_height <= obj_pos[1] <= upper_view_height): return True
    return False

if __name__ == "__main__":
    
    splits = ["test"] #["val","test"]
    
    for split in splits:
        
        object_nav_path = datasets_path / split
        out_dataset_path = datasets_path / "eval" / split / "content"
        os.makedirs(out_dataset_path, exist_ok=True)

        print(f'Starting {split}...')
        augment_dataset_with_viewpoints(object_nav_path, out_dataset_path)
        print(f'Finished {split}...')

