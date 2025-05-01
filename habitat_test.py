"""
This script is used to test the habitat-sim library together with OneMap
"""
import time

# habitat
import habitat_sim
from habitat_sim.utils import common as utils

# numpy
import numpy as np

# rerun
import rerun as rr
import rerun.blueprint as rrb
from habitat_sim import ActionSpec, ActuationSpec
from numpy.lib.function_base import angle

# keyboard input
# from pynput import keyboard

# scipy
from scipy.spatial.transform import Rotation as R

# MON
from mapping import Navigator
from vision_models.clip_dense import ClipModel
from vision_models.yolo_world_detector import YOLOWorldDetector

# from onemap_utils import log_map_rerun
from planning import Planning, Controllers
from config import *
from mapping import rerun_logger

# Global variables
running = True

if __name__ == "__main__":
    config = load_config().Conf
    if type(config.controller) == HabitatControllerConf:
        pass
    else:
        raise NotImplementedError("Spot controller not suited for habitat sim")

    model = ClipModel("weights/clip.pth")
    detector = YOLOWorldDetector(0.3)
    mapper = Navigator(model, detector, config)
    logger = rerun_logger.RerunLogger(mapper, False, "", debug=False) if config.log_rerun else None
    mapper.set_query(["A Couch"])
    hm3d_path = "datasets/scene_datasets/hm3d"

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = hm3d_path + "/val/00853-5cdEh9F2hJL/5cdEh9F2hJL.basis.glb"
    backend_cfg.scene_dataset_config_file = hm3d_path + "/hm3d_annotated_basis.scene_dataset_config.json"

    hfov = 90
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.hfov = hfov
    rgb.position = np.array([0, 0.88, 0])
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    res_x = 640
    res_y = 640
    rgb.resolution = [res_y, res_x]

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.hfov = hfov
    depth.position = np.array([0, 0.88, 0])
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [res_y, res_x]

    hfov = np.deg2rad(hfov)
    focal_length = (res_x / 2) / np.tan(hfov / 2)
    principal_point_x = res_x / 2
    principal_point_y = res_y / 2
    K = np.array([
        [focal_length, 0, principal_point_x],
        [0, focal_length, principal_point_y],
        [0, 0, 1]
    ])

    agent_cfg = habitat_sim.agent.AgentConfiguration(action_space=dict(
        move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        turn_left=ActionSpec("turn_left", ActuationSpec(amount=5.0)),
        turn_right=ActionSpec("turn_right", ActuationSpec(amount=5.0)),
    ))
    agent_cfg.sensor_specifications = [rgb, depth]

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)
    objects = sim.semantic_scene.objects
    categories = [ob.category.name() for ob in objects]
    scene_categories = sim.semantic_scene.categories
    scene_categories = [cat.name() for cat in scene_categories]
    for cat in categories:
        if cat not in scene_categories:
            print("Object category not in scene categories:", cat)

    for cat in scene_categories:
        if cat not in categories:
            print("Scene category not in object categories:", cat)
    print(len(categories), len(scene_categories))
    print("Unique categories:", len(set(categories)))
    print(set(categories))

    # Global flag to control the simulation loop
    running = True

    action_mapping = {
        "w": "move_forward",
        "a": "turn_left",
        "d": "turn_right",
        "q": "enter_query",
        "o": "autonomous",
    }

    # Main simulation loop
    initial_sequence = ["turn_left"] * 28 * 2  # + ["move_forward"]*10
    # initial_sequence = ["turn_left"]*5 + ["move_forward"]*5
    qs = ["A fridge", "A TV", "A toilet", "A Couch", "A bed"]
    running = True
    autonomous = True
    controller = Controllers.HabitatController(sim, config.controller)
    while running:
        action = None
        if len(initial_sequence):
            action = initial_sequence[0]
            initial_sequence.pop(0)
            if not len(initial_sequence):
                mapper.one_map.reset_checked_map()
        elif autonomous:
            action = None
            # print("Goal pos: ", goal_pos)

            state = sim.get_agent(0).get_state()
            orientation = state.rotation
            q0 = orientation.x
            q1 = orientation.y
            q2 = orientation.z
            q3 = orientation.w

            r = R.from_quat([q0, q1, q2, q3])
            # r to euler
            pitch, yaw, roll = r.as_euler("yxz")
            # pitch is actually around z
            # orientation is pitch!
            yaw = pitch
            current_pos = np.array([[-state.position[2]], [-state.position[0]], [state.position[1]]])
            path = mapper.get_path()
            # rr.log("map/path", rr.LineStrips2D(path))
            if path and len(path) > 1:
                path = Planning.simplify_path(np.array(path))
                path = path.astype(np.float32)
                for i in range(path.shape[0]):
                    path[i, :] = mapper.one_map.px_to_metric(path[i, 0], path[i, 1])
                controller.control(current_pos, yaw, path)
                observations = sim.get_sensor_observations()
        if action and action != "enter_query":
            observations = sim.step(action)
        elif not autonomous:
            continue

        state = sim.get_agent(0).get_state()
        pos = np.array(([[-state.position[2]], [-state.position[0]], [state.position[1]]]))
        # print(pos)
        mapper.set_camera_matrix(K)
        orientation = state.rotation
        q0 = orientation.x
        q1 = orientation.y
        q2 = orientation.z
        q3 = orientation.w

        r = R.from_quat([q0, q1, q2, q3])
        # r to euler
        pitch, yaw, roll = r.as_euler("yxz")
        # pitch is actually around z
        r = R.from_euler("xyz", [0, 0, pitch])
        r = r.as_matrix()
        transformation_matrix = np.hstack((r, pos))
        transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))
        t = time.time()
        obj_found = mapper.add_data(
            observations["rgb"][:, :, :-1].transpose(2, 0, 1), observations["depth"].astype(np.float32),
                        transformation_matrix)
        print("Time taken to add data: ", time.time() - t)

        cam_x = pos[0, 0]
        cam_y = pos[1, 0]
        if logger:
            rr.log("camera/rgb", rr.Image(observations["rgb"]))
            rr.log("camera/depth", rr.Image((observations["depth"] - observations["depth"].min()) / (
                    observations["depth"].max() - observations["depth"].min())))
            logger.log_map()
            logger.log_pos(cam_x, cam_y)
        if obj_found:
            mapper.set_query([qs[0]])
            qs.pop(0)
            continue
