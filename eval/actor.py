# abstract classes
from abc import ABC, abstractmethod

from typing import Dict, Tuple

import numpy as np

# MON
from vision_models.clip_dense import ClipModel
from vision_models.blip2_model import BLIP2Model
from vision_models.blip2_model_patched import BLIP2PatchedModel
from vision_models.clip_image import ClipImageModel
from vision_models.lseg_model import LSegModel
from vision_models.yolo_world_detector import YOLOWorldDetector
from vision_models.grounding_dino_detector import GroundingDinoDetector
# from vision_models.yolov8_model import YoloV8Detector
from vision_models.point_nav_policy import WrappedPointNavResNetPolicy
# from vision_models.yolov6_model import YOLOV6Detector
from vision_models.yolov7_model import YOLOv7Detector

from mapping import Navigator
from planning import Planning, Controllers
# scipy
from scipy.spatial.transform import Rotation as R

import rerun as rr

class Actor(ABC):
    @abstractmethod
    def act(self,
            observations: Dict[str, any]) -> Tuple[Dict, bool]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_query(self, query: str):
        pass

    @abstractmethod
    def set_exploit(self):
        pass


class MONActor(Actor):
    def __init__(self, config):
        if len(config.use_model) == 0:
            model = ClipModel("weights/clip.pth", jetson=False, fuse_similarity=config.planner.fuse_similarity)
        elif "blip2_patch" in config.use_model:
            model = BLIP2PatchedModel()
        elif "blip2" in config.use_model:
            model = BLIP2Model()
        elif "clipimage" in config.use_model:
            model = ClipImageModel(jetson=False, fuse_similarity=config.planner.fuse_similarity)
        elif "lseg" in config.use_model:
            model = LSegModel(jetson=False, fuse_similarity=config.planner.fuse_similarity)
        
        if len(config.use_detector_model) == 0:
            if config.planner.using_ov:
                detector = YOLOWorldDetector(config.planner.yolo_confidence)
            else:
                detector = YOLOv7Detector(config.planner.yolo_confidence)
            # else YoloV8Detector(config.planner.yolo_confidence)
        elif "gdino" in config.use_detector_model:
            detector = GroundingDinoDetector(confidence_threshold=config.planner.yolo_confidence)

        self.policy = None
        self.action_lookup = [None, 'move_forward', 'turn_left', 'turn_right']
        self.square = config.square_im
        if config.use_pointnav:
            self.policy = WrappedPointNavResNetPolicy(config.pointnav_ckpt_path, "/home/finn/External/vlfm/pointnav_conf.pth", "cuda")

        self.mapper = Navigator(model, detector, config)

        self.init = 12*2
        hfov = 90 if self.square else 97
        res_x = 640
        res_y = 640 if self.square else 480
        hfov = np.deg2rad(hfov)
        focal_length = (res_x / 2) / np.tan(hfov / 2)
        principal_point_x = res_x / 2
        principal_point_y = res_y / 2
        K = np.array([
            [focal_length, 0, principal_point_x],
            [0, focal_length, principal_point_y],
            [0, 0, 1]
        ])
        self.mapper.set_camera_matrix(K)
        self.controller = Controllers.HabitatController(None, config.controller)

    def act(self, observations: Dict[str, any]) -> Tuple[Dict, bool]:
        return_act = {}
        state = observations["state"]
        pos = np.array(([[-state.position[2]], [-state.position[0]], [state.position[1]]]))

        orientation = state.rotation
        q0 = orientation.x
        q1 = orientation.y
        q2 = orientation.z
        q3 = orientation.w

        r = R.from_quat([q0, q1, q2, q3])
        # r to euler
        yaw, _, _1 = r.as_euler("yxz")
        # pitch is actually around z
        r = R.from_euler("xyz", [0, 0, yaw])
        r = r.as_matrix()
        transformation_matrix = np.hstack((r, pos))
        transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))
        obj_found = self.mapper.add_data(observations["rgb"][:, :, :-1].transpose(2, 0, 1),
                             observations["depth"].astype(np.float32),
                             transformation_matrix)
        if self.init > 0:
            return_act['discrete'] = 'turn_left'
            self.init -= 1
            return return_act, False
        else:
            path = self.mapper.get_path()
            if isinstance(path, str):
                if path == "L":
                    return_act['discrete'] = 'turn_left'
                    return return_act, False
                elif path == "R":
                    return_act['discrete'] = 'turn_right'
                    return return_act, False
            if path and len(path) > 0:
                if self.policy is not None:
                    goal_pt = self.mapper.one_map.px_to_metric(path[-1][0], path[-1][1])
                    action = self.policy.act(observations['depth'], pos[:2, 0], yaw, goal_pt)
                    # action 2 means turn left, action 3 means turn right action 1 means move forward?
                    return_act['discrete'] = self.action_lookup[action.item()]
                    return return_act, obj_found
                else:
                    path = Planning.simplify_path(np.array(path))
                    path = np.array(path).astype(np.float32)
                    rr.log("map/path_simplified",  rr.LineStrips2D(path, colors=np.repeat(np.array([0, 0, 255])[np.newaxis, :],
                                                                                path.shape[0], axis=0)))
                    for i in range(path.shape[0]):
                        path[i, :] = self.mapper.one_map.px_to_metric(path[i, 0], path[i, 1])
                    ang, lin = self.controller.control(pos, yaw, path, False)
                    return_act['continuous'] = {}
                    return_act['continuous']['linear'] = lin
                    return_act['continuous']['angular'] = ang
                    return return_act, obj_found
            else:
                return_act['discrete'] = np.random.choice(['move_forward','turn_left','turn_right'])
            return return_act, False


    def reset(self):
        self.mapper.reset()
        self.init = 12*2

    def set_query(self, query: str, full_query: str=None):
        if full_query is None:
            full_query = query
        self.mapper.set_query([query], full_query)
        
    def set_queries(self, queries: list[str], full_query: str):
        self.mapper.set_query(queries, full_query)

    def set_exploit(self):
        self.mapper.set_exploit()
