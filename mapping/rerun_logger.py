"""
Mapping Rerun Logger. Sets up the experiment blueprint and logs the map and robot position.
"""
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from mapping import Navigator
from onemap_utils import log_map_rerun


def log_pos(x, y):
    rr.log("map/position", rr.Points2D([[x, y]], colors=[[255, 0, 0]], radii=[1]))


def setup_blueprint_debug():
    my_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="camera",
                                  name="rgb",
                                  contents=["$origin/rgb",
                                            "$origin/detection"], ),
                rrb.Spatial2DView(origin="camera/depth")
            ),
            rrb.Vertical(
                rrb.Vertical(
                    rrb.TextLogView(origin="object_detections"),
                    rrb.TextLogView(origin="path_updates"),
                ),
                rrb.Tabs(
                    *[rrb.Spatial2DView(origin="map",
                                  name="Traversable",
                                  contents=["$origin/traversable",
                                            "$origin/position",
                                            "$origin/proj_detect",
                                            "$origin/agent_pos",
                                            "$origin/path",
                                            "$origin/ground_truth",
                                            "$origin/ground_truth_viewpoints",], ),
                        rrb.Spatial2DView(origin="map",
                                  name="Obstacle",
                                  contents=["$origin/obstacle",
                                            "$origin/position",
                                            "$origin/proj_detect",
                                            "$origin/agent_pos"
                                            "$origin/path",], ),
                        rrb.Spatial2DView(origin="map",
                                            name="Obstacle",
                                  contents=["$origin/obstacle/layer1",
                                            "$origin/position",
                                            "$origin/proj_detect",
                                            "$origin/agent_pos"
                                            "$origin/path",]),
                        rrb.Spatial2DView(origin="map",
                                            name="Obstacle",
                                  contents=["$origin/obstacle/layer2",
                                            "$origin/position",
                                            "$origin/proj_detect",
                                            "$origin/agent_pos"
                                            "$origin/path",]),
                        rrb.Spatial2DView(origin="map",
                                            name="Obstacle",
                                  contents=["$origin/obstacle/layer3",
                                            "$origin/position",
                                            "$origin/proj_detect",
                                            "$origin/agent_pos"
                                            "$origin/path",]),
                        rrb.Spatial2DView(origin="map",
                                  name="Occluded",
                                  contents=["$origin/occluded",
                                            "$origin/position",
                                            "$origin/proj_detect",
                                            "$origin/agent_pos"
                                            "$origin/path",], ),
                    ],
                ),
            ),
            rrb.Vertical(
                rrb.Tabs(
                    *[rrb.Spatial2DView(origin="map",
                                        name="Similarity",
                                        contents=
                                        ["$origin/similarity/",
                                         "$origin/proj_detect",
                                         "$origin/agent_pos",
                                         "$origin/frontiers",
                                         "$origin/frontiers_far",
                                         "$origin/position"
                                         "$origin/path",]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityLayer1",
                                        contents=
                                        ["$origin/similarity/layer1",
                                         "$origin/proj_detect",
                                         "$origin/agent_pos",
                                         "$origin/position"]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityLayer2",
                                        contents=
                                        ["$origin/similarity/layer2",
                                         "$origin/proj_detect",
                                         "$origin/agent_pos",
                                         "$origin/position"]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityLayer3",
                                        contents=
                                        ["$origin/similarity/layer3",
                                         "$origin/proj_detect",
                                         "$origin/agent_pos",
                                         "$origin/position"]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityTresholded",
                                        contents=
                                        ["$origin/similarity_th/",
                                         "$origin/proj_detect",
                                         "$origin/position"]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityTresholdedCl",
                                        contents=
                                        ["$origin/similarity_th2/",
                                         "$origin/proj_detect",
                                         "$origin/position"]),
                      ],
                ),
                rrb.Tabs(
                    rrb.Spatial2DView(origin="map",
                                      name="Explored",
                                      contents=["$origin/explored",
                                                "$origin/position",
                                                "$origin/proj_detect",
                                                "$origin/goal_pos",
                                                "$origin/largest_contour",
                                                "$origin/frontier_lines",
                                                "$origin/path",
                                                "$origin/path_simplified",
                                                "$origin/ground_truth",
                                                "$origin/ground_truth_viewpoints",
                                                "$origin/frontiers",
                                                "$origin/frontiers_far", ]),
                    rrb.Spatial2DView(origin="map",
                                      name="Scores",
                                      contents=["$origin/scores",
                                                "$origin/position",
                                                "$origin/goal_pos",
                                                "$origin/path"]),
                    rrb.Spatial2DView(origin="map",
                                      name="Unexplored",
                                      contents=["$origin/frontiers",
                                                "$origin/frontiers_far",
                                                "$origin/largest_contour",
                                                "$origin/position",
                                                "$origin/unexplored"]),
                ),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(my_blueprint)
    rr.log("map/similarity", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity/layer1", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity/layer2", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity/layer3", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity_th", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                             angle=rr.datatypes.Angle(
                                                                                 rad=-np.pi / 2))))
    rr.log("map/similarity_th2", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                              angle=rr.datatypes.Angle(
                                                                                  rad=-np.pi / 2))))
    rr.log("map/traversable", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/obstacle", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/obstacle/layer1", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/obstacle/layer2", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/obstacle/layer3", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/occluded", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/confidence", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/explored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/scores", rr.Transform3D(translation=np.array([0, 600, 0]),
                                        rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                      angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/unexplored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))

    # Point data
    rr.log("map/largest_contour", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                 rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                               angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontier_lines", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                              angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/path", rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                                       angle=rr.datatypes.Angle(
                                                                                                           rad=-np.pi))))
    rr.log("map/path_simplified",
           rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                           angle=rr.datatypes.Angle(
                                                                                               rad=-np.pi))))
    rr.log("map/position", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/proj_detect", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/goal_pos", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/ground_truth", rr.Transform3D(translation=np.array([0, 600, 0]),
                                              rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                            angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/ground_truth_viewpoints", rr.Transform3D(translation=np.array([0, 600, 0]),
                                              rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                            angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontiers", rr.Transform3D(translation=np.array([0, 600, 0]),
                                           rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                         angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontiers_far", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                             angle=rr.datatypes.Angle(rad=-np.pi))))

def setup_blueprint():
    my_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="camera",
                                  name="rgb",
                                  contents=["$origin/rgb",
                                            "$origin/detection",
                                            "$origin/seg_mask"], ),
                rrb.Spatial2DView(origin="camera/depth")
            ),
            rrb.Vertical(
                rrb.Tabs(
                    *[rrb.Spatial2DView(origin="map",
                                        name="Similarity",
                                        contents=
                                        ["$origin/similarity/",
                                         "$origin/position",
                                         "$origin/agent_pos"]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityLayer1",
                                        contents=
                                        ["$origin/similarity/layer1",
                                         "$origin/proj_detect",
                                         "$origin/agent_pos",
                                         "$origin/position"]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityLayer2",
                                        contents=
                                        ["$origin/similarity/layer2",
                                         "$origin/proj_detect",
                                         "$origin/agent_pos",
                                         "$origin/position"]),
                      rrb.Spatial2DView(origin="map",
                                        name="SimilarityLayer3",
                                        contents=
                                        ["$origin/similarity/layer3",
                                         "$origin/proj_detect",
                                         "$origin/agent_pos",
                                         "$origin/position"]),
                      ],
                ),
                rrb.Tabs(
                    rrb.Spatial2DView(origin="map",
                                      name="Explored",
                                      contents=["$origin/explored",
                                                "$origin/position",
                                                # "$origin/proj_detect",
                                                "$origin/goal_pos",
                                                # "$origin/largest_contour",
                                                # "$origin/frontier_lines",
                                                "$origin/path",
                                                "$origin/path_simplified",
                                                # "$origin/ground_truth",
                                                "$origin/frontiers",
                                                "$origin/frontiers_far",
                                                ]),
                    # rrb.Spatial2DView(origin="map",
                    #                   name="Scores",
                    #                   contents=["$origin/scores",
                    #                             "$origin/position",
                    #                             "$origin/goal_pos",
                    #                             "$origin/path"]),
                    # rrb.Spatial2DView(origin="map",
                    #                   name="Unexplored",
                    #                   contents=["$origin/frontiers",
                    #                             "$origin/frontiers_far",
                    #                             "$origin/largest_contour",
                    #                             "$origin/position",
                    #                             "$origin/unexplored"]),
                ),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(my_blueprint)
    rr.log("map/similarity", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity/layer1", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity/layer2", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity/layer3", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(
                                                                              rad=-np.pi / 2))))
    rr.log("map/similarity_th", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                             angle=rr.datatypes.Angle(
                                                                                 rad=-np.pi / 2))))
    rr.log("map/similarity_th2", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                              angle=rr.datatypes.Angle(
                                                                                  rad=-np.pi / 2))))
    rr.log("map/traversable", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/obstacle", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/obstacle/layer1", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/obstacle/layer2", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/obstacle/layer3", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/occluded", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/confidence", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/explored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/scores", rr.Transform3D(translation=np.array([0, 600, 0]),
                                        rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                      angle=rr.datatypes.Angle(rad=-np.pi / 2))))
    rr.log("map/unexplored", rr.Transform3D(translation=np.array([0, 600, 0]),
                                            rotation=rr.RotationAxisAngle(axis=[0, 0, 1],
                                                                          angle=rr.datatypes.Angle(rad=-np.pi / 2))))

    # Point data
    rr.log("map/largest_contour", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                 rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                               angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontier_lines", rr.Transform3D(translation=np.array([0, 600, 0]),
                                                rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                              angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/path", rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                                       angle=rr.datatypes.Angle(
                                                                                                           rad=-np.pi))))
    rr.log("map/path_simplified",
           rr.Transform3D(translation=np.array([0, 600, 0]), rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                                           angle=rr.datatypes.Angle(
                                                                                               rad=-np.pi))))
    rr.log("map/position", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/proj_detect", rr.Transform3D(translation=np.array([0, 600, 0]),
                                             rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                           angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/goal_pos", rr.Transform3D(translation=np.array([0, 600, 0]),
                                          rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                        angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/ground_truth", rr.Transform3D(translation=np.array([0, 600, 0]),
                                              rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                            angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/ground_truth_viewpoints", rr.Transform3D(translation=np.array([0, 600, 0]),
                                              rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                            angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontiers", rr.Transform3D(translation=np.array([0, 600, 0]),
                                           rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                         angle=rr.datatypes.Angle(rad=-np.pi))))
    rr.log("map/frontiers_far", rr.Transform3D(translation=np.array([0, 600, 0]),
                                               rotation=rr.RotationAxisAngle(axis=[1, 0, 0],
                                                                             angle=rr.datatypes.Angle(rad=-np.pi))))


class RerunLogger:
    def __init__(self, mapper: Navigator, to_file: bool, save_path: str, debug: bool = True):
        self.debug_log = debug
        self.to_file = to_file
        self.mapper = mapper
        rr.init("MON", spawn=False)
        if self.to_file:
            rr.save(save_path)
        else:
            rr.connect("127.0.0.1:9876")
        if self.debug_log:
            setup_blueprint_debug()
        else:
            setup_blueprint()
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def log_map(self):
        confidences = self.mapper.get_confidence_map()
        explored = (self.mapper.one_map.navigable_map == 1).astype(np.float32) * 0.1
        explored[confidences > 0] = 0.5
        explored[self.mapper.one_map.fully_explored_map] = 1.0
        explored[self.mapper.one_map.navigable_map == 0] = 0

        # frontiers = np.zeros((confidences.shape[0], confidences.shape[1]), dtype=np.float32)
        # for i, f in enumerate(self.mapper.frontiers):
        #     frontiers[f[:, 0, 0], f[:, 0, 1]] = 1
        # if (frontiers != 0).sum():
        #     log_map_rerun(frontiers, path="map/frontiers")

        log_map_rerun(self.mapper.one_map.navigable_map.astype(np.uint8), path="map/traversable")
        log_map_rerun(self.mapper.one_map.obstacle_map.cpu().numpy(), path="map/obstacle")
        log_map_rerun(self.mapper.one_map.occluded_map.astype(np.uint8), path="map/occluded")
        log_map_rerun(explored, path="map/explored")

        similarities = self.mapper.get_map()
        if len(similarities.shape) > 3 and similarities.shape[-1] > 1:
            similarities = similarities[0]
            # show 3 layers at most for now
            for i in range(3):
                if i >= similarities.shape[-1]:
                    break
                sim = (similarities[:,:,i] + 1.0) / 2.0
                log_map_rerun(sim, path=f"map/similarity/layer{i+1}")
            aggregated_sim = np.sum(similarities, axis=-1)
            aggregated_sim = (aggregated_sim + 1.0) / 2.0
            log_map_rerun(aggregated_sim, path="map/similarity")
        else:
            similarities = (similarities + 1.0) / 2.0
            log_map_rerun(similarities[0], path="map/similarity")

        if self.mapper.one_map.layered:
            obstacles_layered = self.mapper.one_map.obstacle_map_layered.cpu().numpy()
            if len(obstacles_layered.shape) > 2 and obstacles_layered.shape[-1] > 1:
                # show 3 layers at most for now
                for i in range(3):
                    if i >= obstacles_layered.shape[-1]:
                        break
                    log_map_rerun(obstacles_layered[:,:,i], path=f"map/obstacle/layer{i+1}")

        # log_map_rerun(confidences, path="map/confidence")

    def log_pos(self, x, y):
        px, py = self.mapper.one_map.metric_to_px(x, y)
        log_pos(px, py)
