from spock import spock

@spock
class PlanningConf:
    percentile_exploitation: float
    frontier_depth: float
    no_nav_radius: float
    yolo_confidence: float
    filter_detections_depth: bool
    consensus_filtering: bool
    allow_replan: bool
    use_frontiers: bool
    allow_far_plan: bool
    using_ov: bool
    max_detect_distance: float
    obstcl_kernel_size: float
    min_goal_dist: float
    fuse_similarity: str = "sum"
    use_detector: bool = True
