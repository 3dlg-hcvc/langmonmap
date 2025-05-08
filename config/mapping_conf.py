from spock import spock

@spock
class MappingConf:
    n_points: int
    size: int
    agent_radius: float
    blur_kernel_size: float
    obstacle_map_threshold: float
    fully_explored_threshold: float
    checked_map_threshold: float

    depth_factor: float
    gradient_factor: float

    optimal_object_distance: float
    optimal_object_factor: float

    obstacle_min: float
    obstacle_max: float

    filter_stairs: bool
    floor_level: float
    floor_threshold: float

    layered: bool = False
    z_bins_lower: float = 0.0
    z_bins_upper: float = 3.0
    z_bins_step: float = 3.0

    probabilistic_fusion: bool = True

