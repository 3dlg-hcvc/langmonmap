# bosdyn
import argparse
import time

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.client.robot_command
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2, robot_state_pb2, arm_command_pb2, \
    basic_command_pb2, robot_command_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder, \
    block_until_arm_arrives, block_for_trajectory_cmd, CommandFailedErrorWithFeedback, CommandTimedOutError
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.time_sync import TimeSyncClient, TimeSyncThread
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot import Robot
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client.math_helpers import SE2Pose, SE2Velocity
from spot_utils import wait_for_message
import threading
import numpy as np
from nav_msgs.msg import Odometry as OdometryMsg


def get_path_setpoint(path, current_position, forward_distance):
    """
    Find the closest point on the path to the current position,
    then return a setpoint that is a specified distance further along the path.

    :param path: numpy array of shape [N, 2] representing the path
    :param current_position: numpy array of shape [2,] representing current position
    :param forward_distance: float, distance to look ahead on the path
    :return: numpy array of shape [2,] representing the setpoint
    """
    # Calculate distances from current position to all points on the path
    distances = np.linalg.norm(path - current_position, axis=1)

    # Find the index of the closest point
    closest_index = np.argmin(distances)

    # Initialize variables for the loop
    og_target_index = closest_index
    target_index = closest_index
    if target_index == 0:
        pass
    elif target_index == path.shape[0] - 1 or distances[target_index + 1] > distances[target_index - 1]:
        target_index -= 1

    # Project the current position onto the path segment
    segment_start = path[target_index]
    segment_end = path[target_index + 1]
    segment_vector = segment_end - segment_start
    segment_length = np.linalg.norm(segment_vector)
    segment_unit_vector = segment_vector / segment_length

    vector_to_current = current_position - segment_start
    projection_length = np.dot(vector_to_current, segment_unit_vector)
    projection_length = max(0, min(segment_length, projection_length))  # Clamp to segment

    projected_point = segment_start + projection_length * segment_unit_vector
    total_distance = -projection_length
    # Continue from the projected point
    while target_index < len(path) - 1:
        next_index = target_index + 1
        segment = path[next_index] - path[target_index]
        segment_length = np.linalg.norm(segment)

        if total_distance + segment_length > forward_distance:
            # The setpoint lies on this segment
            remaining_distance = forward_distance - total_distance
            ratio = remaining_distance / segment_length
            setpoint = path[target_index] + ratio * segment
            # print(path[target_index], setpoint)
            # print(path.shape)
            yaw = np.arctan2(segment_vector[1], segment_vector[0])
            return setpoint, yaw

        total_distance += segment_length
        target_index = next_index

        # If we've reached the last segment, break the loop
        if target_index == len(path) - 1:
            break

    # If we've reached the end of the path, return the last point
    return path[-1], None

def compute_relative_pose(current_pos, next_pos, yaw, desired_yaw):
    current_pos_robot_centric = current_pos - 0.4 * np.array([np.cos(yaw), np.sin(yaw)])

    desired_pos_robot_centric = next_pos - 0.4 * np.array([np.cos(desired_yaw), np.sin(desired_yaw)])

    delta_pose = desired_pos_robot_centric - current_pos_robot_centric

    dx = delta_pose[0]
    dy = delta_pose[1]
    desired_angle = desired_yaw
    angle_diff = (desired_angle - yaw + np.pi) % (2 * np.pi) - np.pi

    dx_body = dx * np.cos(yaw) + np.sin(yaw) * dy
    dy_body = -dx * np.sin(yaw) + np.cos(yaw) * dy

    return dx_body, dy_body, angle_diff


class SpotLease:
    """
    A class that supports execution with Python's "with" statement for safe return of
    the lease and settle-then-estop upon exit. Grants control of the Spot's motors.
    """

    def __init__(self, spot, hijack=False):
        self.lease_client = spot.robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name
        )
        if hijack:
            self.lease = self.lease_client.take()
        else:
            self.lease = self.lease_client.acquire()
        self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client)
        self.spot = spot
        self.dont_return_lease = False


    def __enter__(self):
        return self.lease

    def __exit__(self, *args, **kwargs):
        self.return_lease()

    def return_lease(self):
        # Exit the LeaseKeepAlive object
        self.lease_keep_alive.shutdown()
        print("Returning Lease")
        # Return the lease
        if not self.dont_return_lease:
            self.lease_client.return_lease(self.lease)
            # Clear lease from Spot object
            self.spot.spot_lease = None


def verify_estop(robot):
    """Verify the robot is not estopped"""
    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = 'Robot is estopped. Please use an external E-Stop client, such as the' \
                        ' estop SDK example, to configure E-Stop.'
        robot.logger.error(error_message)
        raise Exception(error_message)


class Spot:
    def __init__(self):
        # parser = argparse.ArgumentParser()
        self.command_complete = threading.Event()
        self.check_thread = None
        self.current_cmd_id = None
        self.moving = False
        self.path_lock = threading.Lock()
        self.path = None
        self.path_thread = None
        self.get_pos_fun = None
        self.max_vel = 0.4
        self.max_yaw_rate = np.pi / 4
        self.k_p = 2.0
        self.k_p_yaw = 1.0
        # bosdyn.client.util.add_base_arguments(parser)
        # options = parser.parse_args()
        # Initialize the SDK
        bosdyn.client.util.setup_logging(False)
        sdk = bosdyn.client.create_standard_sdk('ArmObjectGraspClient')
        self.robot = sdk.create_robot("192.168.50.3")
        bosdyn.client.util.authenticate(self.robot)  # Need username as environment variable
        self.robot.time_sync.wait_for_sync()
        # assert robot.has_arm(), 'Robot requires an arm to run this example.'
        verify_estop(self.robot)
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_api_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        # Ensure time synchronization
        time_sync_client = self.robot.ensure_client(TimeSyncClient.default_service_name)
        time_sync_thread = TimeSyncThread(time_sync_client)
        time_sync_thread.start()
        time_sync_thread.wait_for_sync()
        self.spot_lease = SpotLease(self, hijack=True)
        # with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), 'Robot power on failed.'
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(self.command_client, timeout_sec=10)
# 
        # self.robot = robot
        # self.command_client = command_client
        # self.robot_state_client = robot_state_client
        # self.image_client = image_client
        # self.manipulation_api_client = manipulation_api_client
        print("Spot Initialized")

    def set_path(self, new_path, get_pose_fun):
        with self.path_lock:
            self.path = new_path
            self.get_pos_fun = get_pose_fun
            if self.path_thread is None:
                self.start_path_tracking()

    def start_path_tracking(self):
        """Start the path tracking thread if it's not already running"""
        self.path_thread = threading.Thread(target=self._track_path)
        self.path_thread.start()

    def _track_path(self):
        """Thread function to continuously track the path"""
        # fetch the current position through wait for message
        while True:
            pos, yaw = self.get_pos_fun()
            with self.path_lock:
                # fetch the next set point, convert it to local and generate velocity command
                set_pt, des_yaw = get_path_setpoint(self.path, pos, 0.4)
                # we need to transform everything to robot center coords
                # current_pos, next_pos, yaw, desired_yaw
            dpos = set_pt - pos
            des_yaw = np.arctan2(dpos[1], dpos[0])
            dx, dy, dyaw = compute_relative_pose(pos, set_pt, yaw, des_yaw)
            vx = self.k_p * dx
            vy = self.k_p * dy
            dphi = self.k_p_yaw * dyaw
            speed = np.sqrt(vx**2 + vy**2)
            if speed > self.max_vel:
                vx = vx / speed * self.max_vel
                vy = vy / speed * self.max_vel
            dphi = np.clip(dphi, -self.max_yaw_rate, self.max_yaw_rate)
            # send to spot
            command = RobotCommandBuilder.synchro_velocity_command(
                v_x=vx,
                v_y=vy,
                v_rot=dphi,
                body_height=-0.1,
                build_on_command=None
            )
            print("b")
            self.current_cmd_id = self.command_client.robot_command(
                command, end_time_secs=time.time() + 0.6
            )
            time.sleep(0.1)  # Sleep for 100ms before checking again

    def set_controls(self, x_rel_body, y_rel_body, heading):
        """
        Set the controls for the robot
        :param distance_meters:
        :param heading:
        :return:
        """
        print("a")
        command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
            goal_x_rt_body=x_rel_body,
            goal_y_rt_body=y_rel_body,
            goal_heading_rt_body=heading,
            body_height=-0.1,
            frame_tree_snapshot=self.robot.get_frame_tree_snapshot(),
            build_on_command=None
        )
        print("b")
        self.current_cmd_id = self.command_client.robot_command(
            command, end_time_secs=time.time() + 20.0
        )
        self.command_complete.clear()
        self.moving = True
        # Start the checking thread
        self.check_thread = threading.Thread(target=self._check_command_status)
        self.check_thread.start()

    def _check_command_status(self):
        """
        Continuously check the status of the current command
        """
        while not self.command_complete.is_set():
            feedback = self.command_client.robot_command_feedback(self.current_cmd_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if traj_feedback.status == traj_feedback.STATUS_AT_GOAL:
                self.command_complete.set()
            else:
                time.sleep(0.05)  # Sleep for 100ms before checking again

    def is_command_complete(self):
        """
        Check if the current command is complete
        :return: True if command is complete, False otherwise
        """

        completed = self.command_complete.is_set()
        if completed:
            self.moving = False
        return completed

    def wait_for_command_completion(self, timeout=None):
        """
        Wait for the current command to complete
        :param timeout: Maximum time to wait (in seconds). None means wait indefinitely.
        :return: True if command completed, False if timed out
        """
        completed = self.command_complete.wait(timeout)
        if completed:
            self.moving = False
        return completed

    def shutdown(self):
        self.spot_lease.return_lease()

