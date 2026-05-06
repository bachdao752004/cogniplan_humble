#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Navigation Planner Node for CogniPlan.
Receives a goal from RViz '2D Goal Pose' and navigates the robot there
using the CogniPlan navigation policy network.
"""
import warnings
warnings.simplefilter("ignore", UserWarning)

import rclpy
from rclpy.node import Node
import numpy as np
from numpy import pad
import torch
import os
import time
import yaml
from PIL import Image
import torchvision.transforms as transforms
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float32, Header
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from .navigation_model import NavPolicyNet
from .generator import Generator, Evaluator
from .node_manager import NodeManager
from .prediction_node_manager import PredictionNodeManager
from .utils import *
from . import parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy


class NavigationRunner(Node):
    def __init__(self):
        super().__init__('nav_planner')
        self.map_info = None
        self.device = 'cpu'
        self.step = 0

        # Parameters (reuse from exploration)
        self.declare_parameter('publish_graph', True)
        self.publish_graph = self.get_parameter('publish_graph').value

        self.declare_parameter('map_resolution', parameter.CELL_SIZE)
        parameter.CELL_SIZE = self.get_parameter('map_resolution').value

        self.declare_parameter('map_free_value', parameter.FREE)
        parameter.FREE = self.get_parameter('map_free_value').value

        self.declare_parameter('map_occupied_value', parameter.OCCUPIED)
        parameter.OCCUPIED = self.get_parameter('map_occupied_value').value

        self.declare_parameter('map_unknown_value', parameter.UNKNOWN)
        parameter.UNKNOWN = self.get_parameter('map_unknown_value').value

        self.declare_parameter('sensor_range', parameter.SENSOR_RANGE)
        parameter.SENSOR_RANGE = self.get_parameter('sensor_range').value

        self.declare_parameter('node_resolution', parameter.NODE_RESOLUTION)
        parameter.NODE_RESOLUTION = self.get_parameter('node_resolution').value

        self.declare_parameter('frontier_cluster_range', parameter.CLUSTER_RANGE)
        parameter.CLUSTER_RANGE = self.get_parameter('frontier_cluster_range').value

        self.declare_parameter('replanning_frequency', 2.5)
        frequency = self.get_parameter('replanning_frequency').value

        self.declare_parameter('model_subdir', 'nav_model')
        self.model_subdir = self.get_parameter('model_subdir').value

        self.declare_parameter('goal_reached_threshold', 2.0)
        self.goal_reached_threshold = self.get_parameter('goal_reached_threshold').value

        # Navigation state
        self.model_file = "checkpoint.pth"
        self.generator_file = "generator.pt"
        self.config_file = "config.yaml"

        self.robot_location = None
        self.robot_cell = None
        self.start = None
        self.goal_position = None
        self.next_waypoint_list = []
        self.next_waypoint = None
        self.done = False
        self.navigating = False

        # Node manager for graph-based path planning
        self.node_manager = None
        self.pred_node_manager = None
        self.route_node = []

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/projected_map',
            self.get_map_callback,
            qos_profile=qos
        )
        self.loc_sub = self.create_subscription(
            Odometry,
            '/state_estimation',
            self.get_loc_callback,
            qos_profile=qos
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

        # Publishers
        self.waypoint_pub = self.create_publisher(PointStamped, '/way_point', 10)
        self.run_time_pub = self.create_publisher(Float32, '/runtime', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/nav_goal_marker', 10)
        self.path_marker_pub = self.create_publisher(Marker, '/nav_path', 10)

        self.init_model()

        self.timer = self.create_timer(1.0 / frequency, self.run)

        self.get_logger().info("Navigation Planner: Waiting for map and location data...")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.map_info is not None and self.robot_location is not None:
                break

        self.get_logger().info("Navigation Planner initialized. Use RViz '2D Goal Pose' to set a goal.")

    def init_model(self):
        """Load the navigation policy model."""
        package_share_dir = get_package_share_directory('rl_planner')
        model_dir = os.path.join(package_share_dir, self.model_subdir)
        model_file = os.path.join(model_dir, self.model_file)
        config_file = os.path.join(model_dir, self.config_file)
        generator_file = os.path.join(model_dir, self.generator_file)

        self.get_logger().info(f"Loading navigation model from: {model_dir}")

        for f in [model_file, config_file, generator_file]:
            if not os.path.exists(f):
                self.get_logger().error(f"File not found: {f}")
                raise FileNotFoundError(f"File not found: {f}")

        # Load predictor (map inpainting)
        with open(config_file, 'r', encoding='utf-8') as stream:
            config = yaml.load(stream, Loader=yaml.SafeLoader)
        generator = Generator(config['netG'], False)
        try:
            generator_state = torch.load(generator_file, map_location=self.device, weights_only=True)
        except TypeError:
            generator_state = torch.load(generator_file, map_location=self.device)
        generator.load_state_dict(generator_state)
        self.predictor = Evaluator(config, generator, device=self.device, nsample=4)

        # Load navigation policy
        self.nav_input_dim = 9  # coords(2) + utility(1) + indicator(1) + direction_vector(3) + pred_prob(1) + pred_signal(1)
        self.nav_embedding_dim = 128
        self.policy_net = NavPolicyNet(self.nav_input_dim, self.nav_embedding_dim).to(self.device)

        try:
            policy_state = torch.load(model_file, map_location=self.device, weights_only=True)
        except TypeError:
            policy_state = torch.load(model_file, map_location=self.device)
        self.policy_net.load_state_dict(policy_state['policy_model'])
        self.policy_net.eval()
        self.get_logger().info("Navigation policy loaded successfully.")

    def goal_callback(self, msg):
        """Handle goal from RViz '2D Goal Pose'."""
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        self.goal_position = np.array([goal_x, goal_y])
        self.navigating = True
        self.done = False
        self.next_waypoint_list = []
        self.next_waypoint = None
        self.route_node = [self.robot_location.copy()] if self.robot_location is not None else []
        self.get_logger().info(
            f"\033[93mNew navigation goal received: ({goal_x:.2f}, {goal_y:.2f})\033[0m"
        )
        self.publish_goal_marker()

    def run(self):
        """Main navigation loop."""
        if self.map_info is None or self.robot_location is None:
            return
        if not self.navigating or self.goal_position is None:
            return
        if self.done:
            return

        t1 = time.time()

        # Check if goal reached
        dist_to_goal = np.linalg.norm(self.robot_location - self.goal_position)
        if dist_to_goal < self.goal_reached_threshold:
            self.get_logger().info(
                f"\033[92mNavigation complete! Reached goal at ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})\033[0m"
            )
            self.done = True
            self.navigating = False
            run_time = Float32()
            run_time.data = 0.0
            self.run_time_pub.publish(run_time)
            return

        # Follow existing waypoint list if available
        if len(self.next_waypoint_list) > 0 and self.next_waypoint is not None:
            if np.linalg.norm(self.next_waypoint - self.robot_location) > parameter.THR_TO_WAYPOINT:
                return  # Still moving to current waypoint
            else:
                # Reached current waypoint, pop next
                self.next_waypoint = self.next_waypoint_list.pop(0)
                waypoint_msg = self.waypoint_wrapper(self.next_waypoint)
                self.waypoint_pub.publish(waypoint_msg)
                return

        # Plan one-step waypoint using learned navigation policy.
        if not self.plan_waypoint_with_policy():
            # Fallback: keep old A* behavior if policy observation is invalid.
            self.plan_path_to_goal_fallback()

        t2 = time.time()
        run_time = Float32()
        run_time.data = float(t2 - t1)
        self.run_time_pub.publish(run_time)

    def plan_waypoint_with_policy(self):
        """Select the next waypoint from learned navigation policy."""
        if self.node_manager is None:
            # Initialize node manager at robot start
            self.node_manager = NodeManager(self.robot_location)

        # Build/update the graph from current map.
        frontier = get_frontier_in_map(self.map_info)
        updating_map_info = self.get_updating_map(self.robot_location)

        snapped_location = self.node_manager.update_graph(
            self.robot_location, frontier, updating_map_info, self.map_info
        )
        self.robot_location = np.array(snapped_location, dtype=float)
        self.node_manager.check_valid_node(self.robot_location, self.map_info)

        # The current location may become invalid after node pruning; re-snap before rarefaction.
        current_node = self.node_manager.nodes_dict.find(self.robot_location.tolist())
        if current_node is None:
            nearest = self.node_manager.nodes_dict.nearest_neighbors(self.robot_location.tolist(), 1)
            if not nearest:
                self.get_logger().warn("Graph has no valid node near robot. Retry next cycle...")
                return False
            self.robot_location = np.array(nearest[0].data.coords, dtype=float)

        # Build rarefied key graph before creating predicted graph observation.
        self.node_manager.get_rarefied_graph(self.robot_location, self.map_info)
        key_coords = []
        for coords in self.node_manager.key_node_dict.keys():
            key_coords.append(np.array(coords))
        if len(key_coords) == 0:
            self.get_logger().warn("No key nodes available for policy inference.")
            return False
        key_coords = np.array(key_coords).reshape(-1, 2)

        # Predicted graph conditioned on inpainted map.
        self.update_predict_map()
        self.pred_node_manager = PredictionNodeManager(
            self.node_manager,
            self.pred_max_map_info,
            self.map_info,
            key_coords,
            self.robot_location,
            device=self.device,
            plot=False,
        )

        observation, graph_data = self.get_nav_observation()
        if observation is None:
            self.get_logger().warn("Navigation policy observation invalid. Retry next cycle...")
            return False

        node_coords = graph_data["node_coords"]
        edge_inputs = graph_data["edge_inputs"]
        with torch.no_grad():
            logp = self.policy_net(*observation)

        action_index = torch.argmax(logp, dim=1).long()
        next_node_index = int(edge_inputs[0, 0, action_index.item()].item())
        next_waypoint = np.array(node_coords[next_node_index], dtype=float)

        if np.linalg.norm(next_waypoint - self.robot_location) < 1e-3:
            self.get_logger().warn("Policy selected current node; waiting for next cycle.")
            return False

        self.next_waypoint_list = []
        self.next_waypoint = next_waypoint
        self.route_node.append(next_waypoint)
        waypoint_msg = self.waypoint_wrapper(self.next_waypoint)
        self.waypoint_pub.publish(waypoint_msg)
        self.publish_path_marker([self.next_waypoint])
        return True

    def get_nav_observation(self):
        """Build NavPolicyNet observation with 9-D node features."""
        [
            _base_node_inputs,
            _,
            edge_mask,
            current_index,
            _current_edge,
            _edge_padding_mask,
        ], [node_coords, utility, _guidepost, explored_sign, adjacent_matrix, neighbor_indices] = (
            self.pred_node_manager.get_predicted_observation(
                self.robot_location, self.pred_mean_map_info
            )
        )

        if len(neighbor_indices) <= 1:
            return None, None

        current_idx = int(current_index.item())
        route_indicator = np.zeros(node_coords.shape[0], dtype=np.float32)
        if len(self.route_node) > 0:
            route_set = {coords[0] + coords[1] * 1j for coords in self.route_node}
            node_complex = node_coords[:, 0] + node_coords[:, 1] * 1j
            route_indicator[np.isin(node_complex, list(route_set))] = 1.0

        # direction_vector = [unit_dx, unit_dy, distance_to_goal(clipped at 40m)]
        delta = self.goal_position.reshape(1, 2) - node_coords
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        unit_dir = np.divide(delta, np.maximum(dist, 1e-6))
        dist_clip = np.minimum(dist, 40.0)
        direction_vector = np.concatenate((unit_dir, dist_clip), axis=1)

        node_coords_norm = node_coords / 250.0
        utility_norm = utility.reshape(-1, 1) / 50.0
        indicator = route_indicator.reshape(-1, 1)
        direction_norm = direction_vector.copy()
        direction_norm[:, 2] /= 40.0
        pred_prob = self.pred_node_manager.pred_prob.reshape(-1, 1) / 100.0
        pred_signal = explored_sign.reshape(-1, 1).astype(np.float32)

        node_inputs = np.concatenate(
            (
                node_coords_norm,
                utility_norm,
                indicator,
                direction_norm,
                pred_prob,
                pred_signal,
            ),
            axis=1,
        )
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        # Keep current node as first candidate (network assumes index 0 = stay).
        sorted_neighbors = [current_idx] + [int(i) for i in neighbor_indices if int(i) != current_idx]
        edge_inputs = torch.tensor(sorted_neighbors, dtype=torch.int64).unsqueeze(0).unsqueeze(0).to(self.device)

        edge_padding_mask = torch.zeros((1, 1, edge_inputs.shape[-1]), dtype=torch.int64).to(self.device)
        edge_padding_mask[0, 0, 0] = 1  # mask "stay" action
        for idx, node_idx in enumerate(sorted_neighbors):
            if idx == 0:
                continue
            if pred_signal[node_idx, 0] == 0:
                edge_padding_mask[0, 0, idx] = 1

        observation = (
            node_inputs,
            edge_inputs,
            current_index,
            None,
            edge_padding_mask,
            edge_mask,
        )
        graph_data = {"node_coords": node_coords, "edge_inputs": edge_inputs}
        return observation, graph_data

    def update_predict_map(self):
        x_belief, mask, x_raw = self.pre_process_input()
        onehots = torch.tensor(
            [[0.333, 0.333, 0.333], [1, 0, 0], [0, 1, 0], [0, 0, 1],
             [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]
        ).unsqueeze(1).float().to(x_belief.device)
        predictions = []
        for i in range(self.predictor.nsample):
            x_inpaint = self.predictor.eval_step(x_belief, mask, onehots[i], self.map_info.map.shape)
            x_inpaint_processed = self.predictor.post_process(x_inpaint, x_raw, kernel_size=5)
            x_inpaint_processed = np.where(
                x_inpaint_processed > 0, parameter.FREE, parameter.OCCUPIED
            )
            predictions.append(x_inpaint_processed)
        self.pred_mean_map_info = MapInfo(
            np.mean(predictions, axis=0),
            self.map_info.map_origin_x,
            self.map_info.map_origin_y,
            parameter.CELL_SIZE,
        )
        self.pred_max_map_info = MapInfo(
            np.min(predictions, axis=0),
            self.map_info.map_origin_x,
            self.map_info.map_origin_y,
            parameter.CELL_SIZE,
        )

    def pre_process_input(self):
        width_in, height_in, _ = self.predictor.config['image_shape']
        height_map, width_map = self.map_info.map.shape
        pad = width_map <= width_in and height_map <= height_in
        if pad:
            pad_left = (width_in - width_map) // 2
            pad_top = (height_in - height_map) // 2
            pad_right = width_in - width_map - pad_left
            pad_bottom = height_in - height_map - pad_top
            belief = np.pad(self.map_info.map, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
        else:
            belief = self.map_info.map

        trans_belief = np.full_like(belief, 0)
        trans_belief[belief == parameter.FREE] = 255
        trans_belief[belief == parameter.OCCUPIED] = 1
        trans_belief[belief == parameter.UNKNOWN] = 127
        trans_raw = np.full_like(self.map_info.map, 0)
        trans_raw[self.map_info.map == parameter.FREE] = 255
        trans_raw[self.map_info.map == parameter.OCCUPIED] = 1
        trans_raw[self.map_info.map == parameter.UNKNOWN] = 127
        mask = np.where(belief == parameter.UNKNOWN, 255.0, 0.0)

        x_raw = Image.fromarray(trans_raw).convert('L')
        x_belief = Image.fromarray(trans_belief).convert('L')
        mask = Image.fromarray(mask).convert('1')
        if not pad:
            x_belief = transforms.Resize((width_in, height_in))(x_belief)
            mask = transforms.Resize((width_in, height_in))(mask)
        x_belief = transforms.ToTensor()(x_belief).unsqueeze(0).to(self.predictor.device)
        x_belief = x_belief.mul_(2).add_(-1)
        x_raw = transforms.ToTensor()(x_raw).unsqueeze(0).to(self.predictor.device)
        x_raw = x_raw.mul_(2).add_(-1)
        mask = transforms.ToTensor()(mask).unsqueeze(0).to(self.predictor.device)
        return x_belief, mask, x_raw

    def plan_path_to_goal_fallback(self):
        """Fallback A* planner if policy fails to produce a valid action."""
        if self.node_manager is None:
            self.node_manager = NodeManager(self.robot_location)

        goal_node = self.node_manager.nodes_dict.nearest_neighbors(
            self.goal_position.tolist(), 1
        )
        if not goal_node:
            self.get_logger().warn("Cannot find goal on graph. Retrying next cycle...")
            return

        goal_coords = goal_node[0].data.coords
        path, dist = self.node_manager.a_star(self.robot_location, goal_coords)
        if dist >= 1e8 or not path:
            self.get_logger().warn(
                f"No fallback path found to goal ({self.goal_position[0]:.1f}, {self.goal_position[1]:.1f})."
            )
            return

        self.next_waypoint_list = [np.array(coords) for coords in path]
        if len(self.next_waypoint_list) > 0:
            self.next_waypoint = self.next_waypoint_list.pop(0)
            waypoint_msg = self.waypoint_wrapper(self.next_waypoint)
            self.waypoint_pub.publish(waypoint_msg)
            self.publish_path_marker(path)

    def get_updating_map(self, location):
        """Extract the local map around the robot for graph updates."""
        updating_map_size = parameter.UPDATING_MAP_SIZE
        updating_map_origin_x = (location[0] - updating_map_size / 2)
        updating_map_origin_y = (location[1] - updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + updating_map_size
        updating_map_top_y = updating_map_origin_y + updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + parameter.CELL_SIZE * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + parameter.CELL_SIZE * (self.map_info.map.shape[0] - 1))

        updating_map_origin_x = max(updating_map_origin_x, min_x)
        updating_map_origin_y = max(updating_map_origin_y, min_y)
        updating_map_top_x = min(updating_map_top_x, max_x)
        updating_map_top_y = min(updating_map_top_y, max_y)

        updating_map_origin_x = (updating_map_origin_x // parameter.CELL_SIZE + 1) * parameter.CELL_SIZE
        updating_map_origin_y = (updating_map_origin_y // parameter.CELL_SIZE + 1) * parameter.CELL_SIZE
        updating_map_top_x = (updating_map_top_x // parameter.CELL_SIZE) * parameter.CELL_SIZE
        updating_map_top_y = (updating_map_top_y // parameter.CELL_SIZE) * parameter.CELL_SIZE

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
                       updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1] + 1,
                       updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0] + 1]

        return MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, parameter.CELL_SIZE)

    def get_map_callback(self, msg):
        """Process incoming occupancy grid from octomap."""
        delta = msg.info.resolution
        map_origin_x = msg.info.origin.position.x
        map_origin_y = msg.info.origin.position.y

        map_width = msg.info.width
        map_height = msg.info.height
        ros_map = np.array(np.array(msg.data).reshape(map_height, map_width).astype(np.int8))

        pad_size = int(parameter.NODE_RESOLUTION // parameter.CELL_SIZE + 1)
        processed_map = pad(ros_map, ((pad_size, pad_size), (pad_size, pad_size)),
                           'constant', constant_values=parameter.UNKNOWN)
        map_origin_x -= delta * pad_size
        map_origin_y -= delta * pad_size

        self.map_info = MapInfo(processed_map, map_origin_x, map_origin_y, delta)

    def get_loc_callback(self, msg):
        """Process robot odometry."""
        if self.map_info is None:
            return
        self.robot_location = np.around(
            np.array([msg.pose.pose.position.x, msg.pose.pose.position.y]), 1
        )
        if self.start is None:
            self.start = self.robot_location.copy()

    def waypoint_wrapper(self, loc):
        way_point = PointStamped()
        way_point.header.frame_id = "map"
        way_point.header.stamp = self.get_clock().now().to_msg()
        way_point.point.x = float(loc[0])
        way_point.point.y = float(loc[1])
        return way_point

    def publish_goal_marker(self):
        """Publish a marker for the goal position in RViz."""
        if self.goal_position is None:
            return
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'nav_goal'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.goal_position[0])
        marker.pose.position.y = float(self.goal_position[1])
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.5
        marker.scale.y = 1.5
        marker.scale.z = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 0.8
        self.goal_marker_pub.publish(marker)

    def publish_path_marker(self, path):
        """Publish the planned path as a line strip in RViz."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'nav_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9
        marker.pose.orientation.w = 1.0

        # Start from robot location
        start_pt = Point()
        start_pt.x = float(self.robot_location[0])
        start_pt.y = float(self.robot_location[1])
        start_pt.z = 0.1
        marker.points.append(start_pt)

        for coords in path:
            pt = Point()
            pt.x = float(coords[0])
            pt.y = float(coords[1])
            pt.z = 0.1
            marker.points.append(pt)

        self.path_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    nav_runner = None
    try:
        nav_runner = NavigationRunner()
        rclpy.spin(nav_runner)
    except KeyboardInterrupt:
        print("Shutting down Navigation Planner...")
    except Exception as e:
        print(f"Exception in Navigation Planner: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if nav_runner is not None:
            nav_runner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
