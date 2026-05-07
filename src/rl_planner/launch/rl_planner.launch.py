#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    task = LaunchConfiguration('task').perform(context)
    nodes = []

    octomap_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap',
        output='screen',
        remappings=[('cloud_in', 'sensor_scan')],
        parameters=[
            {'frame_id': 'map'},
            {'base_frame_id': LaunchConfiguration('base_frame')},
            {'resolution': LaunchConfiguration('map_resolution')},
            {'occupancy_min_z': 0.0},
            {'occupancy_max_z': 1.2},
            {'sensor_model.max_range': LaunchConfiguration('sensor_range')},
            {'sensor_model.hit': 1.0},
            {'sensor_model.miss': 0.45},
            {'sensor_model.max': 1.0},
            {'sensor_model.min': 0.2},
        ]
    )
    nodes.append(octomap_node)

    # Ground-truth octomap instance: convert /overall_map point cloud into a 2D occupancy map
    # for visualization and coverage reference metrics.
    octomap_groundtruth_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_groundtruth',
        output='screen',
        remappings=[
            ('cloud_in', '/overall_map'),
            ('projected_map', '/ground_truth_map'),
            ('projected_map_updates', '/ground_truth_map_updates'),
            ('octomap_binary', '/ground_truth_octomap_binary'),
            ('octomap_full', '/ground_truth_octomap_full'),
        ],
        parameters=[
            {'frame_id': 'map'},
            # Ground-truth cloud is already in map frame; keep base frame fixed to map
            # to avoid tf-related clipping.
            {'base_frame_id': 'map'},
            {'resolution': 0.3},
            # Wider z/range so the projected map covers the full scene.
            {'occupancy_min_z': -2.0},
            {'occupancy_max_z': 5.0},
            {'sensor_model.max_range': 100.0},
            {'compress_map': False},
            {'incremental_2D_projection': False},
            {'sensor_model.hit': 1.0},
            {'sensor_model.miss': 0.45},
            {'sensor_model.max': 1.0},
            {'sensor_model.min': 0.2},
        ]
    )
    nodes.append(octomap_groundtruth_node)

    if task == 'exploration':
        planner_node = Node(
            package='rl_planner',
            executable='expl_planner',
            name='rl_planner',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'publish_graph': True},
                {'node_resolution': LaunchConfiguration('node_resolution')},
                {'sensor_range': LaunchConfiguration('sensor_range')},
                {'utility_range_factor': 0.5},
                {'min_utility': 3.0},
                {'frontier_downsample_factor': 1.0},
                {'map_resolution': LaunchConfiguration('map_resolution')},
                {'waypoint_threshold': 2.0},
                {'next_waypoint_threshold': 4.0},
                {'hard_update_threshold': 10.0},
                {'frontier_cluster_range': 10.0},
                {'enable_save_mode': True},
                {'enable_dstarlite': False},
                {'replanning_frequency': 2.5},
                {'num_gen_sample': LaunchConfiguration('num_gen_sample')},
                {'model_subdir': 'expl_model'},
            ]
        )
    elif task == 'navigation':
        planner_node = Node(
            package='rl_planner',
            executable='nav_planner',
            name='nav_planner',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'publish_graph': True},
                {'node_resolution': LaunchConfiguration('node_resolution')},
                {'sensor_range': LaunchConfiguration('sensor_range')},
                {'map_resolution': LaunchConfiguration('map_resolution')},
                {'frontier_cluster_range': 10.0},
                {'replanning_frequency': 2.5},
                {'model_subdir': 'nav_model'},
                {'goal_reached_threshold': 2.0},
            ]
        )
    else:
        raise ValueError(f"Unknown task: '{task}'. Must be 'exploration' or 'navigation'.")

    nodes.append(planner_node)

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rl_rviz',
        output='log',
        arguments=['-d', PathJoinSubstitution([get_package_share_directory('rl_planner'), 'rviz', 'rviz.rviz']),
                    '--ros-args', '--log-level', 'warn'],
        respawn=True
    )
    nodes.append(rviz_node)

    return nodes


def generate_launch_description():
    ld_preload = SetEnvironmentVariable(
        'LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'
    )

    task_arg = DeclareLaunchArgument(
        'task',
        default_value='exploration',
        description="Task mode: 'exploration' or 'navigation'"
    )

    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='sensor',
        description='Robot base frame ID'
    )

    sensor_range_arg = DeclareLaunchArgument(
        'sensor_range',
        default_value='20.0',
        description='Maximum sensor range in meters'
    )

    map_resolution_arg = DeclareLaunchArgument(
        'map_resolution',
        default_value='0.4',
        description='Map resolution in meters per cell'
    )

    node_resolution_arg = DeclareLaunchArgument(
        'node_resolution',
        default_value='2.0',
        description='Path planning node resolution in meters'
    )

    num_gen_sample_arg = DeclareLaunchArgument(
        'num_gen_sample',
        default_value='4',
        description='Number of layout samples for CogniPlan predictor'
    )

    return LaunchDescription([
        ld_preload,
        task_arg,
        base_frame_arg,
        sensor_range_arg,
        map_resolution_arg,
        node_resolution_arg,
        num_gen_sample_arg,
        OpaqueFunction(function=launch_setup),
    ])
