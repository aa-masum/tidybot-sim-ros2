import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('ugv_sim')
    gazebo_ros = get_package_share_directory('gazebo_ros')

    world = os.path.join(pkg, 'worlds', 'house.sdf')
    robot = os.path.join(pkg, 'models', 'ugv.sdf')

    # Avoid long start-up delays caused by online model database lookups
    no_db = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world}.items())

    spawn = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-file', robot, '-entity', 'ugv', '-timeout', '180',
                   '-x', '-1.5', '-y', '0.0', '-z', '0.01'],
        output='screen')

    # base_link is 0.055 m above the floor.
    # lidar: 0.105 m above floor -> z = 0.05 ; low sensor: 0.035 m -> z = -0.02
    tf_lidar = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0.05',
                   '--frame-id', 'base_link', '--child-frame-id', 'lidar_link'],
        parameters=[{'use_sim_time': True}])

    tf_low = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['--x', '0.16', '--y', '0', '--z', '-0.02',
                   '--frame-id', 'base_link', '--child-frame-id', 'low_lidar_link'],
        parameters=[{'use_sim_time': True}])

    return LaunchDescription([no_db, gazebo, spawn, tf_lidar, tf_low])