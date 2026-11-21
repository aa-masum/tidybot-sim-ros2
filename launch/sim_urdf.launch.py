import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Same simulation as sim.launch.py, but the robot comes from urdf/ugv.urdf.xacro.

    robot_state_publisher publishes base_link -> lidar_link / low_lidar_link / wheels,
    so no static_transform_publisher is needed. odom -> base_link still comes from
    the Gazebo diff_drive plugin, exactly as before.
    """
    pkg = get_package_share_directory('ugv_sim')
    gazebo_ros = get_package_share_directory('gazebo_ros')

    world = os.path.join(pkg, 'worlds', 'house.sdf')
    xacro_file = os.path.join(pkg, 'urdf', 'ugv.urdf.xacro')

    # Avoid long start-up delays caused by online model database lookups
    no_db = SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', '')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world}.items())

    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}])

    # base_link origin = body centre, 0.055 m above the floor -> spawn at z = 0.06
    spawn = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'ugv', '-timeout', '180',
                   '-x', '-1.5', '-y', '0.0', '-z', '0.06'],
        output='screen')

    return LaunchDescription([no_db, gazebo, rsp, spawn])