cmake_minimum_required(VERSION 3.8)
project(ugv_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch worlds models config urdf DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/obstacle_avoider.py
  scripts/generate_world.py
  scripts/coverage_planner.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
