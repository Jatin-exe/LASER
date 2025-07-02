cmake_minimum_required(VERSION 3.8)
project(beta_path_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(beta_perception REQUIRED)
find_package(beta_tracking REQUIRED)
find_package(message_filters REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/beta_path_planning)
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Generate ROS2 interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LaserPath.msg"
  DEPENDENCIES 
  std_msgs
)


# Add the executable
add_executable(planner_node src/planner.cpp)

# Link the executable to the necessary libraries
target_link_libraries(planner_node
  ${OpenCV_LIBRARIES}
)

# Add dependencies for the generated messages
ament_target_dependencies(planner_node
  rclcpp
  cv_bridge
  sensor_msgs
  geometry_msgs
  std_msgs
  beta_tracking
  Eigen3
  nlohmann_json
)

rosidl_target_interfaces(planner_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# Install the executable
install(TARGETS planner_node
  DESTINATION lib/${PROJECT_NAME}
)

# Add the executable
add_executable(camera_laser_transformer src/camera_laser_transformer.cpp)

# Add dependencies for the generated messages
ament_target_dependencies(camera_laser_transformer
  rclcpp
  geometry_msgs
  std_msgs
  nlohmann_json
)

rosidl_target_interfaces(camera_laser_transformer
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# Install the executable
install(TARGETS camera_laser_transformer
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts
install(
  PROGRAMS
    scripts/planner_viz.py
    scripts/camera_laser_transfomer.py
    scripts/camera_laser_transfomer_LS.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install the launch files if any
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Enable testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
