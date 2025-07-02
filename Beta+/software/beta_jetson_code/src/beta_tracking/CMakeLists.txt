cmake_minimum_required(VERSION 3.8)
project(beta_tracking)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(beta_perception REQUIRED)
find_package(fmt REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate ROS2 interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Target.msg"
  "msg/TrackerOutput.msg"
  DEPENDENCIES 
  std_msgs
  geometry_msgs
  rcl_interfaces
  sensor_msgs
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/beta_tracking)
# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

# Add the targeting node as a library
add_executable(targeting
  src/targeting.cpp
)

# Link dependencies to the targeting library
ament_target_dependencies(targeting
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  OpenCV
)

rosidl_target_interfaces(targeting
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# Install the targeting library
install(TARGETS targeting
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(tracking
  src/tracking_node.cpp
  src/annotations.cpp
  src/FFTBoostedTemplateTracker.cpp
  src/logging.cpp
  src/utils.cpp
)

ament_target_dependencies(tracking
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  beta_perception
  message_filters
  fmt
  OpenCV
)

target_link_libraries(tracking fmt::fmt)

rosidl_target_interfaces(tracking
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# Install the targeting library
install(TARGETS tracking
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts
install(
  PROGRAMS
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
