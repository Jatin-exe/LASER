cmake_minimum_required(VERSION 3.8)
project(laser_framework)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find TensorRT headers and libraries.
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 /usr/lib/aarch64-linux-gnu)
message(STATUS "Found TensorRT libs ${TENSORRT_LIBRARY_INFER}")

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 /usr/lib/aarch64-linux-gnu)
message(STATUS "Found TensorRT plugin libs ${TENSORRT_LIBRARY_INFER_PLUGIN}")

# Assume that all TensorRT libraries are found in the same directory.
get_filename_component(TENSORRT_LIBRARY_DIR ${TENSORRT_LIBRARY_INFER} DIRECTORY)
link_directories(${TENSORRT_LIBRARY_DIR})

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Keypoint.msg"
  "msg/BoundingBox.msg"
  "msg/DetectionArray.msg"
  DEPENDENCIES std_msgs
)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/laser_framework)
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)



# Add the System Clock node executable
add_executable(system_clock_publihser
  src/system_clock_publisher.cpp
)

# Link dependencies to the visualization node
ament_target_dependencies(system_clock_publihser
  rclcpp
  std_msgs
)

rosidl_target_interfaces(system_clock_publihser
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# Install the visualization node
install(TARGETS system_clock_publihser
  DESTINATION lib/${PROJECT_NAME}
)

# Add the targeting node as a library
add_executable(keyframe
  src/keyframe.cpp
)

# Link dependencies to the targeting library
ament_target_dependencies(keyframe
  rclcpp
  sensor_msgs
  OpenCV
  cv_bridge
)

rosidl_target_interfaces(keyframe
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# Install the targeting library
install(TARGETS keyframe
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts
install(
  PROGRAMS
    scripts/system_targeting.py
    scripts/system_calib_beta.py
    scripts/point_and_shoot.py
    scripts/system_calib_beta_least_squares.py
    scripts/point_and_shoot_LS.py
    scripts/stop_signal.py
    scripts/storage_stats.py
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
