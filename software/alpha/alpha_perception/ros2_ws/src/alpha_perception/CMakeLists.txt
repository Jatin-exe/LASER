# ------------------------------------------------------------------------------
# Copyright (c) 2025 Laudando & Associates LLC
#
# This file is part of the Alpha version of the L&Aser Software.
#
# Licensed under the L&Aser Public Use License v1.0 (April 2025), based on SSPL v1.
# You may use, modify, and redistribute this file only under the terms of that license.
#
# Commercial use, integration into proprietary systems, or attempts to circumvent the
# license obligations are prohibited without an explicit commercial license.
#
# See the full license in the LICENSE file or contact chris@laudando.com for details.
#
# This license does NOT apply to any AgCeption™ branded systems or L&Aser Beta modules.
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.8)
project(alpha_perception)
enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O3)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
endif()

# Find TensorRT headers and libraries.
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 /usr/lib/x86_64-linux-gnu)
message(STATUS "Found TensorRT libs ${TENSORRT_LIBRARY_INFER}")

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 /usr/lib/x86_64-linux-gnu)
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
find_package(CUDA REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Keypoint.msg"
  "msg/BoundingBox.msg"
  "msg/DetectionArray.msg"
  DEPENDENCIES std_msgs
)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME})
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

add_executable(perception src/perception.cpp src/perception_kernels.cu)
ament_target_dependencies(perception
  rclcpp
  cv_bridge
  sensor_msgs
  std_msgs
  OpenCV
  Eigen3
)

set_target_properties(perception PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 14
)


# TensorRT libraries are linked separately
target_link_libraries(perception
  ${OpenCV_LIBS}
  ${Eigen3_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY_INFER}
  ${TENSORRT_LIBRARY_INFER_PLUGIN}
)

rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
target_link_libraries(perception "${cpp_typesupport_target}")

install(TARGETS
  perception
  DESTINATION lib/${PROJECT_NAME})


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
