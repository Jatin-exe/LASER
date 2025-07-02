cmake_minimum_required(VERSION 3.8)
project(beta_perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find TensorRT headers and libraries
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
message("Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 /usr/lib/x86_64-linux-gnu)
message("Found TensorRT libs ${TENSORRT_LIBRARY_INFER}")

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 /usr/lib/x86_64-linux-gnu)
message("Found TensorRT plugin libs ${TENSORRT_LIBRARY_INFER_PLUGIN}")

# Assume that all TensorRT libraries are in the same directory
get_filename_component(TENSORRT_LIBRARY_DIR ${TENSORRT_LIBRARY_INFER} DIRECTORY)
link_directories(${TENSORRT_LIBRARY_DIR})

# Triton Client directories
set(TritonClientBuild_DIR "/opt/triton-client/build")
set(TritonClientThirdParty_DIR "${TritonClientBuild_DIR}/third-party")

# Print important directories
message("TritonClientBuild_DIR: ${TritonClientBuild_DIR}")
message("TritonClientThirdParty_DIR: ${TritonClientThirdParty_DIR}")

# Define paths for Triton and third-party libraries
set(ABSL_DIR "${TritonClientThirdParty_DIR}/curl/lib/cmake/absl")
set(Protobuf_DIR "${TritonClientThirdParty_DIR}/protobuf/lib/cmake/protobuf")
set(gRPC_DIR "${TritonClientThirdParty_DIR}/grpc/lib/cmake/grpc")
set(c-ares_DIR "${TritonClientThirdParty_DIR}/c-ares/lib/cmake/c-ares")
set(TritonClient_DIR "${TritonClientBuild_DIR}/install/lib/cmake/TritonClient")
set(TritonCommon_DIR "${TritonClientBuild_DIR}/install/lib/cmake/TritonCommon")

# Print third-party library paths
message("ABSL_DIR: ${ABSL_DIR}")
message("Protobuf_DIR: ${Protobuf_DIR}")
message("gRPC_DIR: ${gRPC_DIR}")
message("c-ares_DIR: ${c-ares_DIR}")
message("TritonClient_DIR: ${TritonClient_DIR}")
message("TritonCommon_DIR: ${TritonCommon_DIR}")


# Find dependencies
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
find_package(message_filters REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(CUDAToolkit REQUIRED)
# Find Triton Client package
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)


# Generate ROS2 interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Keypoint.msg"
  "msg/BoundingBox.msg"
  "msg/DetectionArray.msg"
  DEPENDENCIES std_msgs
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/beta_perception)
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${TritonCommon_DIR}/include
  ${TritonClientBuild_DIR}/install/include
)

# Add the visualization node executable
add_executable(visualization
  src/visualization.cpp
)

# Link dependencies to the visualization node
ament_target_dependencies(visualization
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  OpenCV
  message_filters
)

rosidl_target_interfaces(visualization
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# Install the visualization node
install(TARGETS visualization
  DESTINATION lib/${PROJECT_NAME}
)

# Add the perception node executable
add_executable(perception_node
  src/perception.cpp
)

# Link Triton client, Abseil, and other dependencies to the perception node
ament_target_dependencies(perception_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  OpenCV
  message_filters
)

target_link_directories(perception_node PRIVATE
  ${TritonClientBuild_DIR}/install/lib
  ${TritonCommon_DIR}/install/lib
)
target_link_libraries(perception_node
  grpcclient
)



rosidl_target_interfaces(perception_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# Install the perception node
install(TARGETS perception_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts
install(
  PROGRAMS
    scripts/detector_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Enable testing if needed
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
