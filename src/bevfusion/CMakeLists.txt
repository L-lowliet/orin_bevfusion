cmake_minimum_required(VERSION 3.0.2)
project(bevfusion)

set(CMAKE_BUILD_PYTHON "ON")
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3 -DENABLE_TEXT_BACKEND_STB")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
  -gencode arch=compute_87,code=compute_87
)




find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
  message_generation  # 确保包含消息生成的依赖
  pcl_conversions
  pcl_ros
)

catkin_package(
  CATKIN_DEPENDS roscpp message_runtime
)

# # 添加PCL的依赖
# find_package(PCL REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  # ${PCL_INCLUDE_DIRS}
)












include_directories(
  ${Protobuf_INCLUDE_DIRS}
  /usr/local/cuda/include
  /usr/include
  /usr/include/python3.8
  src/Lidar_AI_Solution/dependencies/stb
  src/Lidar_AI_Solution/dependencies/pybind11/include
  src/Lidar_AI_Solution/libraries/cuOSD/src
  src/Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv/include
  src/Lidar_AI_Solution/CUDA-BEVFusion/src/common
  src/Lidar_AI_Solution/CUDA-BEVFusion/src/
)

link_directories(
  /usr/local/cuda/lib64
  /usr/lib/python3.8/dist-packages
  src/Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv/lib/${arch}
  src/Lidar_AI_Solution/CUDA-BEVFusion/build
  /usr/lib
)

file(GLOB_RECURSE CORE_FILES 
  src/Lidar_AI_Solution/CUDA-BEVFusion/src/bevfusion/*.cu 
  src/Lidar_AI_Solution/CUDA-BEVFusion/src/bevfusion/*.cpp
  src/Lidar_AI_Solution/CUDA-BEVFusion/src/common/tensor.cu
  src/Lidar_AI_Solution/CUDA-BEVFusion/src/common/tensorrt.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  src/Lidar_AI_Solution/libraries/cuOSD/src/*.cpp
  src/Lidar_AI_Solution/libraries/cuOSD/src/*.cu
  src/Lidar_AI_Solution/CUDA-BEVFusion/src/common/visualize.cu
)

####################### libbevfusion_core.so ##########################
cuda_add_library(bevfusion_core SHARED 
  ${CORE_FILES}
)

target_link_libraries(bevfusion_core
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
)
#######################################################################


####################### bevfusion program #############################
cuda_add_executable(bevfusion 
  src/Lidar_AI_Solution/CUDA-BEVFusion/src/main.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(bevfusion
  bevfusion_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${catkin_LIBRARIES}
  # ${PCL_LIBRARIES}
)
#######################################################################


########################## libpybev.so ################################
if(${CMAKE_BUILD_PYTHON} STREQUAL "ON")
  cuda_add_library(pybev SHARED
  src/Lidar_AI_Solution/CUDA-BEVFusion/src/python.cpp
  )

  target_link_libraries(pybev
    bevfusion_core
    libnvinfer.so
    libspconv.so
    ${Protobuf_LIBRARIES}
    libnvinfer_plugin.so
    libpython3.8.so
    # /usr/lib/aarch64-linux-gnu/libstdc++.so.6
  )
endif()
#######################################################################
