cmake_minimum_required(VERSION 3.10.2)

project(walking)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -pthread")

# add some paths
set(PATH_TO_REPOS /home/sergio/repos)
set(MUJOCO_DIR ${PATH_TO_REPOS}/mujoco_install)
set(GLFW_DIR ${PATH_TO_REPOS}/glfw_install)
set(eigen_DIR ${PATH_TO_REPOS}/eigen_install)

# find packages
include_directories(${MUJOCO_DIR}/include)
include_directories(${MUJOCO_DIR}/include/mujoco)
include_directories(${MUJOCO_INCLUDE_DIRS})
find_library(MUJOCO_LIB mujoco REQUIRED HINTS ${MUJOCO_DIR}/lib)
find_package(glfw3 REQUIRED HINTS ${GLFW_DIR})
find_package(Eigen3 REQUIRED HINTS ${eigen_DIR})

# add the libraries
add_library(mujoco SHARED IMPORTED GLOBAL)
set_property(TARGET mujoco PROPERTY IMPORTED_LOCATION ${CMAKE_INCLUDE_DIR}/lib/libmujoco.so)

# add the executable
add_executable(s src/simulator.cpp)
target_link_libraries(s PUBLIC glfw)
target_link_libraries(s PUBLIC Eigen3::Eigen)
target_link_libraries(s PUBLIC ${MUJOCO_LIB})
target_link_directories(s PUBLIC ${MUJOCO_PATH})