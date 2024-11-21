cmake_minimum_required(VERSION 3.0)
project(random_walk)
set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

add_definitions(-DWITH_PYTHON)
find_package(Python3 COMPONENTS Development)

file(GLOB HEADERS csrc/*.h)

file(GLOB OPERATOR_SOURCES csrc/rw.cpp csrc/cpu/rw_cpu.* csrc/cpu/utils.h) 

add_library(random_walk SHARED rw.cpp cpu/rw_cpu.cpp)
target_link_libraries(random_walk "${TORCH_LIBRARIES}")

target_include_directories(random_walk PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Ensure C++17 for the target
set_target_properties(${PROJECT_NAME} PROPERTIES 
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)