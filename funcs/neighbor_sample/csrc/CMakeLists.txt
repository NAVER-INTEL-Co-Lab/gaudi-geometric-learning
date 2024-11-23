cmake_minimum_required(VERSION 3.0)
project(neighbor_sample)
set(CMAKE_CXX_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

add_definitions(-DWITH_PYTHON)
find_package(Python3 COMPONENTS Development)

file(GLOB HEADERS *.h)
file(GLOB OPERATOR_SOURCES *.* cpu/*.*)

add_library(${PROJECT_NAME} SHARED ${OPERATOR_SOURCES})
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

target_include_directories(${PROJECT_NAME} PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Ensure C++17 for the target
set_target_properties(${PROJECT_NAME} PROPERTIES 
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)