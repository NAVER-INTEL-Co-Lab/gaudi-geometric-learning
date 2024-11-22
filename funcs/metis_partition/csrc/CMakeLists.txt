cmake_minimum_required(VERSION 3.10)
project(tsmetis)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)
add_definitions(-DWITH_PYTHON)
find_package(Python3 COMPONENTS Development)
add_definitions(-DWITH_METIS)
set(METIS_USE_STATIC_LIBS ON)
find_package(METIS REQUIRED)

file(GLOB HEADERS *.h)
file(GLOB OPERATOR_SOURCES *.* cpu/*.*)

add_library(${PROJECT_NAME} SHARED ${OPERATOR_SOURCES})

# Explicit linking order matters
target_link_libraries(${PROJECT_NAME} 
    PRIVATE
        ${TORCH_LIBRARIES}
        Python3::Python
        -Wl,--whole-archive
        ${METIS_LIBRARIES}
        -Wl,--no-whole-archive
)

target_include_directories(${PROJECT_NAME} 
    PUBLIC 
        ${METIS_INCLUDE_DIRS}
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)

find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME TorchSparse)

target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${HEADERS}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Ensure C++17 for the target
set_target_properties(${PROJECT_NAME} PROPERTIES 
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)