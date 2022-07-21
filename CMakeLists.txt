cmake_minimum_required(VERSION 3.21)
project(untitled12)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_PREFIX_PATH "/home/pakos/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(untitled12 main.cpp Net.cpp)

target_link_libraries(untitled12 "${TORCH_LIBRARIES}")
set_property(TARGET untitled12 PROPERTY CXX_STANDARD 23)