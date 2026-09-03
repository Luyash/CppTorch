cmake_minimum_required(VERSION 3.20)

project(CppTorch VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(CppTorch examples/main.cpp)

target_include_directories(CppTorch PRIVATE
    ${CMAKE_SOURCE_DIR}/external/eigen
)
