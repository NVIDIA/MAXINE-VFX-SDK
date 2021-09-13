cmake_minimum_required(VERSION 3.10)

# Set path where samples will be installed
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH "Path to where the samples will be installed")
option(INSTALL_SDK "Install binaries into the samples folder" OFF)

project(NvVideoEffects_SDK CXX)

set(CMAKE_CONFIGURATION_TYPES "Release")

# Require C++11 and disable non-standard extensions
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN)

# Set common build path for all targets
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(SDK_INCLUDES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/nvvfx/include)
# Add target for NVVideoEffects
add_library(NVVideoEffects INTERFACE)
target_include_directories(NVVideoEffects INTERFACE ${SDK_INCLUDES_PATH})

add_subdirectory(samples)
