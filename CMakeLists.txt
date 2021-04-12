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

if(MSVC)

    set(SDK_INCLUDES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/nvvfx/include)
    # Add target for NVVideoEffects
    add_library(NVVideoEffects INTERFACE)
    target_include_directories(NVVideoEffects INTERFACE ${SDK_INCLUDES_PATH})

else()
    # Add target for NVVideoEffects
    add_library(NVVideoEffects INTERFACE)

    # found in different locations depending on type of package
    find_path(VideoFX_INCLUDES
        NAMES nvVideoEffects.h
        PATHS
        /usr/local/VideoFX/include
        /usr/include/x86_64-linux-gnu
        /usr/include
        REQUIRED
        )

    target_include_directories(NVVideoEffects INTERFACE ${VideoFX_INCLUDES})
    set(SDK_INCLUDES_PATH ${VideoFX_INCLUDES})

    find_library(VideoFX_LIB
        NAMES libVideoFX.so
        PATHS
        /usr/local/VideoFX/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/lib
        REQUIRED
        NO_DEFAULT_PATH)

    target_link_libraries(NVVideoEffects INTERFACE "${VideoFX_LIB}")

    message(STATUS "VideoFX_LIB: ${VideoFX_LIB}")
    message(STATUS "SDK_INCLUDES_PATH: ${SDK_INCLUDES_PATH}")

endif()

add_subdirectory(samples)
