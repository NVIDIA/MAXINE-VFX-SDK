set(SOURCE_FILES
    AigsEffectApp.cpp
    ../../nvvfx/src/nvVideoEffectsProxy.cpp
    ../../nvvfx/src/nvCVImageProxy.cpp)

# Set Visual Studio source filters
source_group("Source Files" FILES ${SOURCE_FILES})

add_executable(AigsEffectApp ${SOURCE_FILES})
target_include_directories(AigsEffectApp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    )
target_include_directories(AigsEffectApp PUBLIC
    ${SDK_INCLUDES_PATH}
    )

target_link_libraries(AigsEffectApp PUBLIC
    opencv346
    NVVideoEffects
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/cuda/lib/x64/cudart.lib
    )

set(OPENCV_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../external/opencv/bin)
set(PATH_STR "PATH=%PATH%" ${OPENCV_PATH_STR})
set(CMD_ARG_STR "--show --in_file=\"${CMAKE_CURRENT_SOURCE_DIR}/../input/input_003054.jpg\" ")
set_target_properties(AigsEffectApp PROPERTIES
    FOLDER SampleApps
    VS_DEBUGGER_ENVIRONMENT "${PATH_STR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"
    )