set(SOURCE_FILES
    BatchEffectApp.cpp
    BatchUtilities.cpp
    ../../nvvfx/src/nvVideoEffectsProxy.cpp
    ../../nvvfx/src/nvCVImageProxy.cpp)


# Set Visual Studio source filters
source_group("Source Files" FILES ${SOURCE_FILES})

add_executable(BatchEffectApp ${SOURCE_FILES})
target_include_directories(BatchEffectApp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    )
target_include_directories(BatchEffectApp PUBLIC
    ${SDK_INCLUDES_PATH}
    )

target_link_libraries(BatchEffectApp PUBLIC
    opencv346
    NVVideoEffects
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/cuda/lib/x64/cudart.lib
    )

set(OPENCV_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../external/opencv/bin)
set(PATH_STR "PATH=%PATH%" ${OPENCV_PATH_STR})
set(CMD_ARG_STR "--show --in_file=\"${CMAKE_CURRENT_SOURCE_DIR}/../input/input_003054.jpg\" ")
set_target_properties(BatchEffectApp PROPERTIES
    FOLDER SampleApps
    VS_DEBUGGER_ENVIRONMENT "${PATH_STR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"
    )

#Batch denoise effect
set(SOURCE_FILES
    BatchDenoiseEffectApp.cpp
    BatchUtilities.cpp
    ../../nvvfx/src/nvVideoEffectsProxy.cpp
    ../../nvvfx/src/nvCVImageProxy.cpp)

# Set Visual Studio source filters
source_group("Source Files" FILES ${SOURCE_FILES})

add_executable(BatchDenoiseEffectApp ${SOURCE_FILES})
target_include_directories(BatchDenoiseEffectApp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    )
target_include_directories(BatchDenoiseEffectApp PUBLIC
    ${SDK_INCLUDES_PATH}
    )


target_link_libraries(BatchDenoiseEffectApp PUBLIC
	opencv346
	NVVideoEffects
	${CMAKE_CURRENT_SOURCE_DIR}/../external/cuda/lib/x64/cudart.lib
	)
target_include_directories(BatchDenoiseEffectApp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../external/cuda/include)
    
set(OPENCV_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../external/opencv/bin)
set(PATH_STR "PATH=%PATH%" ${OPENCV_PATH_STR})
set(CMD_ARG_STR "video1.mp4 video2.mp4 ")
set_target_properties(BatchDenoiseEffectApp PROPERTIES
	FOLDER SampleApps
	VS_DEBUGGER_ENVIRONMENT "${PATH_STR}"
	VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"
	)
