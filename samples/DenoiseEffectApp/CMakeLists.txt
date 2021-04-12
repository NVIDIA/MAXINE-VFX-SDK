set(SOURCE_FILES DenoiseEffectApp.cpp ../../nvvfx/src/nvVideoEffectsProxy.cpp ../../nvvfx/src/nvCVImageProxy.cpp)

# Set Visual Studio source filters
source_group("Source Files" FILES ${SOURCE_FILES})

add_executable(DenoiseEffectApp ${SOURCE_FILES})
target_include_directories(DenoiseEffectApp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
target_include_directories(DenoiseEffectApp PUBLIC ${SDK_INCLUDES_PATH})

if(MSVC)
    target_link_libraries(DenoiseEffectApp PUBLIC
        opencv346
        NVVideoEffects
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/cuda/lib/x64/cudart.lib
        )
    target_include_directories(DenoiseEffectApp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../external/cuda/include)
    set(OPENCV_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../external/opencv/bin)
    set(VFXSDK_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../../bin) # Also the location for CUDA/NVTRT/libcrypto
    set(PATH_STR "PATH=%PATH%" ${VFXSDK_PATH_STR} ${OPENCV_PATH_STR})
    set(CMD_ARG_STR "--model_dir=\"${CMAKE_CURRENT_SOURCE_DIR}/../../bin/models\" --show --webcam")
    set_target_properties(DenoiseEffectApp PROPERTIES
        FOLDER SampleApps
        VS_DEBUGGER_ENVIRONMENT "${PATH_STR}"
        VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"
        )
else()

    target_link_libraries(DenoiseEffectApp PUBLIC
        NVVideoEffects
        NVCVImage
        OpenCV
        TensorRT
        CUDA
        )
endif()
