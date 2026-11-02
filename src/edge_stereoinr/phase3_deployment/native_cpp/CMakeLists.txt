# CMakeLists.txt for Edge-StereoINR Native OpenXR / Vulkan Extension
# Target Platform: Meta Quest 3S (Android arm64-v8a, Snapdragon XR2 Gen 2)

cmake_minimum_required(VERSION 3.22.1)
project("edge_stereoinr_native" C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android compile flags optimized for Snapdragon Kyro CPU & Adreno 740 GPU
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -flto -DANDROID -DVK_USE_PLATFORM_ANDROID_KHR")

# Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Source files
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/edge_stereoinr_openxr.cpp
)

# Create shared library for Android OpenXR APK
add_library(edge_stereoinr SHARED ${SOURCE_FILES})

# Link against Android NDK core libraries & Vulkan
target_link_libraries(
    edge_stereoinr
    android
    log
    vulkan
)

# Compile GLSL Compute Shader to SPIR-V Bytecode
find_program(GLSLC_EXECUTABLE glslc)
if(GLSLC_EXECUTABLE)
    set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/subpixel_patch_inr.comp)
    set(SHADER_SPV ${CMAKE_CURRENT_SOURCE_DIR}/shaders/subpixel_patch_inr.spv)
    
    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${GLSLC_EXECUTABLE} -O --target-env=vulkan1.1 ${SHADER_SRC} -o ${SHADER_SPV}
        DEPENDS ${SHADER_SRC}
        COMMENT "Compiling Adreno Wave64 Compute Shader to SPIR-V..."
    )
    add_custom_target(compile_shaders ALL DEPENDS ${SHADER_SPV})
    add_dependencies(edge_stereoinr compile_shaders)
endif()
