#pragma once

/**
 * Native C++ OpenXR & Vulkan Zero-Copy Passthrough Super-Resolution Bridge
 * Target Architecture: Meta Quest 3 / Quest 3S (Snapdragon XR2 Gen 2 / Adreno 740)
 * 
 * Features:
 * 1. Zero-Allocation C++ Runtime (Static pre-allocated Ring Buffers).
 * 2. Dedicated Asynchronous Compute Queue (VK_QUEUE_COMPUTE_BIT) execution.
 * 3. Direct hardware interop with AHardwareBuffer and OpenXR Swapchain.
 */

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context handle for the Edge-StereoINR Native Engine
typedef struct EdgeStereoINRContext_T* EdgeStereoINRContext;

// Configuration descriptor for initialization
typedef struct EdgeStereoINRConfig {
    uint32_t input_width;           // Low-res passthrough width (e.g. 480)
    uint32_t input_height;          // Low-res passthrough height (e.g. 480)
    float scale_factor;             // Continuous super-resolution scale (e.g. 2.0f)
    float fovea_radius;             // Inner foveal high-definition radius (e.g. 0.35f)
    uint32_t enable_async_queue;    // 1 = Dedicated async compute queue; 0 = Synchronous
} EdgeStereoINRConfig;

/**
 * Initializes the Native C++ Engine, compiling Vulkan compute pipelines
 * and allocating static GPU descriptor sets.
 */
EdgeStereoINRContext EdgeStereoINR_Create(const EdgeStereoINRConfig* config);

/**
 * Dispatches asynchronous super-resolution execution on the target eye view.
 * Zero-copy: takes native AHardwareBuffer pointers or Vulkan VkImage handles.
 */
int32_t EdgeStereoINR_Dispatch(
    EdgeStereoINRContext ctx,
    void* left_ahb_handle,
    void* right_ahb_handle,
    void* output_ahb_handle,
    float gaze_center_x,
    float gaze_center_y
);

/**
 * Destroys the engine context and safely frees Vulkan pipelines and buffers.
 */
void EdgeStereoINR_Destroy(EdgeStereoINRContext ctx);

#ifdef __cplusplus
}
#endif
