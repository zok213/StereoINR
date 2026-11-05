// Auto-generated Dynamic Tanh 1D Texture LUT for Vulkan Compute
#define DYT_LUT_ENTRIES 256
#define DYT_X_MIN (-8.0f)
#define DYT_X_MAX (8.0f)
#define DYT_INV_RANGE (0.06250000f)

float evaluate_dyt_lut(sampler1D lutSampler, float x) {
    float norm_u = clamp((x - DYT_X_MIN) * DYT_INV_RANGE, 0.0f, 1.0f);
    return texture(lutSampler, norm_u).r;
}
