# StereoINR Configuration Guide (`options/`)

This directory contains YAML configuration files controlling model hyperparameters, dataset paths, training schedules, and evaluation metrics for BasicSR and StereoINR.

---

## Directory Structure

```
options/
├── test/
│   └── ASSR/
│       ├── test_StereoINR_reproduce.yml   # Table 1 & Table 2 validation with relative paths
│       ├── test_StereoINR_scale.yml       # Official testing config across arbitrary scales
│       └── ...
└── README.md
```

---

## Key Configuration Parameters

### Model Architecture (`network_g`)
```yaml
network_g:
  type: StereoINR
  upscale: 4
  in_chans: 3
  img_size: [32, 96]
  window_size: 16
  compress_ratio: 3
  squeeze_factor: 30
  conv_scale: 0.01
  overlap_ratio: 0.5
  drop_path_rate: 0.1
  img_range: 1.0
  depths: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: 'continuous'
  resi_connection: '1conv'
```

### Dataset Roots (`datasets`)
All dataset paths in `test_StereoINR_reproduce.yml` use portable relative paths:
- Middlebury: `./datasets/StereoSR/test/Middlebury/hr`
- Flickr1024: `./datasets/StereoSR/test/Flickr1024/hr`
- KITTI2012: `./datasets/StereoSR/test/KITTI2012/hr`
- KITTI2015: `./datasets/StereoSR/test/KITTI2015/hr`

### Metrics Evaluated (`val.metrics`)
- `psnr`: Peak Signal-to-Noise Ratio (RGB, 4-pixel border crop).
- `psnr_left`: Left-view PSNR without crop.
- `ssim`: Structural Similarity Index (via `scikit-image`).
- `ssim_left`: Left-view SSIM.
