"""
Quest 3S In-The-Wild Dataset Batch Harvester & Curation Pipeline
Executes end-to-end synthesis across all available stereo scenes:
1. Simulates/Loads 90 Hz 5-frame ocular micro-tremor bursts.
2. Reconstructs High-Fidelity HR reference pairs via Iterative Back-Projection (IBP).
3. Applies physical Meta Quest 3S sensor degradation (Poisson-Gaussian noise, chromatic aberration, exposure mismatch).
4. Annotates dense sub-pixel disparity and occlusion masks using Census-LRC.
5. Emits a unified dataset manifest for training and benchmarking.
"""

import os
import sys
import glob
import json
import time
from PIL import Image
import numpy as np

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.dataset_tools.quest3s_burst_sr_synthesizer import QuestBurstSRSynthesizer, simulate_micro_tremor_burst
from scripts.dataset_tools.quest3s_dataset_curator import Quest3SSensorDegradationEngine
from scripts.dataset_tools.quest3s_wild_disparity_annotator import compute_bidirectional_disparity


def run_batch_harvest(dataset_root: str = 'datasets/Quest3S_StereoWild'):
    print("=" * 90)
    print(">>> BATCH HARVESTING & CURATION PIPELINE FOR QUEST 3S STEREO IN-THE-WILD")
    print("=" * 90)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Processing Hardware: {device}")

    base_dir = os.path.join(ROOT_DIR, dataset_root)
    hr_raw_dir = os.path.join(base_dir, 'train', 'hr')
    out_burst_hr_dir = os.path.join(base_dir, 'curated_burst_hr')
    out_ann_dir = os.path.join(base_dir, 'curated_annotations')
    out_lr4_dir = os.path.join(base_dir, 'curated_lr_x4')

    for d in [out_burst_hr_dir, out_ann_dir, out_lr4_dir]:
        os.makedirs(d, exist_ok=True)

    left_paths = sorted(glob.glob(os.path.join(hr_raw_dir, '*_L.png')))
    print(f"[*] Discovered {len(left_paths)} stereo scenes in {hr_raw_dir}")

    synthesizer = QuestBurstSRSynthesizer(scale_factor=4)
    degrader = Quest3SSensorDegradationEngine()

    manifest = []
    total_t0 = time.time()

    for idx, lp in enumerate(left_paths, 1):
        rp = lp.replace('_L.png', '_R.png')
        scene_name = os.path.basename(lp).replace('_L.png', '')

        if not os.path.exists(rp):
            continue

        print(f"\n[{idx}/{len(left_paths)}] Processing Scene: '{scene_name}'")
        t_scene = time.time()

        img_l_raw = np.array(Image.open(lp).convert('RGB'), dtype=np.float32) / 255.0
        img_r_raw = np.array(Image.open(rp).convert('RGB'), dtype=np.float32) / 255.0

        H, W, _ = img_l_raw.shape

        # 1. Simulate 5-frame 90 Hz burst with involuntary head micro-tremor (0.45 px jitter)
        burst_l = simulate_micro_tremor_burst(img_l_raw, scale=4, num_frames=5, jitter_std_lr=0.45)
        burst_r = simulate_micro_tremor_burst(img_r_raw, scale=4, num_frames=5, jitter_std_lr=0.45)

        # 2. IBP Multi-Frame Burst Super-Resolution Synthesis
        sr_l, sr_r, burst_meta = synthesizer.process_stereo_burst(burst_l, burst_r)

        # Save Restored High-Resolution Ground Truth
        hr_l_path = os.path.join(out_burst_hr_dir, f"{scene_name}_burst_L.png")
        hr_r_path = os.path.join(out_burst_hr_dir, f"{scene_name}_burst_R.png")
        Image.fromarray((sr_l * 255.0).astype(np.uint8)).save(hr_l_path)
        Image.fromarray((sr_r * 255.0).astype(np.uint8)).save(hr_r_path)

        # 3. Create Real-World Degraded LR x4 with Sensor Noise Engine
        t_sr_l = torch.from_numpy(sr_l).permute(2, 0, 1).unsqueeze(0).to(device)
        t_sr_r = torch.from_numpy(sr_r).permute(2, 0, 1).unsqueeze(0).to(device)

        # Area downsample to 1/4x
        t_lr_l = torch.nn.functional.interpolate(t_sr_l, size=(H // 4, W // 4), mode='area')
        t_lr_r = torch.nn.functional.interpolate(t_sr_r, size=(H // 4, W // 4), mode='area')

        # Apply Quest 3S sensor noise (lux = 250, typical indoor lighting)
        noisy_lr_l = degrader.apply_poisson_gaussian_noise(t_lr_l, lux=250.0)
        noisy_lr_r = degrader.apply_poisson_gaussian_noise(t_lr_r, lux=250.0)
        noisy_lr_l, noisy_lr_r = degrader.apply_exposure_mismatch(noisy_lr_l, noisy_lr_r, mismatch_pct=0.08)

        lr_l_np = (noisy_lr_l.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        lr_r_np = (noisy_lr_r.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        lr4_l_path = os.path.join(out_lr4_dir, f"{scene_name}_lr4_L.png")
        lr4_r_path = os.path.join(out_lr4_dir, f"{scene_name}_lr4_R.png")
        Image.fromarray(lr_l_np).save(lr4_l_path)
        Image.fromarray(lr_r_np).save(lr4_r_path)

        # 4. Dense Sub-Pixel Disparity & LRC Occlusion Annotation
        with torch.no_grad():
            disp_l, disp_r, lrc_err, valid_m, occ_m = compute_bidirectional_disparity(t_sr_l, t_sr_r, min_disp=0, max_disp=48)

        disp_np = disp_l.squeeze().cpu().numpy()
        occ_np = (occ_m.squeeze().cpu().numpy() * 255.0).astype(np.uint8)

        # Export 16-bit Disparity Map (scale factor 256 for sub-pixel precision)
        disp_16bit = np.clip(disp_np * 256.0, 0, 65535).astype(np.uint16)
        disp_path = os.path.join(out_ann_dir, f"{scene_name}_disp16.png")
        occ_path = os.path.join(out_ann_dir, f"{scene_name}_occ.png")
        Image.fromarray(disp_16bit).save(disp_path)
        Image.fromarray(occ_np).save(occ_path)

        dt_scene = time.time() - t_scene
        mean_disp = float(np.mean(disp_np))
        occ_rate = float(np.mean(occ_np > 128)) * 100.0

        print(f"    Restored HR Sharpness Gain: Left +{burst_meta['left_eye']['sharpness_gain_pct']:.2f}% | Right +{burst_meta['right_eye']['sharpness_gain_pct']:.2f}%")
        print(f"    Sub-pixel Mean Disparity:   {mean_disp:.2f} px | Occlusion Rate: {occ_rate:.2f}%")
        print(f"    Elapsed Time:               {dt_scene:.2f}s")

        manifest.append({
            "scene": scene_name,
            "hr_left": hr_l_path,
            "hr_right": hr_r_path,
            "lr_left": lr4_l_path,
            "lr_right": lr4_r_path,
            "disparity_map": disp_path,
            "occlusion_mask": occ_path,
            "mean_disparity": round(mean_disp, 2),
            "occlusion_percent": round(occ_rate, 2),
            "burst_sharpness_gain_left": burst_meta['left_eye']['sharpness_gain_pct'],
            "burst_sharpness_gain_right": burst_meta['right_eye']['sharpness_gain_pct'],
            "resolution": f"{W}x{H}"
        })

    total_time = time.time() - total_t0
    manifest_path = os.path.join(base_dir, 'quest3s_curated_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 90)
    print(f"[+] BATCH HARVESTING COMPLETED: {len(manifest)} scenes processed in {total_time:.2f}s")
    print(f"[+] Manifest exported to: {manifest_path}")
    print("=" * 90)


if __name__ == '__main__':
    run_batch_harvest()
