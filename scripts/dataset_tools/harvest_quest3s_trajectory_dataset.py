"""
Quest 3S Continuous Trajectory Stereo Harvesting Pipeline
=========================================================
Converts the user's continuous 148-second XR Passthrough recording into an 8-phase
spatio-temporal stereo dataset representing a complete closed-loop navigation:
  Phase 1: Indoor Lab & Virtual UI Anchor (Horizon OS)
  Phase 2: Indoor Lab Desk & Aisle Walk
  Phase 3: Penrose Atrium & Glass Hallway Transition
  Phase 4: Main Gate Exit & Building Portal (HDR Transition)
  Phase 5: Outdoor Campus Brickwalk (High-frequency Texture & Sunlight)
  Phase 6: Outdoor Patio & Heading Back Transition
  Phase 7: Indoor Portal & Awards / Honor Display
  Phase 8: Return to Lab Desk (XR Loop Closure)
"""

import os
import sys
import math
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.dataset_tools.quest3s_burst_sr_synthesizer import QuestBurstSRSynthesizer
from scripts.dataset_tools.quest3s_wild_disparity_annotator import compute_bidirectional_disparity
from scripts.dataset_tools.harvest_quest3s_user_dataset import DepthGuidedStereoSynthesizer


def run_trajectory_harvest(
    video_path: str = r"C:\Users\Admin\Downloads\WhatsApp Video 2026-11-11 at 15.49.47.mp4",
    output_dir: str = "datasets/Quest3S_UserStereo_Harvested",
    burst_len: int = 5
):
    print("=" * 85)
    print(">>> QUEST 3S CONTINUOUS XR TRAJECTORY DATASET HARVESTING ENGINE")
    print(f"[*] Source Video: {video_path}")
    print(f"[*] Target Directory: {output_dir}")
    print("=" * 85)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")

    trajectory_phases = [
        {"name": "phase1_indoor_lab_vrui_anchor",   "pos": 300,  "split": "train", "desc": "Phòng Lab: Neo giao diện ảo Horizon OS nổi, bàn làm việc"},
        {"name": "phase2_indoor_lab_desk_walk",     "pos": 1500, "split": "train", "desc": "Lối đi phòng Lab: Di chuyển qua các bàn làm việc"},
        {"name": "phase3_penrose_hall_atrium",      "pos": 3000, "split": "train", "desc": "Sảnh đệm Penrose: Vách kính phản xạ và ánh sáng trần"},
        {"name": "phase4_main_gate_exit",           "pos": 4500, "split": "train", "desc": "Sảnh chính: Bước ra cửa cổng lớn tòa nhà (HDR transition)"},
        {"name": "phase5_outdoor_campus_brickwalk", "pos": 5700, "split": "test",  "desc": "Sân trường ngoài trời: Vân gạch tần số cao, ánh sáng tự nhiên"},
        {"name": "phase6_outdoor_patio_return",     "pos": 6900, "split": "train", "desc": "Sân hiên ngoài trời: Quay đầu hướng về tòa nhà"},
        {"name": "phase7_indoor_portal_awards",     "pos": 8100, "split": "test",  "desc": "Sảnh đón trong nhà: Tủ kính bằng khen, cúp danh dự"},
        {"name": "phase8_return_lab_desk",          "pos": 8700, "split": "test",  "desc": "Trở về phòng Lab: Khép kín quỹ đạo XR SLAM (Loop Closure)"}
    ]

    for split in ["train", "test"]:
        os.makedirs(os.path.join(output_dir, split, "hr"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "lr_x4"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "lr_x2"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "annotations"), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    synthesizer = QuestBurstSRSynthesizer(scale_factor=1)
    stereo_synth = DepthGuidedStereoSynthesizer()

    manifest = []

    for idx, sc in enumerate(trajectory_phases):
        sc_name = sc["name"]
        sc_pos = sc["pos"]
        sc_split = sc["split"]
        sc_desc = sc["desc"]

        print(f"\n[{idx+1}/{len(trajectory_phases)}] Harvesting Phase: '{sc_name}' ({sc_split.upper()}) at Frame {sc_pos} (t={sc_pos/fps:.1f}s)...")
        print(f"    Context: {sc_desc}")

        # 1. Extract burst of frames around pos
        start_frame = max(0, sc_pos - burst_len // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        burst_frames = []
        for _ in range(burst_len):
            ret, frame = cap.read()
            if not ret:
                break
            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            burst_frames.append(f_rgb)

        if len(burst_frames) < burst_len:
            print(f"    [!] Warning: Only read {len(burst_frames)} frames for {sc_name}")

        # 2. Multi-Frame Burst Merge Denoising (DIS Optical Flow + IBP)
        print("    [*] Fusing multi-frame burst with DIS optical flow & Iterative Back-Projection...")
        fused_hr_l_f32, _ = synthesizer.align_and_accumulate_burst(burst_frames)
        fused_hr_l_u8 = (np.clip(fused_hr_l_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
        fused_hr_l = cv2.cvtColor(fused_hr_l_u8, cv2.COLOR_RGB2BGR)

        # 3. Physically Calibrated DIBR Stereo Synthesis
        print("    [*] Estimating dense scene depth (MiDaS) & rendering Right Eye with 3D parallax...")
        hr_r, disp_map, valid_mask = stereo_synth.synthesize_stereo_pair(fused_hr_l, min_disp=2.0, max_disp=26.0)

        # 4. Bidirectional LRC Check for Sub-Pixel Disparity & Occlusion Masks
        print("    [*] Computing bidirectional LRC disparity & occlusion map...")
        t_l = torch.from_numpy(fused_hr_l_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        hr_r_u8 = cv2.cvtColor(hr_r, cv2.COLOR_BGR2RGB)
        t_r = torch.from_numpy(hr_r_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        disp_l, disp_r, lrc_err, v_mask, occ_m = compute_bidirectional_disparity(
            t_l, t_r, min_disp=0, max_disp=35
        )
        disp16 = (disp_l.squeeze().cpu().numpy() * 256.0).astype(np.uint16)
        occ_mask = (occ_m.squeeze().cpu().numpy() * 255.0).astype(np.uint8)

        # 5. Generate Low-Res LR x4 and LR x2 Inputs
        H, W = fused_hr_l.shape[:2]
        lr4_w, lr4_h = W // 4, H // 4
        lr2_w, lr2_h = W // 2, H // 2

        lr4_l = cv2.resize(fused_hr_l, (lr4_w, lr4_h), interpolation=cv2.INTER_CUBIC)
        lr4_r = cv2.resize(hr_r, (lr4_w, lr4_h), interpolation=cv2.INTER_CUBIC)

        lr2_l = cv2.resize(fused_hr_l, (lr2_w, lr2_h), interpolation=cv2.INTER_CUBIC)
        lr2_r = cv2.resize(hr_r, (lr2_w, lr2_h), interpolation=cv2.INTER_CUBIC)

        # 6. Save Files
        split_dir = os.path.join(output_dir, sc_split)
        cv2.imwrite(os.path.join(split_dir, "hr", f"{sc_name}_L.png"), fused_hr_l)
        cv2.imwrite(os.path.join(split_dir, "hr", f"{sc_name}_R.png"), hr_r)

        cv2.imwrite(os.path.join(split_dir, "lr_x4", f"{sc_name}_L.png"), lr4_l)
        cv2.imwrite(os.path.join(split_dir, "lr_x4", f"{sc_name}_R.png"), lr4_r)

        cv2.imwrite(os.path.join(split_dir, "lr_x2", f"{sc_name}_L.png"), lr2_l)
        cv2.imwrite(os.path.join(split_dir, "lr_x2", f"{sc_name}_R.png"), lr2_r)

        cv2.imwrite(os.path.join(split_dir, "annotations", f"{sc_name}_disp16.png"), disp16)
        cv2.imwrite(os.path.join(split_dir, "annotations", f"{sc_name}_occ.png"), occ_mask)

        # Calculate statistics
        occ_ratio = float((occ_mask > 0).mean()) * 100.0
        mean_disp = float(disp_map.mean())
        min_disp_val = float(disp_map.min())
        max_disp_val = float(disp_map.max())
        mean_lum = float(fused_hr_l.mean())

        info = {
            "name": sc_name,
            "split": sc_split,
            "timestamp_sec": float(sc_pos / fps),
            "frame_idx": sc_pos,
            "description": sc_desc,
            "dimensions": {"H": H, "W": W, "lr4_H": lr4_h, "lr4_W": lr4_w, "lr2_H": lr2_h, "lr2_W": lr2_w},
            "disparity_stats": {"min_px": min_disp_val, "max_px": max_disp_val, "mean_px": mean_disp},
            "occlusion_ratio_pct": occ_ratio,
            "mean_luminance": mean_lum
        }
        manifest.append(info)
        print(f"    [+] Saved! Disp: [{min_disp_val:.1f}, {max_disp_val:.1f}] px, Occ: {occ_ratio:.2f}%, Lum: {mean_lum:.1f}")

    cap.release()

    manifest_path = os.path.join(output_dir, "quest3s_trajectory_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 85)
    print(f"[SUCCESS] Continuous Trajectory Dataset generated at: {output_dir}")
    print(f"Manifest written to: {manifest_path}")
    print("=" * 85)


if __name__ == "__main__":
    run_trajectory_harvest()
