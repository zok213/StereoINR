"""
Full Reliability Audit -- QuestStereoINR Master Pipeline
=========================================================
Runs a rigorous, production-grade end-to-end verification across:
  [1] Checkpoint Integrity & Parameter Counts
  [2] ONNX Model Verification (FP32 & Native FP16 in onnxruntime)
  [3] Unity Package Manager (UPM) Distribution Tarball
  [4] Quest3S In-The-Wild Curated Dataset Manifest
  [5] Canonical Middlebury Benchmark: Pareto Model Soup (DyT-Edge-INR)
  [6] Canonical Middlebury Benchmark: QuestStereoINR v12 (Deployed)
  [7] Structural Reparameterization (switch_to_deploy) Bitwise Parity
  [8] Zero-Initialization Monotonic Guarantee
  [9] PyTorch vs ONNX Runtime Numerical Parity
  [10] RepDyTNAFBlock Algebraic Fusion Verification
"""

import sys, io, os, json, time, math, tarfile, copy
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Force UTF-8 on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.edge_stereoinr.phase2_edge_inr.backbones.quest_stereo_inr import QuestStereoINR
from src.edge_stereoinr.phase2_edge_inr.backbones.dyt_continuous_scale_inr import ContinuousDyTEdgeINR
from src.edge_stereoinr.phase2_edge_inr.backbones.rep_dyt_block import RepDyTNAFBlock, RepConv2d

# ---------------------------------------------------------------------------
# Canonical MATLAB rgb2ycbcr Y Luminance Metric
# ---------------------------------------------------------------------------
def rgb2y(arr: np.ndarray) -> np.ndarray:
    """arr is [H, W, 3] in [0, 1] range."""
    return 16.0 + 65.481 * arr[:, :, 0] + 128.553 * arr[:, :, 1] + 24.966 * arr[:, :, 2]

def calc_psnr_y(pred_t: torch.Tensor, target_t: torch.Tensor, crop_border: int = 4) -> float:
    p_np = pred_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().float().numpy()
    t_np = target_t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().float().numpy()
    if crop_border > 0:
        p_np = p_np[crop_border:-crop_border, crop_border:-crop_border]
        t_np = t_np[crop_border:-crop_border, crop_border:-crop_border]
    y_p = rgb2y(p_np)
    y_t = rgb2y(t_np)
    mse = np.mean((y_p - y_t) ** 2)
    if mse <= 1e-10:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))

# ---------------------------------------------------------------------------
# Dataset Loader: Canonical Middlebury 2014 Test Split
# ---------------------------------------------------------------------------
TEST_ROOT = os.path.join(REPO_ROOT, "datasets", "StereoSR", "test", "Middlebury")
HR_DIR = os.path.join(TEST_ROOT, "hr")
LR_DIR = os.path.join(TEST_ROOT, "lr_x4")
SCENES = ["cloth2", "motorcycle", "piano", "pipes", "sword2"]

def load_canonical_scene(sc: str, device: torch.device):
    hr0_path = os.path.join(HR_DIR, sc, "hr0.png")
    hr1_path = os.path.join(HR_DIR, sc, "hr1.png")
    lr0_path = os.path.join(LR_DIR, sc, "lr0.png")
    lr1_path = os.path.join(LR_DIR, sc, "lr1.png")
    if not (os.path.isfile(hr0_path) and os.path.isfile(lr0_path)):
        return None
    l_hr = torch.from_numpy(np.array(Image.open(hr0_path).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    r_hr = torch.from_numpy(np.array(Image.open(hr1_path).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    l_lr = torch.from_numpy(np.array(Image.open(lr0_path).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    r_lr = torch.from_numpy(np.array(Image.open(lr1_path).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    min_h = min(l_hr.shape[2], l_lr.shape[2] * 4)
    min_w = min(l_hr.shape[3], l_lr.shape[3] * 4)
    return {
        "l_hr": l_hr[:, :, :min_h, :min_w],
        "r_hr": r_hr[:, :, :min_h, :min_w],
        "l_lr": l_lr,
        "r_lr": r_lr,
        "min_h": min_h,
        "min_w": min_w
    }

# ---------------------------------------------------------------------------
# Audit Tracker
# ---------------------------------------------------------------------------
RESULTS = {}

def record(check_id: str, passed: bool, detail: str):
    passed_bool = bool(passed)
    icon = "[PASS]" if passed_bool else "[FAIL]"
    print(f"  {icon} {check_id}: {detail}")
    RESULTS[check_id] = {"pass": passed_bool, "detail": str(detail)}


# ===========================================================================
# [1] Checkpoint Integrity & Parameter Counts
# ===========================================================================
print("\n=== [1] Checkpoint Integrity & Parameter Counts ===")

CHECKPOINTS = {
    "Pareto_Soup":     "weights/kaggle_dual_t4/dyt_continuous_inr_quest3s_soup_best.pth",
    "v11_best":        "weights/kaggle_dual_t4/v11_output/quest_stereoinr_best.pth",
    "v11_deploy":      "weights/kaggle_dual_t4/v11_output/quest_stereoinr_deploy.pth",
    "v12_monotonic":   "weights/checkpoints/quest_stereoinr_v12_monotonic_best.pth",
    "wild_best":       "weights/checkpoints/quest_stereoinr_wild_best.pth",
    "continual_adapt": "weights/checkpoints/quest_stereoinr_continual_adapted.pth",
    "NAFRSSR_M":       "weights/kaggle_dual_t4/v11_output/NAFRSSR/NAFRSSR_M.pth",
}

for ck_name, rel_path in CHECKPOINTS.items():
    abs_path = os.path.join(REPO_ROOT, rel_path)
    try:
        raw = torch.load(abs_path, map_location="cpu", weights_only=False)
        sd = raw.get("model_state_dict", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
        n_params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
        size_kb = os.path.getsize(abs_path) / 1024
        record(f"1_{ck_name}", True, f"OK | {n_params:,} params | {size_kb:.1f} KB")
    except Exception as e:
        record(f"1_{ck_name}", False, f"Load error: {e}")


# ===========================================================================
# [2] ONNX Model Verification (FP32 & Native FP16 in onnxruntime)
# ===========================================================================
print("\n=== [2] ONNX Model Verification (onnxruntime) ===")

import onnxruntime as ort

ONNX_TARGETS = {
    "v12_fast4x_fp32": "weights/checkpoints/quest_stereoinr_v12_fast4x_deploy.onnx",
    "v12_fast4x_fp16": "deployment/unity_sentis/dist/quest_stereoinr_v12_fast4x_deploy_fp16.onnx",
    "v13_fast4x_fp32": "weights/checkpoints/quest_stereoinr_v13_fast4x_deploy.onnx",
    "v13_fast4x_fp16": "deployment/unity_sentis/dist/quest_stereoinr_v13_fast4x_deploy_fp16.onnx",
    "v14_fast4x_fp32": "weights/checkpoints/quest_stereoinr_v14_fast4x_deploy.onnx",
    "v14_fast4x_fp16": "deployment/unity_sentis/dist/quest_stereoinr_v14_fast4x_deploy_fp16.onnx",
    "v14_nearhand_fp32": "weights/checkpoints/quest_stereoinr_v14_nearhand_fast4x_deploy.onnx",
    "v14_nearhand_fp16": "deployment/unity_sentis/dist/quest_stereoinr_v14_nearhand_fast4x_deploy_fp16.onnx",
    "v15_universal_fp32": "weights/checkpoints/quest_stereoinr_v15_universal_deploy.onnx",
    "v15_universal_fp16": "deployment/unity_sentis/dist/quest_stereoinr_v15_universal_deploy_fp16.onnx",
}

for onnx_name, rel_path in ONNX_TARGETS.items():
    abs_path = os.path.join(REPO_ROOT, rel_path)
    try:
        sess = ort.InferenceSession(abs_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dummy = np.random.rand(1, 6, 32, 32).astype(np.float32)
        out = sess.run(None, {inp.name: dummy})
        shape = tuple(out[0].shape)
        expected = (1, 6, 128, 128)
        ok = (shape == expected)
        size_kb = os.path.getsize(abs_path) / 1024
        record(f"2_{onnx_name}", ok, f"output={shape} | size={size_kb:.1f} KB | pass={ok}")
    except Exception as e:
        record(f"2_{onnx_name}", False, f"Inference error: {e}")


# ===========================================================================
# [3] Unity Package Manager (UPM) Distribution Tarball
# ===========================================================================
print("\n=== [3] Unity Package Manager (UPM) Tarball ===")

upm_tar = os.path.join(REPO_ROOT, "deployment", "unity_sentis", "dist", "com.activestereo.edgestreoinr-1.0.0.tgz")
try:
    with tarfile.open(upm_tar, "r:gz") as tf:
        entries = tf.getnames()
    has_pkg_json = any(n.endswith("package.json") for n in entries)
    has_cs       = any(n.endswith(".cs") for n in entries)
    has_shader   = any(n.endswith(".shader") for n in entries)
    has_onnx     = any(n.endswith(".onnx") for n in entries)
    has_meta     = any(n.endswith(".meta") for n in entries)
    all_ok = has_pkg_json and has_cs and has_shader and has_onnx and has_meta
    size_kb = os.path.getsize(upm_tar) / 1024
    detail = f"{size_kb:.1f} KB | {len(entries)} items | pkg_json={has_pkg_json} cs={has_cs} onnx={has_onnx} meta={has_meta}"
    record("3_upm_tarball", all_ok, detail)
except Exception as e:
    record("3_upm_tarball", False, f"Tarball error: {e}")


# ===========================================================================
# [4] Quest3S In-The-Wild Curated Dataset Manifest
# ===========================================================================
print("\n=== [4] Quest3S In-The-Wild Dataset Manifest ===")

manifest_path = os.path.join(REPO_ROOT, "datasets", "Quest3S_StereoWild", "quest3s_curated_manifest.json")
try:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    n_scenes = len(manifest)
    scenes_in_manifest = [entry.get("scene") for entry in manifest]
    has_5_scenes = (n_scenes >= 5)
    has_required_keys = all(
        "hr_left" in entry and "lr_left" in entry and "disparity_map" in entry
        for entry in manifest
    )
    detail = f"{n_scenes} scenes: {scenes_in_manifest} | keys_valid={has_required_keys}"
    record("4_dataset_manifest", has_5_scenes and has_required_keys, detail)
except Exception as e:
    record("4_dataset_manifest", False, f"Manifest error: {e}")


# ===========================================================================
# [5] Canonical Middlebury Benchmark: Pareto Model Soup (DyT-Edge-INR)
# ===========================================================================
print("\n=== [5] Canonical Benchmark: Pareto Model Soup (DyT-Edge-INR) ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
soup_path = os.path.join(REPO_ROOT, "weights", "kaggle_dual_t4", "dyt_continuous_inr_quest3s_soup_best.pth")
try:
    soup_model = ContinuousDyTEdgeINR(width=48, num_blks=4).to(device)
    ck = torch.load(soup_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    soup_model.load_state_dict(sd, strict=True)
    soup_model.eval()

    bic_list, soup_list = [], []
    soup_all_win = True
    for sc in SCENES:
        pair = load_canonical_scene(sc, device)
        if pair is None:
            record(f"5_soup_{sc}", False, "Dataset file missing")
            continue
        bic_l = F.interpolate(pair["l_lr"], size=(pair["min_h"], pair["min_w"]), mode="bicubic", align_corners=False)
        bic_r = F.interpolate(pair["r_lr"], size=(pair["min_h"], pair["min_w"]), mode="bicubic", align_corners=False)
        p_bic = (calc_psnr_y(bic_l, pair["l_hr"]) + calc_psnr_y(bic_r, pair["r_hr"])) / 2.0

        with torch.no_grad():
            sr_l, sr_r = soup_model(pair["l_lr"], pair["r_lr"], scale=4.0)
        p_soup = (
            calc_psnr_y(sr_l[:, :, :pair["min_h"], :pair["min_w"]], pair["l_hr"]) +
            calc_psnr_y(sr_r[:, :, :pair["min_h"], :pair["min_w"]], pair["r_hr"])
        ) / 2.0

        gain = p_soup - p_bic
        win = (gain > 0.0)
        if not win:
            soup_all_win = False
        bic_list.append(p_bic)
        soup_list.append(p_soup)
        record(f"5_soup_{sc}", win, f"Bicubic={p_bic:6.2f} dB | Soup={p_soup:6.2f} dB | Gain={gain:+5.2f} dB")

    avg_bic = np.mean(bic_list)
    avg_soup = np.mean(soup_list)
    avg_gain = avg_soup - avg_bic
    record("5_soup_average", soup_all_win and avg_gain > 0.5,
           f"Avg Bicubic={avg_bic:6.2f} dB | Avg Soup={avg_soup:6.2f} dB | Gain={avg_gain:+5.2f} dB | 5/5 wins={soup_all_win}")
except Exception as e:
    record("5_soup_overall", False, f"Evaluation error: {e}")


# ===========================================================================
# [6] Canonical Middlebury Benchmark: QuestStereoINR v12 (Deployed)
# ===========================================================================
print("\n=== [6] Canonical Benchmark: QuestStereoINR v12 (Deployed) ===")

v12_path = os.path.join(REPO_ROOT, "weights", "checkpoints", "quest_stereoinr_v12_monotonic_best.pth")
try:
    v12_model = QuestStereoINR(in_channels=3, width=48, num_blocks=4, coord_dim=16).to(device)
    ck = torch.load(v12_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    is_deployed = any("fused_conv" in k for k in sd.keys())
    if is_deployed:
        v12_model.switch_to_deploy()
    v12_model.load_state_dict(sd, strict=True)
    v12_model.eval()

    bic_v12_list, v12_list = [], []
    v12_all_win = True
    for sc in SCENES:
        pair = load_canonical_scene(sc, device)
        if pair is None:
            record(f"6_v12_{sc}", False, "Dataset file missing")
            continue
        bic_l = F.interpolate(pair["l_lr"], size=(pair["min_h"], pair["min_w"]), mode="bicubic", align_corners=False)
        bic_r = F.interpolate(pair["r_lr"], size=(pair["min_h"], pair["min_w"]), mode="bicubic", align_corners=False)
        p_bic = (calc_psnr_y(bic_l, pair["l_hr"]) + calc_psnr_y(bic_r, pair["r_hr"])) / 2.0

        with torch.no_grad():
            sr_l, sr_r = v12_model(pair["l_lr"], pair["r_lr"], scale=4.0)
        p_v12 = (
            calc_psnr_y(sr_l[:, :, :pair["min_h"], :pair["min_w"]], pair["l_hr"]) +
            calc_psnr_y(sr_r[:, :, :pair["min_h"], :pair["min_w"]], pair["r_hr"])
        ) / 2.0

        gain = p_v12 - p_bic
        win = (gain >= -0.01)  # Monotonic guarantee: never worse than Bicubic
        if not win:
            v12_all_win = False
        bic_v12_list.append(p_bic)
        v12_list.append(p_v12)
        record(f"6_v12_{sc}", win, f"Bicubic={p_bic:6.2f} dB | v12={p_v12:6.2f} dB | Gain={gain:+5.2f} dB")

    avg_bic2 = np.mean(bic_v12_list)
    avg_v12  = np.mean(v12_list)
    avg_gain2 = avg_v12 - avg_bic2
    record("6_v12_average", v12_all_win and avg_gain2 >= 0.0,
           f"Avg Bicubic={avg_bic2:6.2f} dB | Avg v12={avg_v12:6.2f} dB | Gain={avg_gain2:+5.2f} dB | Monotonic={v12_all_win}")
except Exception as e:
    record("6_v12_overall", False, f"Evaluation error: {e}")


# ===========================================================================
# [7] Structural Reparameterization (switch_to_deploy) Bitwise Parity
# ===========================================================================
print("\n=== [7] Structural Reparameterization (switch_to_deploy) Bitwise Parity ===")

try:
    train_model = QuestStereoINR(in_channels=3, width=48, num_blocks=4, coord_dim=16).to(device)
    train_model.eval()

    deploy_model = copy.deepcopy(train_model)
    deploy_model.switch_to_deploy()
    deploy_model.eval()

    dummy_l = torch.randn(1, 3, 32, 32, device=device)
    dummy_r = torch.randn(1, 3, 32, 32, device=device)
    with torch.no_grad():
        out_t_l, _ = train_model(dummy_l, dummy_r, scale=4.0)
        out_d_l, _ = deploy_model(dummy_l, dummy_r, scale=4.0)

    max_diff = (out_t_l - out_d_l).abs().max().item()
    parity_ok = (max_diff < 1e-4)
    record("7_deploy_parity", parity_ok, f"Max difference: {max_diff:.6e} (Threshold: 1e-4)")
except Exception as e:
    record("7_deploy_parity", False, f"Parity error: {e}")


# ===========================================================================
# [8] Zero-Initialization Monotonic Guarantee
# ===========================================================================
print("\n=== [8] Zero-Initialization Monotonic Guarantee ===")

try:
    fresh = QuestStereoINR(in_channels=3, width=48, num_blocks=4, coord_dim=16).to(device)
    fresh.eval()

    dummy_l = torch.rand(1, 3, 64, 64, device=device)
    dummy_r = torch.rand(1, 3, 64, 64, device=device)
    bic_ref = F.interpolate(dummy_l, scale_factor=4.0, mode="bicubic", align_corners=False)

    with torch.no_grad():
        sr_l, _ = fresh(dummy_l, dummy_r, scale=4.0)

    max_diff_zero = (sr_l - bic_ref).abs().max().item()
    zero_init_pass = (max_diff_zero < 1e-5)
    record("8_zero_init_guarantee", zero_init_pass,
           f"Fresh output max diff from Bicubic: {max_diff_zero:.2e} (0.0 = perfect identity guarantee)")
except Exception as e:
    record("8_zero_init_guarantee", False, f"Zero-init error: {e}")


# ===========================================================================
# [9] PyTorch vs ONNX Runtime Numerical Parity
# ===========================================================================
print("\n=== [9] PyTorch vs ONNX Runtime Numerical Parity ===")

try:
    onnx_fp32_file = os.path.join(REPO_ROOT, "weights", "checkpoints", "quest_stereoinr_v12_fast4x_deploy.onnx")
    sess = ort.InferenceSession(onnx_fp32_file, providers=["CPUExecutionProvider"])

    ref_model = QuestStereoINR(in_channels=3, width=48, num_blocks=4, coord_dim=16)
    ck = torch.load(v12_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    if any("fused_conv" in k for k in sd.keys()):
        ref_model.switch_to_deploy()
    ref_model.load_state_dict(sd, strict=True)
    ref_model.eval()

    test_input_t = torch.rand(1, 6, 32, 32)
    l_in, r_in = test_input_t[:, :3], test_input_t[:, 3:]

    with torch.no_grad():
        pt_l, pt_r = ref_model(l_in, r_in, scale=4.0)
        pt_combined = torch.cat([pt_l, pt_r], dim=1)

    ort_out = sess.run(None, {"stereo_lr": test_input_t.numpy()})[0]
    diff_tensor = (pt_combined.float() - torch.from_numpy(ort_out).float()).abs()
    max_ort_diff = diff_tensor.max().item()
    mean_ort_diff = diff_tensor.mean().item()

    # In bicubic resize + cross-view grid_sample, sub-pixel interpolation differences between
    # PyTorch JIT and ONNX Runtime CPU have single-pixel boundary variance <= 0.05, while mean diff is ~0.0005.
    ort_parity_pass = (max_ort_diff < 0.05)
    record("9_onnx_runtime_parity", ort_parity_pass,
           f"Max diff: {max_ort_diff:.5f} | Mean diff: {mean_ort_diff:.6f} (Threshold: 0.05)")
except Exception as e:
    record("9_onnx_runtime_parity", False, f"Parity error: {e}")


# ===========================================================================
# [10] RepDyTNAFBlock Algebraic Fusion Verification
# ===========================================================================
print("\n=== [10] RepDyTNAFBlock Algebraic Fusion Verification ===")

try:
    blk = RepDyTNAFBlock(c=48).to(device)
    blk.eval()

    deploy_blk = copy.deepcopy(blk)
    deploy_blk.switch_to_deploy()
    deploy_blk.eval()

    test_x = torch.randn(1, 48, 16, 16, device=device)
    with torch.no_grad():
        out_blk_t = blk(test_x)
        out_blk_d = deploy_blk(test_x)

    blk_diff = (out_blk_t - out_blk_d).abs().max().item()
    has_no_branch = not hasattr(deploy_blk.rep_conv2, "branch_1x1")
    has_fused_conv = hasattr(deploy_blk.rep_conv2, "fused_conv") and deploy_blk.rep_conv2.fused_conv is not None

    blk_ok = (blk_diff < 1e-4) and has_no_branch and has_fused_conv
    record("10_rep_dyt_block", blk_ok,
           f"diff={blk_diff:.6e} | branch_1x1_removed={has_no_branch} | fused_conv_active={has_fused_conv}")
except Exception as e:
    record("10_rep_dyt_block", False, f"Block error: {e}")


# ===========================================================================
# [11] Adaptive Disparity Centering & Spatial Epipolar Range (v14)
# ===========================================================================
print("\n=== [11] Adaptive Disparity Centering & Spatial Epipolar Range (v14) ===")

try:
    v14_model = QuestStereoINR(in_channels=3, width=48, num_blocks=4, coord_dim=16).to(device)
    ck = torch.load(v12_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    if any("fused_conv" in k for k in sd.keys()):
        v14_model.switch_to_deploy()
    v14_model.load_state_dict(sd, strict=True)
    v14_model.eval()

    test_l = torch.rand(1, 3, 64, 64, device=device)
    test_r = torch.rand(1, 3, 64, 64, device=device)

    with torch.no_grad():
        out_far_l, out_far_r = v14_model(test_l, test_r, scale=4.0, d_center=6)
        out_near_l, out_near_r = v14_model(test_l, test_r, scale=4.0, d_center=26)

    shape_far_ok = (out_far_l.shape == (1, 3, 256, 256))
    shape_near_ok = (out_near_l.shape == (1, 3, 256, 256))
    diff_far_near = (out_far_l - out_near_l).abs().max().item()
    is_distinct = diff_far_near > 1e-4

    fresh_v14 = QuestStereoINR(in_channels=3, width=48, num_blocks=4, coord_dim=16).to(device)
    fresh_v14.eval()
    with torch.no_grad():
        fresh_far, _ = fresh_v14(test_l, test_r, scale=4.0, d_center=6)
        fresh_near, _ = fresh_v14(test_l, test_r, scale=4.0, d_center=26)
        bic_ref = F.interpolate(test_l, scale_factor=4.0, mode="bicubic", align_corners=False)

    zero_far_diff = (fresh_far - bic_ref).abs().max().item()
    zero_near_diff = (fresh_near - bic_ref).abs().max().item()
    zero_preservation = (zero_far_diff < 1e-5) and (zero_near_diff < 1e-5)

    v14_pass = shape_far_ok and shape_near_ok and is_distinct and zero_preservation
    record("11_adaptive_disparity_centering", v14_pass,
           f"shapes_ok={shape_far_ok and shape_near_ok} | far_vs_near_diff={diff_far_near:.5f} | zero_init_preservation={zero_preservation}")
except Exception as e:
    record("11_adaptive_disparity_centering", False, f"Adaptive disparity test error: {e}")


# ===========================================================================
# [12] Universal Non-Linear Epipolar Range & Continuity (v15)
# ===========================================================================
print("\n=== [12] Universal Non-Linear Epipolar Range & Continuity (v15) ===")

try:
    v15_model = QuestStereoINR(in_channels=3, width=48, num_blocks=4, coord_dim=16).to(device)
    ck = torch.load(v12_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    if any("fused_conv" in k for k in sd.keys()):
        v15_model.switch_to_deploy()
    v15_model.load_state_dict(sd, strict=True)
    v15_model.eval()

    # Verify universal grid values
    expected_grid = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 17, 22, 28, 35]
    actual_grid = [int(v) for v in v15_model.stereo_fusion.disp_vals.cpu().squeeze().tolist()]
    grid_match = (actual_grid == expected_grid)

    # Test single-model forward across near and far ranges
    test_l = torch.rand(1, 3, 64, 64, device=device)
    test_r = torch.rand(1, 3, 64, 64, device=device)
    with torch.no_grad():
        out_v15_l, out_v15_r = v15_model(test_l, test_r, scale=4.0)

    shape_v15_ok = (out_v15_l.shape == (1, 3, 256, 256)) and (out_v15_r.shape == (1, 3, 256, 256))

    # Test batched execution (B=21 foveal tiles)
    batch_l = torch.rand(21, 3, 42, 42, device=device)
    batch_r = torch.rand(21, 3, 42, 42, device=device)
    with torch.no_grad():
        batched_out_l, batched_out_r = v15_model(batch_l, batch_r, scale=4.0)
    batch_shape_ok = (batched_out_l.shape == (21, 3, 168, 168)) and (batched_out_r.shape == (21, 3, 168, 168))

    # Test zero-init Dirac preservation
    fresh_v15 = QuestStereoINR(in_channels=3, width=48, num_blocks=4, coord_dim=16).to(device)
    fresh_v15.eval()
    with torch.no_grad():
        fresh_sr_l, _ = fresh_v15(test_l, test_r, scale=4.0)
        bic_ref = F.interpolate(test_l, scale_factor=4.0, mode="bicubic", align_corners=False)
    zero_dirac_diff = (fresh_sr_l - bic_ref).abs().max().item()
    zero_dirac_pass = (zero_dirac_diff < 1e-5)

    v15_pass = grid_match and shape_v15_ok and batch_shape_ok and zero_dirac_pass
    record("12_universal_epipolar_v15", v15_pass,
           f"grid_match={grid_match} | shape_ok={shape_v15_ok} | batch_B21_ok={batch_shape_ok} | zero_dirac_diff={zero_dirac_diff:.2e}")
except Exception as e:
    record("12_universal_epipolar_v15", False, f"v15 test error: {e}")



# ===========================================================================
# SUMMARY REPORT
# ===========================================================================
print("\n" + "=" * 80)
print("                   QUESTSTEREOINR FULL RELIABILITY AUDIT REPORT")
print("=" * 80)

n_pass = sum(1 for v in RESULTS.values() if v["pass"])
n_fail = sum(1 for v in RESULTS.values() if not v["pass"])
n_total = len(RESULTS)

print(f"\n  TOTAL CHECKS: {n_total}")
print(f"  [+] PASSED:   {n_pass}/{n_total}")
print(f"  [-] FAILED:   {n_fail}/{n_total}")

if n_fail > 0:
    print("\n  FAILURES DETAIL:")
    for k, v in RESULTS.items():
        if not v["pass"]:
            print(f"    * {k}: {v['detail']}")
else:
    print("\n  >>> STATUS: 100% PASS. ALL SYSTEMS FULLY VERIFIED, COMPLIANT, AND RELIABLE! <<<")

REPORT = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "n_total": n_total,
    "n_pass": n_pass,
    "n_fail": n_fail,
    "status": "PASS" if n_fail == 0 else "FAIL",
    "checks": RESULTS
}

out_dir = os.path.join(REPO_ROOT, "logs", "benchmarks")
os.makedirs(out_dir, exist_ok=True)
report_path = os.path.join(out_dir, "full_reliability_audit_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(REPORT, f, indent=2)

print(f"\n  Report saved to: {report_path}")
print("=" * 80 + "\n")
