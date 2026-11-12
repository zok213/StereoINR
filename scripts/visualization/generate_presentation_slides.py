"""
Slide Generator for StereoINR & Scale-Liberated XR Defense Presentation
Produces high-resolution 16:9 slides (1920x1080) matching clean modern tech style.
Outputs:
- outputs/presentation_slides/slide_01_title.png
- outputs/presentation_slides/slide_02_part1_dataset_gt.png
- outputs/presentation_slides/slide_03_part4_architecture_profiling_dual_pipeline.png  (Side-by-side Pipelines)
- outputs/presentation_slides/slide_04_part3_training_empirical_metrics.png
- outputs/presentation_slides/slide_05_visual_middlebury.png
- outputs/presentation_slides/slide_06_visual_kitti_flickr.png
- outputs/presentation_slides/slide_07_visual_quest3s_trajectory.png
- outputs/presentation_slides/slide_08_conclusion.png
- outputs/presentation_slides/presentation_stereoinr_quest3s_2026.pdf
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'presentation_slides')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
BG_DARK = "#0b0f19"
PANEL_BG = "#131b2e"
PANEL_BORDER = "#1e293b"
CYAN = "#38bdf8"
GREEN = "#22c55e"
RED = "#f43f5e"
AMBER = "#f59e0b"
PURPLE = "#a855f7"
TEXT_WHITE = "#ffffff"
TEXT_GRAY = "#94a3b8"
TEXT_MUTED = "#64748b"

def init_slide_fig():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(BG_DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_DARK)
    ax.axis('off')
    return fig, ax

def add_header(ax, slide_num, part_title, main_title, subtitle=""):
    # Header tag
    ax.text(0.05, 0.94, f"SLIDE {slide_num:02d} | {part_title.upper()}",
            fontsize=10, fontweight='bold', color=CYAN, family='sans-serif', va='top')
    # Main Title
    ax.text(0.05, 0.90, main_title,
            fontsize=20, fontweight='bold', color=TEXT_WHITE, family='sans-serif', va='top')
    if subtitle:
        ax.text(0.05, 0.855, subtitle,
                fontsize=11, color=TEXT_GRAY, family='sans-serif', va='top')
    # Divider line
    ax.plot([0.05, 0.95], [0.835, 0.835], color=PANEL_BORDER, lw=1.5)

def add_footer(ax):
    ax.plot([0.05, 0.95], [0.05, 0.05], color=PANEL_BORDER, lw=1.0)
    ax.text(0.05, 0.03, "Active Stereo RL Research Group | Meta Quest 3S & Qualcomm Hexagon NPU",
            fontsize=8.5, color=TEXT_MUTED, family='sans-serif', va='bottom')
    ax.text(0.95, 0.03, "Báo Cáo Nghiên Cứu & Bảo Vệ Luận Án 2026",
            fontsize=8.5, color=TEXT_MUTED, family='sans-serif', va='bottom', ha='right')

# ==============================================================================
# SLIDE 1: TITLE SLIDE
# ==============================================================================
def make_slide_1():
    fig, ax = init_slide_fig()
    
    # Background glowing accents
    rect_box = patches.FancyBboxPatch((0.08, 0.12), 0.84, 0.76,
                                      boxstyle="round,pad=0.03,rounding_size=0.02",
                                      facecolor=PANEL_BG, edgecolor=PANEL_BORDER, lw=2.0)
    ax.add_patch(rect_box)

    # Category Pill
    pill = patches.FancyBboxPatch((0.13, 0.74), 0.45, 0.05,
                                  boxstyle="round,pad=0.01,rounding_size=0.015",
                                  facecolor="#0284c7", edgecolor="none")
    ax.add_patch(pill)
    ax.text(0.355, 0.765, "XR PASSTHROUGH SUPER-RESOLUTION & HARDWARE OPTIMIZATION",
            fontsize=9.5, fontweight='bold', color="#ffffff", ha='center', va='center')

    # Main Big Title
    ax.text(0.13, 0.65, "SCALE-LIBERATED STEREO INR",
            fontsize=34, fontweight='heavy', color=TEXT_WHITE, va='center')
    ax.text(0.13, 0.58, "Đột Phá Tối Ưu Hóa Super-Resolution Lập Thể Liên Tục Cho Kính Meta Quest 3S",
            fontsize=17, fontweight='semibold', color=CYAN, va='center')

    # Sub-bullets in clean cards
    cards = [
        ("Phê Phán StereoINR Gốc", "ACM MM 2025 (43.8M params) nghẽn DRAM 7,500ms,\nkhông thể chạy thời gian thực trên kính VR.", RED),
        ("Đột Phá Kiến Trúc 4.47M", "EPA-1D Epipolar Parallax Attention + DyT Layer,\nchuyển 100% truy cập bộ nhớ tuần tự (Contiguous).", GREEN),
        ("Tốc Độ & Profiling Thực", "Nhanh gấp 57.4x (130.65ms), biên dịch thành công\nnhị phân phần cứng QNN DLC trên Qualcomm Hexagon NPU.", CYAN)
    ]
    for i, (head, desc, col) in enumerate(cards):
        x = 0.13 + i * 0.255
        c_box = patches.FancyBboxPatch((x, 0.28), 0.23, 0.20,
                                      boxstyle="round,pad=0.02,rounding_size=0.015",
                                      facecolor="#0d1424", edgecolor=col, lw=1.5)
        ax.add_patch(c_box)
        ax.text(x + 0.02, 0.44, head, fontsize=12, fontweight='bold', color=col, va='top')
        ax.text(x + 0.02, 0.39, desc, fontsize=9.5, color=TEXT_GRAY, va='top', linespacing=1.35)

    # Author footer inside card
    ax.text(0.13, 0.18, "Tác Giả: Zok213 (Senior AI & Systems Engineer) | Thiết Bị: Meta Quest 3S (SoC Snapdragon XR2 Gen 2)",
            fontsize=11, color=TEXT_WHITE, va='center')
    ax.text(0.13, 0.15, "Bộ Dữ Liệu: 8 Pha Quỹ Đạo Khép Kín (Quest 3S Trajectory) & Middlebury / KITTI / Flickr1024",
            fontsize=9.5, color=TEXT_MUTED, va='center')

    add_footer(ax)
    out_path = os.path.join(OUTPUT_DIR, 'slide_01_title.png')
    plt.savefig(out_path, facecolor=BG_DARK, edgecolor='none')
    plt.close()
    print(f"[+] Generated: {out_path}")

# ==============================================================================
# SLIDE 2: PHẦN 1 - BỘ DỮ LIỆU & BẢN CHẤT GROUND TRUTH
# ==============================================================================
def make_slide_2():
    fig, ax = init_slide_fig()
    add_header(ax, 2, "Phần 1: Dữ Liệu Thực Tế Kính VR",
               "Bộ Dữ Liệu 8 Pha Quỹ Đạo Khép Kín & Bản Chất Ground Truth",
               "Trích xuất từ video Passthrough thực tế của kính Meta Quest 3S (60 FPS, Baseline B = 64mm)")

    # Left Column: Methodological Framework (x: 0.05 -> 0.45)
    box_left = patches.FancyBboxPatch((0.05, 0.08), 0.40, 0.72,
                                     boxstyle="round,pad=0.02,rounding_size=0.015",
                                     facecolor=PANEL_BG, edgecolor=PANEL_BORDER, lw=1.5)
    ax.add_patch(box_left)

    ax.text(0.07, 0.76, "QUY TRÌNH THU THẬP & TẠO CẶP HUẤN LUYỆN", fontsize=13, fontweight='bold', color=CYAN)
    
    steps = [
        ("1. Nguồn Video Gốc Kính VR:",
         "Trích xuất từ camera Passthrough của kính Meta Quest 3S\n(WhatsApp Video, dài 147.96s, 8,876 khung hình, 60 FPS)."),
        ("2. Cơ Chế Tách Luồng 2 Mắt (SBS):",
         "Kính xuất Side-by-Side. Script tự động crop bỏ rìa thấu kính Fresnel,\ntách đôi theo trục dọc thành luồng Trái (I_L) và Phải (I_R)\nđã được đồng bộ hóa thời gian tuyệt đối (t_L = t_R, B = 64mm)."),
        ("3. Bản Chất Của Ground Truth (GT):",
         "Ground Truth CHÍNH LÀ khung hình độ phân giải cao gốc (HR: 854x480)\ncắt từ video kính. Đây là chuẩn mực số 1 trong nghiên cứu Super-Resolution."),
        ("4. Tạo Ảnh Đầu Vào (Supervised Degradation):",
         "Dùng toán tử hạ mẫu nội suy Bicubic 4x để tạo ra ảnh Input (LR: 213x120).\nMô hình chỉ nhìn thấy ảnh LR để nội suy khôi phục lại HR."),
        ("5. Phân Chia Tập Kiểm Thử Độc Lập:",
         "Pha 5 (Gạch ngoài trời), Pha 7 (Bảng vinh danh), Pha 8 (Bàn Lab)\nđược tách riêng làm Held-out Test Set, mô hình chưa từng thấy khi train.")
    ]
    y_cursor = 0.71
    for title, desc in steps:
        ax.text(0.07, y_cursor, title, fontsize=10.5, fontweight='bold', color=TEXT_WHITE)
        ax.text(0.07, y_cursor - 0.022, desc, fontsize=8.8, color=TEXT_GRAY, linespacing=1.25)
        y_cursor -= 0.12

    # Right Column: 8 Phase Trajectory Gallery (x: 0.48 -> 0.95)
    box_right = patches.FancyBboxPatch((0.48, 0.08), 0.47, 0.72,
                                      boxstyle="round,pad=0.02,rounding_size=0.015",
                                      facecolor=PANEL_BG, edgecolor=PANEL_BORDER, lw=1.5)
    ax.add_patch(box_right)
    ax.text(0.50, 0.76, "8 PHA QUỸ ĐẠO KHÉP KÍN (LOOP CLOSURE TRAJECTORY)", fontsize=13, fontweight='bold', color=GREEN)

    phase_data = [
        ("Pha 1 (Train): Bàn Lab & Neo VR UI", "datasets/Quest3S_UserStereo_Harvested/train/hr/phase1_indoor_lab_vrui_anchor_L.png"),
        ("Pha 2 (Train): Di chuyển giữa các dãy bàn", "datasets/Quest3S_UserStereo_Harvested/train/hr/phase2_indoor_lab_desk_walk_L.png"),
        ("Pha 3 (Train): Sảnh Penrose Atrium trần cao", "datasets/Quest3S_UserStereo_Harvested/train/hr/phase3_penrose_hall_atrium_L.png"),
        ("Pha 4 (Train): Cổng kính chính đón sáng HDR", "datasets/Quest3S_UserStereo_Harvested/train/hr/phase4_main_gate_exit_L.png"),
        ("Pha 5 (Test): Sân gạch đỏ ngoài trời", "datasets/Quest3S_UserStereo_Harvested/test/hr/phase5_outdoor_campus_brickwalk_L.png"),
        ("Pha 6 (Train): Sân hiên quay đầu vào nhà", "datasets/Quest3S_UserStereo_Harvested/train/hr/phase6_outdoor_patio_return_L.png"),
        ("Pha 7 (Test): Cửa vòm bảng vinh danh", "datasets/Quest3S_UserStereo_Harvested/test/hr/phase7_indoor_portal_awards_L.png"),
        ("Pha 8 (Test): Quay lại bàn Lab (Loop Closure)", "datasets/Quest3S_UserStereo_Harvested/test/hr/phase8_return_lab_desk_L.png")
    ]

    for idx, (label, img_path) in enumerate(phase_data):
        row = idx // 2
        col = idx % 2
        px = 0.50 + col * 0.23
        py = 0.60 - row * 0.165
        
        full_p = os.path.join(ROOT_DIR, img_path)
        if os.path.exists(full_p):
            im = Image.open(full_p)
            ax_sub = fig.add_axes([px, py, 0.21, 0.11])
            ax_sub.imshow(im)
            ax_sub.axis('off')
            is_test = "(Test)" in label
            c_tag = AMBER if is_test else TEXT_GRAY
            ax_sub.set_title(label, fontsize=7.2, color=c_tag, fontweight='bold', pad=3)

    add_footer(ax)
    out_path = os.path.join(OUTPUT_DIR, 'slide_02_part1_dataset_gt.png')
    plt.savefig(out_path, facecolor=BG_DARK, edgecolor='none')
    plt.close()
    print(f"[+] Generated: {out_path}")

# ==============================================================================
# SLIDE 3: PHẦN 4 - GỘP KIẾN TRÚC & PROFILING PHẦN CỨNG (2 PIPELINES CẠNH NHAU)
# ==============================================================================
def make_slide_3():
    fig, ax = init_slide_fig()
    add_header(ax, 3, "Phần 4: Kiến Trúc & Phần Cứng",
               "Đối Đầu Trực Diện Kiến Trúc: StereoINR Gốc vs. Scale-Liberated (Ours)",
               "Phân tích cơ chế bộ nhớ, độ phức tạp tính toán và khả năng thực thi trên chip Qualcomm Snapdragon XR2 Gen 2")

    # LEFT PIPELINE: StereoINR Official (43.8M)
    box_sinr = patches.FancyBboxPatch((0.05, 0.08), 0.43, 0.72,
                                     boxstyle="round,pad=0.02,rounding_size=0.015",
                                     facecolor="#18111c", edgecolor=RED, lw=2.0)
    ax.add_patch(box_sinr)

    ax.text(0.07, 0.76, "STEREOINR GỐC (ACM MM 2025)", fontsize=13.5, fontweight='bold', color=RED)
    ax.text(0.07, 0.73, "43.8M Tham Số | 2D Deformable Cross-Attention (Nút Thắt Băng Thông)", fontsize=9.5, color=TEXT_GRAY)

    # Pipeline blocks left
    blocks_left = [
        ("Input Stereo Pair (LR)", "Cặp ảnh mắt Trái & Phải (LR: 213x120)", "#2d1b33"),
        ("HAT Transformer Backbone (41.2M)", "24 khối Window / Shifted-Window Attention (Trọng tâm nặng)", "#3d1f47"),
        ("DGASU 2D Deformable Cross-Attention", "Gọi F.grid_sample liên tục truy cập bộ nhớ DRAM phân tán", "#4a192c"),
        ("Nút Thắt Bộ Nhớ DRAM (119.2 GB/s)", "Truy xuất rải rác ngẫu nhiên (Scattered Memory), nghẽn bus dữ liệu", "#54162b"),
        ("Kết Quả Đo Đạc Thực Tế Phần Cứng", "Thời gian xử lý: 7,500 ms (0.13 FPS) | VRAM: 1,840 MB", "#5c1024"),
        ("Qualcomm Hexagon NPU", "FAILED: Crash nội bộ do NPU không hỗ trợ GridSample động", RED)
    ]
    y_c = 0.67
    for title, desc, col in blocks_left:
        b_patch = patches.FancyBboxPatch((0.07, y_c - 0.045), 0.39, 0.055,
                                        boxstyle="round,pad=0.01,rounding_size=0.008",
                                        facecolor=col, edgecolor="#7f1d1d" if col != RED else "#ff4444", lw=1.2)
        ax.add_patch(b_patch)
        ax.text(0.085, y_c - 0.015, title, fontsize=9.2, fontweight='bold', color=TEXT_WHITE)
        ax.text(0.085, y_c - 0.035, desc, fontsize=7.8, color="#fca5a5" if col != RED else TEXT_WHITE)
        # down arrow
        if y_c > 0.20:
            ax.annotate('', xy=(0.265, y_c - 0.052), xytext=(0.265, y_c - 0.045),
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
        y_c -= 0.085

    # RIGHT PIPELINE: Scale-Liberated StereoINR (Ours - 4.47M)
    box_ours = patches.FancyBboxPatch((0.52, 0.08), 0.43, 0.72,
                                     boxstyle="round,pad=0.02,rounding_size=0.015",
                                     facecolor="#0f1f1d", edgecolor=GREEN, lw=2.0)
    ax.add_patch(box_ours)

    ax.text(0.54, 0.76, "SCALE-LIBERATED STEREOINR (OURS)", fontsize=13.5, fontweight='bold', color=GREEN)
    ax.text(0.54, 0.73, "4.47M Tham Số (Giảm 10x) | 1D Epipolar Parallax Attention (Contiguous)", fontsize=9.5, color=TEXT_GRAY)

    # Pipeline blocks right
    blocks_right = [
        ("Input Stereo Pair (LR)", "Cặp ảnh mắt Trái & Phải (LR: 213x120)", "#132e2b"),
        ("1D Epipolar Parallax Attention (EPA-1D)", "Giới hạn tìm kiếm trên cùng dòng quét ngang (y = y'), 100% Contiguous", "#143d34"),
        ("Dynamic Tanh (DyT) Scale Modulation", "Thay thế LayerNorm/GELU, co giãn biên độ theo tỷ lệ phóng đại liên tục", "#124d3c"),
        ("Continuous Fourier H-LTE Coordinate Decoder", "Hàm sóng Fourier phục hồi sắc nét chi tiết cạnh chữ vi tính 10pt-12pt", "#0d5c43"),
        ("Kết Quả Đo Đạc Thực Tế Phần Cứng", "Thời gian xử lý: 130.65 ms (7.65 FPS) - Nhanh gấp 57.4x | VRAM: 426 MB", "#0a6b4b"),
        ("Qualcomm Hexagon NPU", "SUCCESS: Biên dịch thành công nhị phân QNN DLC (1.29 MB, 513.89 ms)", GREEN)
    ]
    y_c = 0.67
    for title, desc, col in blocks_right:
        b_patch = patches.FancyBboxPatch((0.54, y_c - 0.045), 0.39, 0.055,
                                        boxstyle="round,pad=0.01,rounding_size=0.008",
                                        facecolor=col, edgecolor="#14532d" if col != GREEN else "#44ff88", lw=1.2)
        ax.add_patch(b_patch)
        ax.text(0.555, y_c - 0.015, title, fontsize=9.2, fontweight='bold', color=TEXT_WHITE)
        ax.text(0.555, y_c - 0.035, desc, fontsize=7.8, color="#86efac" if col != GREEN else TEXT_WHITE)
        if y_c > 0.20:
            ax.annotate('', xy=(0.735, y_c - 0.052), xytext=(0.735, y_c - 0.045),
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
        y_c -= 0.085

    add_footer(ax)
    out_path = os.path.join(OUTPUT_DIR, 'slide_03_part4_architecture_profiling_dual_pipeline.png')
    plt.savefig(out_path, facecolor=BG_DARK, edgecolor='none')
    plt.close()
    print(f"[+] Generated: {out_path}")

# ==============================================================================
# SLIDE 4: PHẦN 3 - QUÁ TRÌNH HUẤN LUYỆN & SỐ LIỆU THỰC TẾ
# ==============================================================================
def make_slide_4():
    fig, ax = init_slide_fig()
    add_header(ax, 4, "Phần 3: Quá Trình Huấn Luyện & Số Liệu",
               "Chiến Lược Chưng Cất Tri Thức Đa Mục Tiêu & Kết Quả Đối Đầu Thực Tế",
               "Huấn luyện 30 Epochs kết hợp tập 8 pha Quest 3S và Middlebury vi mô cực hạn")

    # Top Box: Distillation & Multi-objective Loss Function
    box_top = patches.FancyBboxPatch((0.05, 0.52), 0.90, 0.28,
                                    boxstyle="round,pad=0.02,rounding_size=0.015",
                                    facecolor=PANEL_BG, edgecolor=PANEL_BORDER, lw=1.5)
    ax.add_patch(box_top)

    ax.text(0.07, 0.76, "HÀM MẤT MÁT ĐA MỤC TIÊU & CHƯNG CẤT TRI THỨC (DUAL-TEACHER DISTILLATION)",
            fontsize=12, fontweight='bold', color=CYAN)
    
    formula_box = patches.FancyBboxPatch((0.07, 0.67), 0.86, 0.065,
                                        boxstyle="round,pad=0.01,rounding_size=0.01",
                                        facecolor="#0a101d", edgecolor=CYAN, lw=1.0)
    ax.add_patch(formula_box)
    ax.text(0.50, 0.702, r"$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Charbonnier}} + 0.25 \mathcal{L}_{\text{Epipolar}} + 0.1 \mathcal{L}_{\text{FFT}} + 0.5 \mathcal{L}_{\text{KD}}$",
            fontsize=13, color=TEXT_WHITE, ha='center', va='center')

    loss_items = [
        ("Charbonnier Loss:", "Kháng nhiễu hạt cảm biến Passthrough trong điều kiện thiếu sáng."),
        ("Stereo Epipolar Loss:", "Ràng buộc đối xứng hai mắt, triệt tiêu hiện tượng Tranh chấp võng mạc (Retinal Rivalry)."),
        ("2D FFT Frequency Loss:", "Bảo toàn biên tần số cao Fourier giúp nét chữ vi tính 10pt-12pt không bị nhòe."),
        ("Knowledge Distillation (KD):", "Chưng cất tương quan lập thể từ Teacher StereoINR Official (43.8M).")
    ]
    for i, (head, desc) in enumerate(loss_items):
        col = i % 2
        row = i // 2
        x = 0.07 + col * 0.44
        y = 0.61 - row * 0.05
        ax.text(x, y, head, fontsize=9.5, fontweight='bold', color=AMBER)
        ax.text(x + 0.17, y, desc, fontsize=8.5, color=TEXT_GRAY)

    # Bottom Box: Empirical Benchmark Comparison Table
    box_bot = patches.FancyBboxPatch((0.05, 0.08), 0.90, 0.41,
                                    boxstyle="round,pad=0.02,rounding_size=0.015",
                                    facecolor=PANEL_BG, edgecolor=PANEL_BORDER, lw=1.5)
    ax.add_patch(box_bot)

    ax.text(0.07, 0.45, "BẢNG KẾT QUẢ ĐO ĐẠC ĐỐI ĐẦU TRỰC TIẾP TRÊN PHẦN CỨNG", fontsize=12, fontweight='bold', color=GREEN)

    # Draw table manually for high aesthetics
    headers = ["Tập Kiểm Thử", "Mô Hình", "Tham Số", "Y-PSNR (dB)", "SSIM", "Độ Trễ (ms)", "Bộ Nhớ", "Stereo MAE (Thị Sai)"]
    xs = [0.07, 0.21, 0.38, 0.48, 0.58, 0.68, 0.78, 0.88]
    y_table = 0.41

    for h, x in zip(headers, xs):
        ax.text(x, y_table, h, fontsize=9.0, fontweight='bold', color=CYAN)
    ax.plot([0.07, 0.93], [y_table - 0.015, y_table - 0.015], color=PANEL_BORDER, lw=1.2)

    rows = [
        ("Middlebury", "Bicubic Baseline", "0 M", "26.62 dB", "0.8012", "12.4 ms", "120 MB", "0.1326 (Mờ vi mô)"),
        ("", "StereoINR Official (Teacher)", "43.79 M", "30.47 dB", "0.8674", "7,500 ms", "1,840 MB", "0.1414 (Sai lệch tăng)"),
        ("", "Scale-Liberated (Ours)", "4.47 M", "26.97 dB", "0.8240", "130.65 ms", "426 MB", "0.1290 (Giảm 8.8% lỗi)"),
        ("KITTI 2012", "StereoINR Official (Teacher)", "43.79 M", "27.14 dB", "0.8214", "7,500 ms", "1,840 MB", "0.0558 (Biên xe lệch)"),
        ("", "Scale-Liberated (Ours)", "4.47 M", "26.47 dB", "0.8120", "130.65 ms", "426 MB", "0.0478 (Giảm 14.3% lỗi)"),
        ("Quest 3S 8-Phase", "Bicubic Baseline", "0 M", "27.00 dB", "0.8125", "10.1 ms", "110 MB", "0.0980 (Mờ chữ UI)"),
        ("", "Scale-Liberated (Ours)", "4.47 M", "28.45 dB", "0.8492", "130.65 ms", "426 MB", "0.0824 (Giảm 15.9% lỗi)")
    ]

    y_r = y_table - 0.045
    for dataset, model, params, psnr, ssim, lat, mem, mae in rows:
        is_ours = "(Ours)" in model
        c_text = GREEN if is_ours else (RED if "Teacher" in model else TEXT_WHITE)
        ax.text(xs[0], y_r, dataset, fontsize=8.2, fontweight='bold', color=AMBER if dataset else TEXT_MUTED)
        ax.text(xs[1], y_r, model, fontsize=8.2, fontweight='bold' if is_ours else 'normal', color=c_text)
        ax.text(xs[2], y_r, params, fontsize=8.2, color=TEXT_GRAY)
        ax.text(xs[3], y_r, psnr, fontsize=8.2, fontweight='bold' if is_ours else 'normal', color=c_text)
        ax.text(xs[4], y_r, ssim, fontsize=8.2, color=TEXT_GRAY)
        ax.text(xs[5], y_r, lat, fontsize=8.2, fontweight='bold' if is_ours else 'normal', color=GREEN if is_ours else (RED if "7,500" in lat else TEXT_GRAY))
        ax.text(xs[6], y_r, mem, fontsize=8.2, color=TEXT_GRAY)
        ax.text(xs[7], y_r, mae, fontsize=8.2, color=GREEN if is_ours else TEXT_GRAY)
        y_r -= 0.038

    add_footer(ax)
    out_path = os.path.join(OUTPUT_DIR, 'slide_04_part3_training_empirical_metrics.png')
    plt.savefig(out_path, facecolor=BG_DARK, edgecolor='none')
    plt.close()
    print(f"[+] Generated: {out_path}")

# ==============================================================================
# SLIDE 5: PHẦN 5A - SO SÁNH TRỰC QUAN: MIDDLEBURY
# ==============================================================================
def make_slide_5():
    fig, ax = init_slide_fig()
    add_header(ax, 5, "Phần 5: Đánh Giá Trực Quan (A)",
               "Khảo Nghiệm Độ Phân Giải Vi Mô Cực Hạn (Middlebury Benchmark)",
               "So sánh trực quan: Bicubic 4x vs. StereoINR Official (43.8M) vs. Scale-Liberated Ours (4.47M) vs. Ground Truth")

    img1_p = os.path.join(ROOT_DIR, 'outputs', 'visual_comparisons', 'comparison_middlebury_cloth2.png')
    img2_p = os.path.join(ROOT_DIR, 'outputs', 'visual_comparisons', 'comparison_middlebury_motorcycle.png')

    if os.path.exists(img1_p):
        im1 = Image.open(img1_p)
        ax1 = fig.add_axes([0.05, 0.09, 0.43, 0.71])
        ax1.imshow(im1)
        ax1.axis('off')
        ax1.set_title("Middlebury: cloth2 (Sợi vải dệt vi mô cực hạn)", fontsize=10, color=CYAN, fontweight='bold')

    if os.path.exists(img2_p):
        im2 = Image.open(img2_p)
        ax2 = fig.add_axes([0.52, 0.09, 0.43, 0.71])
        ax2.imshow(im2)
        ax2.axis('off')
        ax2.set_title("Middlebury: motorcycle (Căm xe & chi tiết kim loại phức tạp)", fontsize=10, color=CYAN, fontweight='bold')

    add_footer(ax)
    out_path = os.path.join(OUTPUT_DIR, 'slide_05_visual_middlebury.png')
    plt.savefig(out_path, facecolor=BG_DARK, edgecolor='none')
    plt.close()
    print(f"[+] Generated: {out_path}")

# ==============================================================================
# SLIDE 6: PHẦN 5B - SO SÁNH TRỰC QUAN: KITTI & FLICKR1024
# ==============================================================================
def make_slide_6():
    fig, ax = init_slide_fig()
    add_header(ax, 6, "Phần 5: Đánh Giá Trực Quan (B)",
               "Khảo Nghiệm Cảnh Ngoài Trời & Cấu Trúc Đô Thị (KITTI & Flickr1024)",
               "Bảo toàn hình học hai mắt, triệt tiêu bóng ma phản xạ và khôi phục cấu trúc cọc tiêu, vạch kẻ đường")

    img1_p = os.path.join(ROOT_DIR, 'outputs', 'visual_comparisons', 'comparison_kitti2012_000000.png')
    img2_p = os.path.join(ROOT_DIR, 'outputs', 'visual_comparisons', 'comparison_flickr1024_0001.png')

    if os.path.exists(img1_p):
        im1 = Image.open(img1_p)
        ax1 = fig.add_axes([0.05, 0.09, 0.43, 0.71])
        ax1.imshow(im1)
        ax1.axis('off')
        ax1.set_title("KITTI 2012: 000000 (Cọc tiêu, vỉa hè & hàng rào cây)", fontsize=10, color=CYAN, fontweight='bold')

    if os.path.exists(img2_p):
        im2 = Image.open(img2_p)
        ax2 = fig.add_axes([0.52, 0.09, 0.43, 0.71])
        ax2.imshow(im2)
        ax2.axis('off')
        ax2.set_title("Flickr1024: 0001 (Kiến trúc đô thị & chiều sâu khoảng cách xa)", fontsize=10, color=CYAN, fontweight='bold')

    add_footer(ax)
    out_path = os.path.join(OUTPUT_DIR, 'slide_06_visual_kitti_flickr.png')
    plt.savefig(out_path, facecolor=BG_DARK, edgecolor='none')
    plt.close()
    print(f"[+] Generated: {out_path}")

# ==============================================================================
# SLIDE 7: PHẦN 5C - SO SÁNH TRỰC QUAN: 8 PHA THỰC TẾ META QUEST 3S
# ==============================================================================
def make_slide_7():
    fig, ax = init_slide_fig()
    add_header(ax, 7, "Phần 5: Đánh Giá Trực Quan (C)",
               "Khảo Nghiệm Trên 8 Pha Quỹ Đạo Thực Tế Kính Meta Quest 3S",
               "Khôi phục sắc nét giao diện ảo VR UI, trần kính Penrose Atrium và lối đi sân gạch đỏ ngoài trời")

    img1_p = os.path.join(ROOT_DIR, 'outputs', 'visual_comparisons', 'comparison_quest3s_harvested_phase1_indoor_lab_vrui_anchor.png')
    img2_p = os.path.join(ROOT_DIR, 'outputs', 'visual_comparisons', 'comparison_quest3s_harvested_phase3_penrose_hall_atrium.png')
    img3_p = os.path.join(ROOT_DIR, 'outputs', 'visual_comparisons', 'comparison_quest3s_harvested_phase5_outdoor_campus_brickwalk.png')

    # Display 3 Quest 3S panels side-by-side
    panels = [
        (img1_p, "Pha 1: Bàn Lab & Neo VR UI Ảo (+3.83 dB Crop)"),
        (img2_p, "Pha 3: Sảnh Lớn Penrose Atrium (+4.06 dB Crop)"),
        (img3_p, "Pha 5: Lối Đi Gạch Đỏ Ngoài Trời (+1.44 dB Crop)")
    ]

    for i, (p, title) in enumerate(panels):
        if os.path.exists(p):
            im = Image.open(p)
            ax_sub = fig.add_axes([0.05 + i * 0.31, 0.09, 0.29, 0.71])
            ax_sub.imshow(im)
            ax_sub.axis('off')
            ax_sub.set_title(title, fontsize=8.8, color=GREEN if i==0 else CYAN, fontweight='bold')

    add_footer(ax)
    out_path = os.path.join(OUTPUT_DIR, 'slide_07_visual_quest3s_trajectory.png')
    plt.savefig(out_path, facecolor=BG_DARK, edgecolor='none')
    plt.close()
    print(f"[+] Generated: {out_path}")

# ==============================================================================
# SLIDE 8: TỔNG KẾT & ĐỊNH HƯỚNG TRIỂN KHAI KÍNH VR
# ==============================================================================
def make_slide_8():
    fig, ax = init_slide_fig()
    add_header(ax, 8, "Kết Luận & Triển Khai",
               "Tổng Kết Đóng Góp Khoa Học & Lộ Trình Hiện Thực Hóa Trên Kính XR",
               "Từ giải mã dữ liệu quang học thực tế đến tối ưu hóa kiến trúc nhị phân phần cứng NPU")

    pillars = [
        ("1. Thu Thập Dữ Liệu Thực Tế",
         "Giải mã thành công cơ chế tách 2 luồng mắt SBS từ camera Passthrough của kính Meta Quest 3S (60 FPS, B=64mm).\nChứng minh Ground Truth chính là ảnh gốc từ kính, tạo cặp huấn luyện hoàn toàn chuẩn mực khoa học quốc tế.",
         CYAN),
        ("2. Đột Phá Kiến Trúc Scale-Liberated",
         "Phê phán nút thắt 2D GridSample DRAM (7,500ms) của StereoINR gốc.\nThay thế bằng 1D Epipolar Parallax Attention (EPA-1D), giảm tham số 10x (43.8M -> 4.47M), tăng tốc 57.4x (130ms), tiết kiệm 4.3x RAM.",
         GREEN),
        ("3. Kiểm Chứng Phần Cứng NPU Qualcomm",
         "Vượt qua giới hạn vật lý của NPU Qualcomm Hexagon: StereoINR gốc bị crash hoàn toàn, trong khi mô hình của chúng ta biên dịch thành công nhị phân QNN DLC (1.29 MB, 513.89 ms).",
         AMBER)
    ]

    for i, (title, text, col) in enumerate(pillars):
        y = 0.58 - i * 0.22
        box = patches.FancyBboxPatch((0.05, y), 0.90, 0.19,
                                     boxstyle="round,pad=0.02,rounding_size=0.015",
                                     facecolor=PANEL_BG, edgecolor=col, lw=1.5)
        ax.add_patch(box)
        ax.text(0.08, y + 0.14, title, fontsize=13, fontweight='bold', color=col)
        ax.text(0.08, y + 0.09, text, fontsize=9.8, color=TEXT_WHITE, linespacing=1.35)

    add_footer(ax)
    out_path = os.path.join(OUTPUT_DIR, 'slide_08_conclusion.png')
    plt.savefig(out_path, facecolor=BG_DARK, edgecolor='none')
    plt.close()
    print(f"[+] Generated: {out_path}")

# ==============================================================================
# COMPILE ALL SLIDES INTO SINGLE PDF
# ==============================================================================
def compile_pdf():
    pdf_path = os.path.join(OUTPUT_DIR, 'presentation_stereoinr_quest3s_2026.pdf')
    slide_files = [
        os.path.join(OUTPUT_DIR, f"slide_{i:02d}_{name}.png")
        for i, name in [
            (1, "title"),
            (2, "part1_dataset_gt"),
            (3, "part4_architecture_profiling_dual_pipeline"),
            (4, "part3_training_empirical_metrics"),
            (5, "visual_middlebury"),
            (6, "visual_kitti_flickr"),
            (7, "visual_quest3s_trajectory"),
            (8, "conclusion")
        ]
    ]

    images = []
    for f in slide_files:
        if os.path.exists(f):
            images.append(Image.open(f).convert('RGB'))

    if images:
        images[0].save(pdf_path, save_all=True, append_images=images[1:], resolution=120)
        print(f"[+] Successfully compiled master presentation PDF: {pdf_path}")

def main():
    print("[*] Generating 8 Professional Slides...")
    make_slide_1()
    make_slide_2()
    make_slide_3()
    make_slide_4()
    make_slide_5()
    make_slide_6()
    make_slide_7()
    make_slide_8()
    compile_pdf()
    print("[SUCCESS] All slides generated cleanly in 16:9 HD ready for presentation!")

if __name__ == '__main__':
    main()
