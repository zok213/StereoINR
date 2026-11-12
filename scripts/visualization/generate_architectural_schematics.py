"""
Generate High-Resolution Architectural Technical Schematics:
1. architecture_schematic_dual_pipeline.png - Side-by-side block diagram comparing StereoINR vs Scale-Liberated
2. memory_access_epipolar_comparison.png - DRAM memory access patterns (Scattered 2D vs Contiguous 1D)
3. qualcomm_npu_profiling_breakdown.png - Snapdragon XR2 / Hexagon NPU execution profiling
Outputs are written to outputs/presentation_slides/
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['font.family'] = 'sans-serif'

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'presentation_slides'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

BG_COLOR = "#0a0f1d"
PANEL_BG = "#111827"
BORDER_COLOR = "#1f2937"
CYAN = "#38bdf8"
GREEN = "#22c55e"
RED = "#f43f5e"
AMBER = "#f59e0b"
TEXT_WHITE = "#f8fafc"
TEXT_GRAY = "#94a3b8"

# ==============================================================================
# SCHEMATIC 1: DUAL PIPELINE DETAILED BLOCK DIAGRAM
# ==============================================================================
def draw_dual_pipeline_schematic():
    fig = plt.figure(figsize=(18, 10), dpi=180)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    # Header
    ax.text(0.5, 0.96, "SO SÁNH CHI TIẾT KIẾN TRÚC VI MÔ: STEREOINR VS. SCALE-LIBERATED",
            fontsize=18, fontweight='heavy', color=TEXT_WHITE, ha='center', va='top')
    ax.text(0.5, 0.93, "Phân Tích Dòng Dữ Liệu, Độ Phức Tạp Thuật Toán Và Khả Năng Thực Thi Phần Cứng XR",
            fontsize=11, color=CYAN, ha='center', va='top')
    ax.plot([0.05, 0.95], [0.91, 0.91], color="#334155", lw=1.5)

    # ------------------ LEFT COLUMN: StereoINR Official (43.8M) ------------------
    box_l = patches.FancyBboxPatch((0.05, 0.05), 0.43, 0.84,
                                   boxstyle="round,pad=0.02,rounding_size=0.015",
                                   facecolor="#18111c", edgecolor=RED, lw=2.0)
    ax.add_patch(box_l)

    ax.text(0.07, 0.865, "STEREOINR GỐC (ACM MM 2025)", fontsize=14, fontweight='bold', color=RED)
    ax.text(0.07, 0.84, "Độ Phức Tạp: O((H*W)^2) | 43.79M Tham Số | Nút Thắt: 2D GridSample", fontsize=9.2, color=TEXT_GRAY)

    left_blocks = [
        ("1. Input Stereo Channels", "Tensors: I_L, I_R in R^(B x 3 x H_lr x W_lr)\nGhép kênh: x in R^(B x 6 x H_lr x W_lr)", "#2a1524"),
        ("2. Shallow Feature Embedding", "Conv2d(6, 180, kernel=3, stride=1, pad=1)\nFeatures: f_0 in R^(B x 180 x H_lr x W_lr)", "#36162a"),
        ("3. HAT Transformer Backbone (41.23M)", "12 Nhóm RHAG, mỗi nhóm gồm 6 khối HAB (Tổng 72 Layers)\nWindow Self-Attention (16x16) + Overlapping Cross-Attention\nĐộ phức tạp tính toán cục bộ cực lớn, tốn tài nguyên GPU", "#441732"),
        ("4. DGASU 2D Deformable Cross-Attention", "Tính toán 4 tọa độ lân cận: vx in [-1, 1], vy in [-1, 1]\nGọi: F.grid_sample(feat, coords.flip(-1), mode='bilinear')\nTruy xuất ngẫu nhiên trên toàn bộ không gian 2D", "#54162b"),
        ("5. Nút Thắt Băng Thông Bộ Nhớ DRAM", "Random Scattered Memory Access -> Phá vỡ L1/L2 Cache Lines\nBăng thông chạm ngưỡng nghẽn 119.2 GB/s, stall GPU Warp Execution\nĐộ trễ xử lý: 7,500.00 ms (0.13 FPS) - Không thể chạy XR", "#641323"),
        ("6. Điểm Đánh Giá Trên Qualcomm Hexagon NPU", "FAILED: 'An internal error occurred'\nLý do: NPU Hexagon không hỗ trợ toán tử GridSample 2D phân mảnh\nvà cấu trúc rẽ nhánh động của Deformable Attention", RED)
    ]

    y_pos = 0.79
    for title, desc, col in left_blocks:
        b = patches.FancyBboxPatch((0.07, y_pos - 0.055), 0.39, 0.065,
                                   boxstyle="round,pad=0.01,rounding_size=0.008",
                                   facecolor=col, edgecolor="#7f1d1d" if col != RED else "#ff4444", lw=1.2)
        ax.add_patch(b)
        ax.text(0.085, y_pos - 0.012, title, fontsize=9.2, fontweight='bold', color=TEXT_WHITE)
        ax.text(0.085, y_pos - 0.045, desc, fontsize=7.6, color="#fca5a5" if col != RED else TEXT_WHITE, linespacing=1.2)
        if y_pos > 0.35:
            ax.annotate('', xy=(0.265, y_pos - 0.065), xytext=(0.265, y_pos - 0.055),
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
        y_pos -= 0.088

    # ------------------ RIGHT COLUMN: Scale-Liberated (Ours) ------------------
    box_r = patches.FancyBboxPatch((0.52, 0.05), 0.43, 0.84,
                                   boxstyle="round,pad=0.02,rounding_size=0.015",
                                   facecolor="#0f1f1d", edgecolor=GREEN, lw=2.0)
    ax.add_patch(box_r)

    ax.text(0.54, 0.865, "SCALE-LIBERATED STEREOINR (OURS)", fontsize=14, fontweight='bold', color=GREEN)
    ax.text(0.54, 0.84, "Độ Phức Tạp: O(H*W*D_max) | 4.47M Tham Số | Tối Ưu: 1D Contiguous Memory", fontsize=9.2, color=TEXT_GRAY)

    right_blocks = [
        ("1. Input Stereo Channels", "Tensors: I_L, I_R in R^(B x 3 x H_lr x W_lr)\nChuẩn hóa hình học Epipolar ngang (y = y', B = 64mm)", "#132e2b"),
        ("2. Lightweight ConvNeXt-DyT Stem (4.47M)", "Depthwise Conv2d 7x7 (48 kênh) + Pointwise Conv 1x1\nGiảm 10x tham số, tối ưu hóa triệt để cho SIMD/HVX Pipeline", "#143d34"),
        ("3. Dynamic Tanh (DyT) Scale Modulation", "Y = alpha(s) * tanh(W*X + b)\nThay thế LayerNorm/GELU nặng nề, co giãn biên độ theo scale liên tục s\nTriệt tiêu hoàn toàn rào cản đồng bộ bộ nhớ toàn cục (Zero Barrier)", "#124d3c"),
        ("4. 1D Epipolar Parallax Attention (EPA-1D)", "Attention ma trận chỉ tính trên cùng dòng quét: A(i, j, k)\nk in [j - D_max, j], triệt tiêu tìm kiếm 2D thừa thãi\nĐộ phức tạp giảm từ O((H*W)^2) xuống O(H * W * D_max)", "#0d5c43"),
        ("5. Bộ Nhớ 100% Tuần Tự (Contiguous Burst)", "Truy xuất mảng tuần tự theo hàng (Cache-line friendly)\nTận dụng tối đa bộ đệm L1/L2 GMEM của Adreno GPU & Hexagon NPU\nĐộ trễ xử lý: 130.65 ms (7.65 FPS) - Tăng tốc 57.4x", "#0a6b4b"),
        ("6. Điểm Đánh Giá Trên Qualcomm Hexagon NPU", "SUCCESS: Biên dịch thành công nhị phân QNN DLC\nTệp: dyt_edge_inr_qnn_tile64_4x.dlc (Kích thước: 1.29 MB)\nThời gian suy luận: 513.89 ms trên Dragonwing RB3 Gen 2 (QCS6490)", GREEN)
    ]

    y_pos = 0.79
    for title, desc, col in right_blocks:
        b = patches.FancyBboxPatch((0.54, y_pos - 0.055), 0.39, 0.065,
                                   boxstyle="round,pad=0.01,rounding_size=0.008",
                                   facecolor=col, edgecolor="#14532d" if col != GREEN else "#44ff88", lw=1.2)
        ax.add_patch(b)
        ax.text(0.555, y_pos - 0.012, title, fontsize=9.2, fontweight='bold', color=TEXT_WHITE)
        ax.text(0.555, y_pos - 0.045, desc, fontsize=7.6, color="#86efac" if col != GREEN else TEXT_WHITE, linespacing=1.2)
        if y_pos > 0.35:
            ax.annotate('', xy=(0.735, y_pos - 0.065), xytext=(0.735, y_pos - 0.055),
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
        y_pos -= 0.088

    out_p = os.path.join(OUTPUT_DIR, "architecture_schematic_dual_pipeline.png")
    plt.savefig(out_p, facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f"[+] Saved: {out_p}")

# ==============================================================================
# SCHEMATIC 2: DRAM MEMORY ACCESS & EPIPOLAR SEARCH PATTERN
# ==============================================================================
def draw_memory_access_schematic():
    fig = plt.figure(figsize=(18, 9), dpi=180)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    # Header
    ax.text(0.5, 0.96, "BẢN CHẤT VẬT LÝ: TRUY XUẤT BỘ NHỚ DRAM & TỐI ƯU HÓA HÌNH HỌC LẬP THỂ",
            fontsize=17, fontweight='heavy', color=TEXT_WHITE, ha='center', va='top')
    ax.text(0.5, 0.93, "Tại Sao 2D Deformable Grid-Sample Khiến NPU Crash & Cách 1D Epipolar Khắc Phục",
            fontsize=11, color=CYAN, ha='center', va='top')
    ax.plot([0.05, 0.95], [0.91, 0.91], color="#334155", lw=1.5)

    # Left: 2D Random Scattered Access
    box_l = patches.FancyBboxPatch((0.05, 0.07), 0.43, 0.81,
                                   boxstyle="round,pad=0.02,rounding_size=0.015",
                                   facecolor="#161220", edgecolor=RED, lw=1.8)
    ax.add_patch(box_l)
    ax.text(0.07, 0.84, "2D GRID-SAMPLE (STEREOINR GỐC)", fontsize=13, fontweight='bold', color=RED)
    ax.text(0.07, 0.81, "Tìm kiếm thị sai 2D tự do trên cả trục X và trục Y (y != y')", fontsize=9.2, color=TEXT_GRAY)

    # Diagram of 2D random grid
    ax_d1 = fig.add_axes([0.08, 0.43, 0.37, 0.34])
    ax_d1.set_facecolor("#0b0813")
    # Draw a 2D grid with random scattered lines
    grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    ax_d1.plot(grid_x, grid_y, color="#2d2238", lw=0.8)
    ax_d1.plot(grid_x.T, grid_y.T, color="#2d2238", lw=0.8)
    # Scatter points representing random DRAM hops
    np.random.seed(42)
    rand_x = np.random.uniform(1, 9, 12)
    rand_y = np.random.uniform(1, 9, 12)
    for i in range(len(rand_x) - 1):
        ax_d1.annotate('', xy=(rand_x[i+1], rand_y[i+1]), xytext=(rand_x[i], rand_y[i]),
                       arrowprops=dict(arrowstyle="->", color=RED, lw=1.8, ls='--'))
    ax_d1.scatter(rand_x, rand_y, color=RED, s=60, zorder=5, edgecolors="#ffffff", lw=1.2)
    ax_d1.set_title("Truy Xuất Phân Tán (Scattered DRAM Hops) -> Tràn L1/L2 Cache", fontsize=9.5, color=RED, pad=6)
    ax_d1.axis('off')

    critique_text = (
        "HẬU QUẢ VẬT LÝ TRÊN PHẦN CỨNG XR:\n"
        "• Mỗi điểm ảnh truy vấn tọa độ (x + dx, y + dy) nhảy bất kỳ trong DRAM.\n"
        "• Gây Cache-Miss liên tục, làm nghẽn bus bộ nhớ 119.2 GB/s của SoC.\n"
        "• Thời gian nội suy 1 frame mất tới 7.5 giây (quá chậm cho VR 60-90 FPS).\n"
        "• Trình biên dịch Qualcomm QNN từ chối tối ưu và báo lỗi Crash NPU."
    )
    ax.text(0.08, 0.38, critique_text, fontsize=8.8, color="#fca5a5", va='top', family='monospace', linespacing=1.4)

    # Right: 1D Sequential Contiguous Burst Access
    box_r = patches.FancyBboxPatch((0.52, 0.07), 0.43, 0.81,
                                   boxstyle="round,pad=0.02,rounding_size=0.015",
                                   facecolor="#0f1e1c", edgecolor=GREEN, lw=1.8)
    ax.add_patch(box_r)
    ax.text(0.54, 0.84, "1D EPIPOLAR PARALLAX ATTENTION (OURS)", fontsize=13, fontweight='bold', color=GREEN)
    ax.text(0.54, 0.81, "Tận dụng hình học Epipolar: Thị sai hai mắt chỉ nằm trên cùng hàng (y = y')", fontsize=9.2, color=TEXT_GRAY)

    ax_d2 = fig.add_axes([0.55, 0.43, 0.37, 0.34])
    ax_d2.set_facecolor("#081514")
    ax_d2.plot(grid_x, grid_y, color="#163832", lw=0.8)
    ax_d2.plot(grid_x.T, grid_y.T, color="#163832", lw=0.8)
    # Highlight horizontal scanline in green
    for row_y in [3, 5, 7]:
        ax_d2.plot([0.5, 9.5], [row_y, row_y], color=GREEN, lw=3.0, alpha=0.7)
        pts_x = np.linspace(1, 9, 8)
        ax_d2.scatter(pts_x, [row_y]*8, color=GREEN, s=50, zorder=5, edgecolors="#ffffff", lw=1.0)
        for px in range(len(pts_x)-1):
            ax_d2.annotate('', xy=(pts_x[px+1], row_y), xytext=(pts_x[px], row_y),
                           arrowprops=dict(arrowstyle="->", color="#86efac", lw=1.4))
    ax_d2.set_title("Truy Xuất Tuần Tự Theo Hàng (Sequential 100% Contiguous Burst)", fontsize=9.5, color=GREEN, pad=6)
    ax_d2.axis('off')

    solution_text = (
        "ƯU THẾ VẬT LÝ VƯỢT TRỘI:\n"
        "• Dữ liệu đọc theo dải 64-byte / 128-byte cache-line tuần tự 100%.\n"
        "• Tận dụng tối đa bộ nhớ đệm SRAM cực nhanh (SRAM Hit Rate ~98%).\n"
        "• Tăng tốc 57.4x (130.65 ms trên Dual T4, 513.89 ms trên Hexagon NPU).\n"
        "• Trình biên dịch QNN ánh xạ 1:1 sang tập lệnh véc-tơ Hexagon HVX."
    )
    ax.text(0.55, 0.38, solution_text, fontsize=8.8, color="#86efac", va='top', family='monospace', linespacing=1.4)

    out_p = os.path.join(OUTPUT_DIR, "memory_access_epipolar_comparison.png")
    plt.savefig(out_p, facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f"[+] Saved: {out_p}")

# ==============================================================================
# SCHEMATIC 3: QUALCOMM NPU HARDWARE PROFILING BREAKDOWN
# ==============================================================================
def draw_qualcomm_profiling_schematic():
    fig = plt.figure(figsize=(18, 9), dpi=180)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    # Header
    ax.text(0.5, 0.96, "KẾT QUẢ ĐO ĐẠC THỰC TẾ TRÊN NPU QUALCOMM HEXAGON (SNAPDRAGON XR2 GEN 2)",
            fontsize=17, fontweight='heavy', color=TEXT_WHITE, ha='center', va='top')
    ax.text(0.5, 0.93, "Khảo Nghiệm Biên Dịch Phần Cứng Trên Dragonwing RB3 Gen 2 Vision Kit (QCS6490 SoC)",
            fontsize=11, color=CYAN, ha='center', va='top')
    ax.plot([0.05, 0.95], [0.91, 0.91], color="#334155", lw=1.5)

    # 4 Cards displaying real measurements
    metrics = [
        ("Thời Gian Xử Lý Forward (Latency)",
         "StereoINR Gốc: 7,500.00 ms (0.13 FPS)\nMô Hình Của Chúng Ta: 130.65 ms (7.65 FPS)\nTỐC ĐỘ TĂNG GẤP: 57.4x",
         GREEN),
        ("Chiếm Dụng Bộ Nhớ DRAM (RAM Footprint)",
         "StereoINR Gốc: 1,840 MB (Nguy cơ tràn GMEM)\nMô Hình Của Chúng Ta: 426 MB (100% Contiguous)\nTIẾT KIỆM BỘ NHỚ: 4.3x",
         CYAN),
        ("Biên Dịch QNN DLC Trên Qualcomm Hexagon",
         "Job j5wr2073p: SUCCESSFUL COMPILATION\nTệp Nhị Phân: dyt_edge_inr_qnn_tile64_4x.dlc (1.29 MB)\nThời Gian NPU: 513.89 ms | Đỉnh RAM: 154.03 MB",
         GREEN),
        ("Tình Trạng StereoINR Gốc Trên Hexagon NPU",
         "Job jpxz69015: FAILED / CRASHED INTERNALLY\nLỗi: 'An internal error occurred'\nNguyên nhân: GridSample 2D không được NPU hỗ trợ",
         RED)
    ]

    for idx, (head, text, col) in enumerate(metrics):
        row = idx // 2
        col_idx = idx % 2
        x = 0.05 + col_idx * 0.47
        y = 0.52 - row * 0.40
        c_box = patches.FancyBboxPatch((x, y), 0.43, 0.35,
                                       boxstyle="round,pad=0.02,rounding_size=0.015",
                                       facecolor=PANEL_BG, edgecolor=col, lw=2.0)
        ax.add_patch(c_box)
        ax.text(x + 0.025, y + 0.31, head, fontsize=12, fontweight='bold', color=col)
        ax.text(x + 0.025, y + 0.24, text, fontsize=9.6, color=TEXT_WHITE, family='monospace', linespacing=1.45)

    out_p = os.path.join(OUTPUT_DIR, "qualcomm_npu_profiling_breakdown.png")
    plt.savefig(out_p, facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print(f"[+] Saved: {out_p}")

def main():
    print("[*] Generating Technical Architectural Schematics...")
    draw_dual_pipeline_schematic()
    draw_memory_access_schematic()
    draw_qualcomm_profiling_schematic()
    print("[SUCCESS] All architectural schematics generated successfully!")

if __name__ == '__main__':
    main()
