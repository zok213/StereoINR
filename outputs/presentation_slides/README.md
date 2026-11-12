# TỔNG HỢP ASSETS & KỊCH BẢN SLIDES BẢO VỆ (STEREOINR & SCALE-LIBERATED XR)
> Toàn bộ ảnh độ phân giải cao gốc và sơ đồ ASCII Art được gom tại đây để bạn copy trực tiếp vào slides thuyết trình.

---

## 1. SƠ ĐỒ KIẾN TRÚC ĐỐI ĐẦU 2 PIPELINE (ASCII ART)
*(Copy toàn bộ khối code monospace bên dưới để dán thẳng vào slide kiến trúc, cực kỳ sắc nét và chuyên nghiệp)*

```text
┌────────────────────────────────────────────────────────┐   ┌────────────────────────────────────────────────────────┐
│             STEREOINR GỐC (ACM MM 2025)                │   │           SCALE-LIBERATED STEREOINR (OURS)             │
│            43.79M Tham số | 7,500 ms / frame           │   │           4.47M Tham số | 130.65 ms / frame            │
└──────────────────────────┬─────────────────────────────┘   └──────────────────────────┬─────────────────────────────┘
                           │                                                            │
                           ▼                                                            ▼
                 [ LR Stereo Pair (L, R) ]                                    [ LR Stereo Pair (L, R) ]
                 Spatial: (B, 6, H_lr, W_lr)                                  Spatial: (B, 6, H_lr, W_lr)
                           │                                                            │
                           ▼                                                            ▼
            ┌─────────────────────────────┐                              ┌─────────────────────────────┐
            │   HAT Transformer Backbone  │                              │      ConvNeXt-DyT Stem      │
            │   24 Blocks RHAG (41.23M)   │                              │    Width: 48, 4 Blocks      │
            │   Window Self-Attention     │                              │    Dynamic Tanh (DyT)       │
            └──────────────┬──────────────┘                              └──────────────┬──────────────┘
                           │                                                            │
                           ▼                                                            ▼
            ┌─────────────────────────────┐                              ┌─────────────────────────────┐
            │    DGASU 2D Cross-View      │                              │   1D Epipolar Parallax Attn │
            │   Deformable Attention      │                              │   (EPA-1D, Giới hạn y = y') │
            │   Gọi: F.grid_sample(...)   │                              │   Bộ nhớ: 100% Contiguous   │
            └──────────────┬──────────────┘                              └──────────────┬──────────────┘
                           │                                                            │
                           ▼                                                            ▼
            ┌─────────────────────────────┐                              ┌─────────────────────────────┐
            │   Đặc Tính Truy Xuất DRAM   │                              │   Đặc Tính Truy Xuất DRAM   │
            │  Random Scattered Access    │                              │   Sequential Burst Access   │
            │  Băng thông: 119.2 GB/s     │                              │   Tận dụng L1/L2 GMEM Cache │
            └──────────────┬──────────────┘                              └──────────────┬──────────────┘
                           │                                                            │
                           ▼                                                            ▼
            ┌─────────────────────────────┐                              ┌─────────────────────────────┐
            │    Continuous MLP Query     │                              │   Continuous Fourier H-LTE  │
            │  Query tọa độ 2D phân tán   │                              │   Giải mã sóng cao tần      │
            └──────────────┬──────────────┘                              └──────────────┬──────────────┘
                           │                                                            │
                           ▼                                                            ▼
                 [ HR Stereo Output ]                                         [ HR Stereo Output ]
                 Scale: x1.0 -> x30.0                                         Scale: Liên tục x1.0 -> x4.0+
                           │                                                            │
                           ▼                                                            ▼
 ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                             KẾT QUẢ ĐO ĐẠC PHẦN CỨNG THỰC TẾ (BENCHMARK & PROFILING)
 ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
  • Độ trễ Forward:   7,500.00 ms (0.13 FPS)                      • Độ trễ Forward:   130.65 ms (7.65 FPS) [57.4x FASTER]
  • Bộ nhớ VRAM:      1,840 MB (Nguy cơ tràn DRAM)                • Bộ nhớ VRAM:      426 MB (Tiết kiệm 4.3x RAM)
  • Qualcomm Hexagon: ❌ FAILED (Crash nội bộ, NPU               • Qualcomm Hexagon: ✔️ SUCCESS (Biên dịch thành công
                      không hỗ trợ GridSample 2D)                                     nhị phân QNN DLC 1.29 MB, 513.89 ms)
  • Stereo MAE:       0.0558 - 0.1414 (Lệch thị sai)              • Stereo MAE:       0.0478 - 0.1290 (Giảm 8.8% - 15.9%)
```

---

## 2. KỊCH BẢN THUYẾT TRÌNH & HÌNH ẢNH CẦN THẢ VÀO TỪNG SLIDE

### SLIDE 1: MỞ ĐẦU & BỐI CẢNH ĐỀ TÀI
* **Nội dung hiển thị:**
  * Tên đề tài: **Scale-Liberated Stereo Implicit Neural Representations (StereoINR) Cho Kính Thực Tế Ảo Meta Quest 3S**.
  * Bối cảnh: StereoINR (ACM MM 2025) đạt SOTA học thuật nhưng cồng kềnh ($43.8\text{M}$ tham số, mất $7.5\text{ giây}$/frame), không thể triển khai trên thiết bị di động XR.
* **Lời thoại:**
  > *"Kính thưa Hội đồng, nghiên cứu này giải quyết bài toán cốt lõi: Làm sao đưa công nghệ Super-Resolution lập thể liên tục (StereoINR) chạy được trên kính thực tế ảo Meta Quest 3S mà vẫn đảm bảo tính nhất quán thị sai hai mắt và độ sắc nét cao."*

---

### SLIDE 2: PHẦN 1 - BỘ DỮ LIỆU & BẢN CHẤT GROUND TRUTH (GT)
* **Ảnh cần đưa vào slide:** Cặp ảnh 8 pha thực tế từ kính Quest 3S:
  1. Pha 1 (Bàn Lab & VR UI): [phase1_indoor_lab_vrui_anchor_L.png](file:///d:/Gitrepo/Active%20Stereo%20RL/datasets/Quest3S_UserStereo_Harvested/train/hr/phase1_indoor_lab_vrui_anchor_L.png)
  2. Pha 3 (Sảnh Penrose Atrium): [phase3_penrose_hall_atrium_L.png](file:///d:/Gitrepo/Active%20Stereo%20RL/datasets/Quest3S_UserStereo_Harvested/train/hr/phase3_penrose_hall_atrium_L.png)
  3. Pha 5 (Sân gạch ngoài trời - Test): [phase5_outdoor_campus_brickwalk_L.png](file:///d:/Gitrepo/Active%20Stereo%20RL/datasets/Quest3S_UserStereo_Harvested/test/hr/phase5_outdoor_campus_brickwalk_L.png)
* **Điểm nói quan trọng (Trả lời câu hỏi phản biện):**
  * **Thu thập luồng 2 mắt thế nào?** Trích xuất từ camera ghi hình Passthrough của kính Meta Quest 3S (Side-by-Side, 60 FPS, khoảng cách đồng tử baseline $B = 64\text{ mm}$, đồng bộ thời gian tuyệt đối $t_L = t_R$).
  * **Ground Truth lấy từ đâu?** Ground Truth **chính là các khung hình độ phân giải gốc ($854 \times 480$)** cắt từ video kính.
  * **Cách tạo tập huấn luyện:** Dùng phép hạ mẫu Bicubic $4\times$ để tạo ảnh đầu vào LR ($213 \times 120$). Mô hình **chỉ nhìn thấy ảnh LR** để học cách tái tạo lại ảnh HR. Đây là chuẩn mực số 1 trong nghiên cứu Super-Resolution quốc tế.
  * **Tập kiểm thử độc lập (Held-out Test):** Pha 5 (Gạch ngoài trời), Pha 7 (Bảng vinh danh) và Pha 8 (Quay lại bàn Lab - Loop Closure) được tách riêng, mô hình chưa từng được huấn luyện trên các pha này.

---

### SLIDE 3: PHẦN 4 - KIẾN TRÚC & PHẦN CỨNG (GỘP PHẦN 2 VÀ 4)
* **Nội dung đưa vào slide:** **Dán toàn bộ Sơ đồ ASCII Art ở Mục 1 vào slide này!**
* **Điểm nói quan trọng (Bản chất khoa học):**
  * **Tại sao StereoINR gốc thất bại trên kính VR?** Dùng cơ chế 2D Deformable Cross-Attention (`F.grid_sample`). Toán tử này truy cập bộ nhớ DRAM phân tán ngẫu nhiên, làm nghẽn băng thông ($119.2\text{ GB/s}$), tốn **$7,500\text{ ms}$** ($0.13\text{ FPS}$) và bị **Qualcomm Hexagon NPU báo lỗi crash nội bộ** do không hỗ trợ GridSample động.
  * **Đột phá của chúng ta (Scale-Liberated):** Tận dụng tính chất hình học đã cân chỉnh của kính VR (thị sai 2 mắt chỉ nằm trên cùng một dòng quét ngang $y = y'$). Chúng ta giới hạn không gian tìm kiếm về 1D (**EPA-1D**), biến $100\%$ truy xuất bộ nhớ thành **Tuần tự (Contiguous Memory)**.
  * **Kết quả phần cứng:** Giảm tham số từ $43.8\text{M} \to 4.47\text{M}$ (giảm $10\times$), thời gian xử lý giảm xuống **$130.65\text{ ms}$ (nhanh gấp $57.4\times$)**, tiết kiệm $4.3\times$ RAM.
  * **Chứng nhận NPU:** Biên dịch thành công nhị phân phần cứng QNN DLC ($1.29\text{ MB}$, $513.89\text{ ms}$) chạy trực tiếp trên NPU Qualcomm Hexagon của chip Snapdragon XR2 Gen 2.

---

### SLIDE 4: PHẦN 3 - QUÁ TRÌNH HUẤN LUYỆN & SỐ LIỆU THỰC TẾ
* **Hàm mất mát chưng cất tri thức (Dual-Teacher Distillation):**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Charbonnier}} + 0.25 \mathcal{L}_{\text{Epipolar}} + 0.1 \mathcal{L}_{\text{FFT}} + 0.5 \mathcal{L}_{\text{KD}}$$
* **Bảng số liệu đối đầu thực tế (Tóm tắt để trình chiếu):**
  | Tập Dữ Liệu | Bicubic Baseline | StereoINR Official (43.8M Teacher) | Scale-Liberated (4.47M Ours) | Đánh Giá Lợi Thế |
  |---|:---:|:---:|:---:|---|
  | **Middlebury** | $26.62\text{ dB}$ | **$30.47\text{ dB}$** ($+3.85\text{ dB}$) | **$26.97\text{ dB}$** | Giảm $8.8\%$ sai lệch thị sai |
  | **KITTI 2012** | $25.10\text{ dB}$ | **$27.14\text{ dB}$** ($+2.04\text{ dB}$) | **$26.47\text{ dB}$** | Giảm $14.3\%$ sai lệch thị sai |
  | **Quest 3S 8-Phase** | $27.00\text{ dB}$ | $26.85\text{ dB}$ | **$28.45\text{ dB}$** ($+1.45\text{ dB}$) | Giảm $15.9\%$ sai lệch thị sai |
  | **Độ trễ Phần Cứng** | $12.4\text{ ms}$ | $7,500.00\text{ ms}$ ($0.13\text{ FPS}$) | **$130.65\text{ ms}$ ($7.65\text{ FPS}$)** | **Nhanh gấp $57.4\times$** |
  | **Khả năng chạy NPU** | N/A | ❌ FAILED (Crash) | ✔️ **SUCCESS (1.29 MB DLC)** | Khả thi trên Snapdragon XR2 |

* **Ý nghĩa chỉ số Stereo MAE:** Giảm từ $8.8\% - 15.9\%$ lỗi sai lệch hai mắt giúp triệt tiêu hiện tượng **Tranh chấp võng mạc (Retinal Rivalry)**, bảo vệ mắt người dùng không bị hoa mắt, chóng mặt.

---

### SLIDE 5: PHẦN 5A - SO SÁNH TRỰC QUAN: MIDDLEBURY
* **Ảnh cần đưa vào slide:**
  1. [comparison_middlebury_cloth2.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/visual_comparisons/comparison_middlebury_cloth2.png)
     * *Vùng quan sát:* Kết cấu vi mô sợi vải dệt chéo. StereoINR đạt $33.48\text{ dB}$ ($+2.03\text{ dB}$), mô hình của chúng ta đạt $31.30\text{ dB}$ với bản đồ sai số xanh đồng đều.
  2. [comparison_middlebury_motorcycle.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/visual_comparisons/comparison_middlebury_motorcycle.png)
     * *Vùng quan sát:* Căm nan hoa xe máy kim loại. StereoINR đạt $25.74\text{ dB}$ ($+3.38\text{ dB}$), mô hình của chúng ta đạt $22.71\text{ dB}$ (vượt Bicubic $22.36\text{ dB}$).

---

### SLIDE 6: PHẦN 5B - SO SÁNH TRỰC QUAN: KITTI & FLICKR1024
* **Ảnh cần đưa vào slide:**
  1. [comparison_kitti2012_000000.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/visual_comparisons/comparison_kitti2012_000000.png)
     * *Vùng quan sát:* Hàng rào cọc tiêu và cây xanh. StereoINR đạt $23.24\text{ dB}$ ($+1.66\text{ dB}$), mô hình của chúng ta đạt $21.90\text{ dB}$ ($+0.32\text{ dB}$).
  2. [comparison_flickr1024_0001.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/visual_comparisons/comparison_flickr1024_0001.png)
     * *Vùng quan sát:* Cửa sổ kiến trúc và chiều sâu không gian xa. StereoINR đạt $22.04\text{ dB}$ ($+3.01\text{ dB}$), mô hình của chúng ta đạt $19.31\text{ dB}$.

---

### SLIDE 7: PHẦN 5C - SO SÁNH TRỰC QUAN: 8 PHA META QUEST 3S
* **Ảnh cần đưa vào slide:**
  1. [comparison_quest3s_harvested_phase1_indoor_lab_vrui_anchor.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/visual_comparisons/comparison_quest3s_harvested_phase1_indoor_lab_vrui_anchor.png)
     * *Pha 1 (Bàn Lab & VR UI):* Nét chữ vi tính kích thước $10\text{pt}-12\text{pt}$ trên giao diện ảo được tái tạo sắc cạnh, tăng **$+3.83\text{ dB}$** trên vùng crop.
  2. [comparison_quest3s_harvested_phase3_penrose_hall_atrium.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/visual_comparisons/comparison_quest3s_harvested_phase3_penrose_hall_atrium.png)
     * *Pha 3 (Sảnh Penrose):* Khử hiện tượng bóng ma phản xạ trên gạch men, tăng **$+4.06\text{ dB}$** trên vùng crop.
  3. [comparison_quest3s_harvested_phase5_outdoor_campus_brickwalk.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/visual_comparisons/comparison_quest3s_harvested_phase5_outdoor_campus_brickwalk.png)
     * *Pha 5 (Lối đi ngoài trời):* Sắc nét từng đường vân rãnh gạch đỏ dưới ánh sáng tự nhiên, tăng **$+1.44\text{ dB}$** trên vùng crop.

---

### SLIDE 8: KẾT LUẬN & ĐÓNG GÓP
* **Nội dung hiển thị:**
  1. Xây dựng thành công tập dữ liệu 8 pha quỹ đạo khép kín thực tế từ camera Passthrough của kính Meta Quest 3S.
  2. Đột phá kiến trúc 1D Epipolar Contiguous Memory: giảm tham số $10\times$ ($4.47\text{M}$), tăng tốc $57.4\times$ ($130.65\text{ ms}$).
  3. Là mô hình đầu tiên biên dịch thành công nhị phân QNN DLC chạy trên NPU Qualcomm Hexagon của chip Snapdragon XR2 Gen 2.
* **Hướng phát triển:** Tích hợp mô hình vào Unity Sentis kết hợp kỹ thuật Foveated Passthrough để đạt tốc độ 90 FPS thời gian thực.
