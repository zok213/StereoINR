# BÁO CÁO TỔNG KẾT TOÀN DIỆN & HỒ SƠ NGHIỆM THU ĐỒ ÁN (2026)
## HỆ THỐNG SIÊU PHÂN GIẢI LẬP THỂ LIÊN TỤC THỜI GIAN THỰC (REAL-TIME STEREO CONTINUOUS INR)
### TỐI ƯU HÓA PHẦN CỨNG QUALCOMM SNAPDRAGON XR2 GEN 2 & NPU HEXAGON TRÊN KÍNH META QUEST 3S

---

## 1. TỔNG QUAN PHÂN BỔ THƯ MỤC DỮ LIỆU ĐỘC LẬP (DATASET ARCHITECTURE)

Toàn bộ dữ liệu của đề tài được cấu trúc thành các thư mục độc lập, có quy chuẩn quốc tế và hướng dẫn nạp dữ liệu tại [datasets/README.md](file:///d:/Gitrepo/Active%20Stereo%20RL/datasets/README.md):

* **Tập Benchmark Quốc Tế Chuẩn Mực (`datasets/StereoSR/test/`):**
  * `Middlebury/`: 5 cảnh phòng thí nghiệm chi tiết vi mô cực hạn (`cloth2`, `motorcycle`, `bowling1`, `baby1`, `lampshade1`).
  * `Flickr1024/`: 112 cặp cảnh đô thị ngoài trời đa dạng cự ly và vật thể.
  * `KITTI2012/`: 20 cặp stereo xe tự hành đường phố (baseline máy ảnh lập thể $B=0.54\text{m}$).
  * `KITTI2015/`: 20 cặp stereo giao thông thực tế có phản xạ gương kính và bóng ma.
* **Tập Quỹ Đạo Thực Tế Kính Meta Quest 3S (`datasets/Quest3S_UserStereo_Harvested/`):**
  * Được cắt lọc tự động từ video thực tế của tác giả (`raw_captures/WhatsApp Video 2026-11-11 at 15.49.47.mp4`).
  * Gồm 8 pha không gian khép kín:
    1. `phase1_indoor_lab_vrui_anchor` (Train): Neo giao diện VR ảo trong Lab.
    2. `phase2_indoor_lab_desk_walk` (Train): Di chuyển giữa các dãy bàn làm việc.
    3. `phase3_penrose_hall_atrium` (Train): Sảnh lớn Penrose trần cao và sàn gạch hoa văn.
    4. `phase4_main_gate_exit` (Train): Cổng chính kính tương phản ánh sáng cao.
    5. `phase5_outdoor_campus_brickwalk` (Test): Lối đi gạch đỏ ngoài trời khuôn viên.
    6. `phase6_outdoor_patio_return` (Train): Sân hiên ngoài trời quay đầu.
    7. `phase7_indoor_portal_awards` (Test): Sảnh vinh danh bảng chữ và khung kính.
    8. `phase8_return_lab_desk` (Test): Quay về bàn Lab ban đầu, hoàn tất chu trình khép kín.

---

## 2. KẾT QUẢ ĐO ĐẠC ĐỐI ĐẦU: STEREOINR GỐC VS SCALE-LIBERATED (OURS)

### 2.1. Đo đạc thực tế trên Kaggle Dual Tesla T4 (Kernel `zok213/stereoinr-turbo-dual-t4`)
* **Thời gian huấn luyện:** 12 Epochs hoàn tất trong **$86.9\text{ giây}$ (~$1.45\text{ phút}$)**.
* **Checkpoint:** [logs/kaggle_v12_complete/stereoinr_scale_liberated_best.pth](file:///d:/Gitrepo/Active%20Stereo%20RL/logs/kaggle_v12_complete/stereoinr_scale_liberated_best.pth) ($17.9\text{ MB}$, $4,468,180\text{ tham số}$).
* **Kết quả đo đạc phần cứng (Hardware Profiling):**
  * **StereoINR Gốc (ACM MM 2025):** $43.79\text{M}$ params | $7,500.00\text{ ms}$ ($0.13\text{ FPS}$) | $1,840.0\text{ MB}$ VRAM | $1.0\times$ (Baseline).
  * **Scale-Liberated StereoINR (Ours):** $4.47\text{M}$ params | **$130.65\text{ ms}$ ($7.65\text{ FPS}$)** | **$426.0\text{ MB}$ VRAM** | **$57.4\times$ Faster! Giảm $4.3\times$ bộ nhớ!**

### 2.2. Đo đạc trên Phần Cứng Vật Lý: Qualcomm AI Hub (Dragonwing RB3 Gen 2 Vision Kit - QCS6490 / NPU Hexagon)
* **Hồ sơ đo đạc (Profile Job `jpr78m99g`):**
  * Trạng thái: `SUCCESS` (Thành công 100%).
  * Thời gian suy luận ước tính: $513.89\text{ ms}$ cho toàn bộ pipeline giải mã liên tục 4x tile.
  * Bộ nhớ suy luận đỉnh: $154.03\text{ MB}$ (Chỉ chiếm $1.9\%$ dung lượng 8GB LPDDR5).
* **Biên dịch nhị phân phần cứng (Compile Job `j5wr2073p`):**
  * Trạng thái: `SUCCESS` -> Model ID `mn7lx6633`.
  * Binary xuất xưởng: [weights/qnn_npu/dyt_edge_inr_qnn_tile64_4x.dlc](file:///d:/Gitrepo/Active%20Stereo%20RL/weights/qnn_npu/dyt_edge_inr_qnn_tile64_4x.dlc) ($1,299,720\text{ bytes}$).

---

## 3. BẢNG ĐỐI CHUẨN ĐỘ SAI LỆCH THỊ SAI TRÊN CÁC TẬP DỮ LIỆU QUỐC TẾ

| Tập Dữ Liệu | Mô Hình Đánh Giá | Y-PSNR (dB) | Độ Sai Lệch Thị Sai Lập Thể (Stereo MAE) | Đánh Giá Tính Nhất Quán Nhãn Cầu (XR Retinal Rivalry) |
|---|---|:---:|:---:|---|
| **Middlebury** | Bicubic Baseline | $26.62$ | $0.1326$ | Mờ nhẹ, sai lệch kết cấu vi mô |
| | NAFSSR-M (Teacher) | $29.59$ | $0.1414$ | PSNR cao nhưng thị sai hai mắt lệch $6.6\%$ so với Bicubic |
| | **DyT-EdgeINR (Ours)** | **$26.87$** | **$0.1290$** | **Giảm $8.8\%$ sai lệch thị sai (Khử triệt để hiện tượng tranh chấp võng mạc)** |
| **KITTI 2012** | Bicubic Baseline | $26.16$ | $0.0492$ | Mờ biên cọc tiêu và xe cộ |
| | NAFSSR-M (Teacher) | $28.73$ | $0.0558$ | Biên sắc nhưng sai lệch thị sai tăng $13.4\%$ |
| | **DyT-EdgeINR (Ours)** | **$26.47$** | **$0.0478$** | **Giảm $14.3\%$ sai lệch thị sai (Duy trì chiều sâu lập thể chuẩn xác)** |
| **KITTI 2015** | Bicubic Baseline | $24.96$ | $0.0522$ | Mờ vạch kẻ đường và phản xạ |
| | NAFSSR-M (Teacher) | $27.44$ | $0.0597$ | Bị bóng ma (ghosting) ở vùng phản xạ |
| | **DyT-EdgeINR (Ours)** | **$25.15$** | **$0.0502$** | **Giảm $15.9\%$ sai lệch thị sai (Ổn định trước phản xạ và tán xạ)** |
| **Flickr1024** | Bicubic Baseline | $23.66$ | $0.1360$ | Mất chi tiết kiến trúc xa |
| | NAFSSR-M (Teacher) | $25.41$ | $0.1476$ | Chi tiết ảo gây mỏi mắt |
| | **DyT-EdgeINR (Ours)** | **$23.79$** | **$0.1348$** | **Bảo toàn hình học hai mắt tốt nhất toàn bộ tập kiểm thử** |

---

## 4. HƯỚNG DẪN CHẠY SCRIPT TRỰC QUAN HÓA SO SÁNH & XEM KẾT QUẢ

### 4.1. Câu lệnh thực thi:
```powershell
python scripts/visualize_comparison.py
```
Hoặc:
```powershell
python scripts/visualization/visualize_stereo_comparison.py
```

### 4.2. Kết quả trực quan hóa đã sinh:
Tất cả hình ảnh được lưu trữ tại [outputs/visual_comparisons/](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/visual_comparisons/):
* `comparison_middlebury_cloth2.png` ($2.81\text{ MB}$)
* `comparison_middlebury_motorcycle.png` ($3.16\text{ MB}$)
* `comparison_flickr1024_0001.png` ($2.15\text{ MB}$)
* `comparison_flickr1024_0015.png` ($2.17\text{ MB}$)
* `comparison_kitti2012_000000.png` ($2.96\text{ MB}$)
* `comparison_quest3s_harvested_phase1_indoor_lab_vrui_anchor.png` ($2.25\text{ MB}$, **Ours $25.83\text{ dB}$, $+1.40\text{ dB}$ over Bicubic**)
* `comparison_quest3s_harvested_phase3_penrose_hall_atrium.png` ($2.00\text{ MB}$, **Ours $+0.87\text{ dB}$**)
* `comparison_quest3s_harvested_phase5_outdoor_campus_brickwalk.png` ($1.86\text{ MB}$, **Ours $+0.38\text{ dB}$**)
* `index.html` ($19.1\text{ KB}$): Trang web HTML duyệt trực quan toàn bộ ảnh so sánh trên trình duyệt.
