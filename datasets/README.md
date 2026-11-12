# DANH MỤC & CẤU TRÚC PHÂN TÁCH DỮ LIỆU ĐỒ ÁN (DATASET CATALOG & ARCHITECTURE)
## HỆ THỐNG SIÊU PHÂN GIẢI LẬP THỂ THỜI GIAN THỰC (REAL-TIME STEREO SUPER-RESOLUTION)
### KÍNH THỰC TẾ ẢO META QUEST 3S (QUALCOMM SNAPDRAGON XR2 GEN 2)

---

## 1. TỔNG QUAN PHÂN BỔ THƯ MỤC DỮ LIỆU (DIRECTORY STRUCTURE)

Toàn bộ dữ liệu được chuẩn hóa và phân chia độc lập thành các thư mục riêng biệt theo đúng chuẩn nghiên cứu thị giác máy tính quốc tế:

```
datasets/
│
├── StereoSR/                               # BỘ DỮ LIỆU ĐỐI CHUẨN QUỐC TẾ (ACADEMIC CANONICAL BENCHMARKS)
│   └── test/
│       ├── Flickr1024/                     # Tập kiểm thử Flickr1024 (112 cảnh stereo ngoài trời/đô thị)
│       │   ├── hr/                         #   Ảnh gốc độ phân giải cao (HR): hr0.png (Trái), hr1.png (Phải)
│       │   ├── lr_x2/                      #   Ảnh độ phân giải thấp nội suy 2x (LR x2)
│       │   └── lr_x4/                      #   Ảnh độ phân giải thấp nội suy 4x (LR x4)
│       ├── Middlebury/                     # Tập kiểm thử Middlebury (5 cảnh phòng lab chi tiết vi mô cực cao)
│       │   ├── hr/                         #   Cảnh: baby1, bowling1, cloth2, lampshade1, motorcycle
│       │   ├── lr_x2/                      #   LR x2
│       │   └── lr_x4/                      #   LR x4
│       ├── KITTI2012/                      # Tập kiểm thử KITTI 2012 (20 cảnh stereo xe tự hành đường phố)
│       │   ├── hr/                         #   HR: 000000_10 ... 000019_10
│       │   ├── lr_x2/                      #   LR x2
│       │   └── lr_x4/                      #   LR x4
│       └── KITTI2015/                      # Tập kiểm thử KITTI 2015 (20 cảnh stereo giao thông thực tế)
│           ├── hr/                         #   HR: 000000_10 ... 000019_10
│           ├── lr_x2/                      #   LR x2
│           └── lr_x4/                      #   LR x4
│
├── Quest3S_UserStereo_Harvested/           # BỘ DỮ LIỆU QUỸ ĐẠO THỰC TẾ META QUEST 3S (8-PHASE TRAJECTORY)
│   ├── quest3s_trajectory_manifest.json    # Bảng chỉ mục metadata chi tiết của 8 pha quỹ đạo
│   ├── train/                              # Tập huấn luyện nội bộ kính (5 pha không gian)
│   │   ├── hr/                             #   Ảnh gốc 854x480 trích từ video Quest 3S (Trái/Phải)
│   │   ├── lr_x2/                          #   Ảnh thu nhỏ 2x (427x240)
│   │   ├── lr_x4/                          #   Ảnh thu nhỏ 4x (213x120)
│   │   ├── disp16/                         #   Bản đồ thị sai 16-bit dưới điểm ảnh (Sub-pixel Disparity)
│   │   └── occ/                            #   Mặt nạ che khuất hai mắt (Occlusion Mask)
│   └── test/                               # Tập kiểm thử tổng quát hóa (3 pha không gian biệt lập)
│       ├── hr/                             #   HR 854x480
│       ├── lr_x2/                          #   LR x2
│       ├── lr_x4/                          #   LR x4
│       ├── disp16/                         #   Disparity map 16-bit
│       └── occ/                            #   Occlusion Mask
│
├── middlebury_2014/                        # BỘ DỮ LIỆU ĐỘ PHÂN GIẢI CỰC CAO (2000x1500) CÓ GROUND-TRUTH DEPTH
│   └── training/                           # Phục vụ huấn luyện nhận thức chiều sâu học có giám sát
│
└── raw_captures/                           # THƯ MỤC LƯU TRỮ VIDEO GỐC TỪ THIẾT BỊ
    └── WhatsApp Video 2026-11-11 at 15.49.47.mp4 (147.96 giây, 8,876 frames, 60 FPS)
```

---

## 2. BẢNG MÔ TẢ ĐẶC TÍNH CHI TIẾT TỪNG TẬP DỮ LIỆU

| Tên Bộ Dữ Liệu | Nguồn Dữ Liệu | Số Lượng Mẫu | Độ Phân Giải HR | Đặc Trưng Quang Học & Hình Học | Mục Đích Sử Dụng |
|---|---|:---:|:---:|---|---|
| **Flickr1024 (Test)** | Wang et al. (CVPR 2019) | $112\text{ cặp}$ | $1024 \times 512$ | Đa dạng cảnh tự nhiên, kiến trúc đô thị, nhiều vật thể xa gần | So sánh trực diện với StereoINR (ACM MM 2025) và các baseline |
| **Middlebury (Test)** | Scharstein et al. | $5\text{ cặp}$ | $680 \times 440$ | Ánh sáng kiểm soát trong lab, kết cấu vi mô (vải, kim loại, bóng đổ) | Đánh giá khả năng khôi phục vân bề mặt cực hạn |
| **KITTI 2012 (Test)** | Geiger et al. | $20\text{ cặp}$ | $1242 \times 375$ | Baseline camera lập thể rộng ($B=0.54\text{m}$), chuyển động xe ô tô | Đánh giá tính nhất quán thị sai khoảng cách xa |
| **KITTI 2015 (Test)** | Menze et al. | $20\text{ cặp}$ | $1242 \times 375$ | Cảnh giao thông phức tạp, người đi bộ, phản xạ kính lái | Đánh giá độ chịu lỗi khi có bóng ma (Ghosting) |
| **Quest 3S 8-Phase Trajectory** | Video thực tế từ kính (`WhatsApp Video`) | $8\text{ pha khép kín}$ | $854 \times 480$ | Quang học thấu kính Fresnel, méo góc ngoại vi, nhiễu cảm biến passthrough | Kiểm chứng trực tiếp trên kính Meta Quest 3S |

---

## 3. CHI TIẾT BỘ DỮ LIỆU 8 PHA QUỸ ĐẠO KHÉP KÍN (QUEST 3S TRAJECTORY)

Dữ liệu được cắt và lọc hình học tự động từ video thực tế của bạn, bao quát một chu trình di chuyển hoàn chỉnh từ trong phòng Lab, ra sảnh lớn, đi qua cổng xuất phát, ra sân ngoài trời và quay trở về bàn làm việc:

* **Pha 1 (`phase1_indoor_lab_vrui_anchor` - Train):** Bàn làm việc Lab, neo giao diện VR ảo trên tường, ánh sáng nhân tạo.
* **Pha 2 (`phase2_indoor_lab_desk_walk` - Train):** Di chuyển giữa các dãy bàn máy tính trong Lab, thị sai biến đổi liên tục.
* **Pha 3 (`phase3_penrose_hall_atrium` - Train):** Bước ra sảnh Penrose rộng lớn, trần nhà cao, gạch hoa văn phản xạ.
* **Pha 4 (`phase4_main_gate_exit` - Train):** Cửa chính kính đón ánh sáng ban ngày tương phản cao, chuyển tiếp sáng/tối.
* **Pha 5 (`phase5_outdoor_campus_brickwalk` - Test):** Đi trên lối đi gạch đỏ ngoài trời, ánh sáng tự nhiên đầy đủ, vân gạch lặp lại.
* **Pha 6 (`phase6_outdoor_patio_return` - Train):** Sân hiên ngoài trời, quay đầu chuẩn bị đi ngược vào trong tòa nhà.
* **Pha 7 (`phase7_indoor_portal_awards` - Test):** Bước qua cửa vòm vào sảnh trưng bày bảng vinh danh, chữ viết và khung kính.
* **Pha 8 (`phase8_return_lab_desk` - Test):** Quay trở lại bàn Lab ban đầu, khép kín vòng lặp quỹ đạo (Loop Closure).

---

## 4. HƯỚNG DẪN SỬ DỤNG DỮ LIỆU & SCRIPT HỖ TRỢ

1. **Kiểm tra tính toàn vẹn của dữ liệu:**
   ```powershell
   python scripts/benchmarks/evaluate_stereoinr_paper_metrics.py
   ```
2. **Chạy đối so trên toàn bộ 8 pha quỹ đạo Quest 3S:**
   ```powershell
   python scripts/benchmarks/benchmark_mamba_stereo_trajectory.py
   ```
3. **Chạy trực quan hóa so sánh đa mô hình có ảnh Inset Zoom và Heatmap:**
   ```powershell
   python scripts/visualization/visualize_stereo_comparison.py
   ```
