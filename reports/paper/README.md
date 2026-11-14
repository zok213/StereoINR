# Research Paper Package: Scale-Liberated StereoINR
## Hướng Dẫn Biên Dịch & Nộp Báo Cáo Khoa Học (Submission Guide)

Tài liệu này đóng gói bài báo khoa học hoàn chỉnh (`main.tex`), kèm toàn bộ dữ liệu hình ảnh, bảng biểu thực nghiệm và thư viện trích dẫn tài liệu tham khảo cho đề tài:
**"Scale-Liberated StereoINR: Continuous DyT-EdgeINR with 1D Epipolar Parallax Attention for Meta Quest 3S Passthrough"**

---

## 1. Cấu Trúc Thư Mục Gói Báo Cáo (`reports/paper/`)

```text
reports/paper/
├── main.tex                     # Tệp mã nguồn LaTeX chính (431 dòng)
├── references.bib               # Cơ sở dữ liệu tài liệu tham khảo BibTeX (25 tài liệu chuẩn)
├── README.md                    # Hướng dẫn biên dịch và tổng quan nội dung cho giảng viên
└── figures/                     # Toàn bộ hình ảnh và sơ đồ độ phân giải cao phục vụ bài báo
    ├── architecture_schematic_dual_pipeline.png        # Hình 1: So sánh sơ đồ kiến trúc StereoINR vs DyT-EdgeINR
    ├── memory_access_epipolar_comparison.png           # Hình 2: Phân tích luồng đọc DRAM 2D ngẫu nhiên vs 1D tuần tự
    ├── foveated_passthrough_reconstruction_demo.png    # Hình 3: Toàn bộ pipeline Foveated Passthrough 4.2ms
    ├── training_loss_curves.png                        # Hình 4: Đồ thị hội tụ hàm mất mát 12 epochs trên Dual T4
    ├── benchmark_psnr_15_scenes.png                    # Hình 5: Biểu đồ cột PSNR đối sánh trên 15 cảnh thực nghiệm
    ├── correlation_pareto_matrix.png                   # Hình 6: Ma trận tương quan và đường biên tối ưu Pareto
    ├── qualcomm_npu_profiling_breakdown.png            # Hình 7: Kết quả profiling NPU Qualcomm Hexagon V73
    ├── hardware_latency_budget_breakdown.png           # Hình 8: Phân tích ngân sách thời gian 11.11ms của kính 90Hz
    ├── comparison_middlebury_cloth2.png                # Mẫu so sánh trực quan phòng Lab Middlebury
    ├── comparison_middlebury_motorcycle.png            # Mẫu so sánh trực quan vi chi tiết vi mô
    ├── comparison_kitti2015_000000.png                 # Mẫu so sánh thị sai đường phố xe tự hành
    └── comparison_quest3s_harvested_phase7_indoor_portal_awards.png # Mẫu đọc bảng chữ thực tế Quest 3S
```

Gói nén độc lập sẵn sàng tải lên Overleaf: [`reports/paper_bundle.zip`](../paper_bundle.zip) ($13.7\text{ MB}$).

---

## 2. Hướng Dẫn Biên Dịch PDF

### Cách 1: Nộp hoặc Biên dịch trên Overleaf (Khuyên Dùng - Đơn Giản Nhất)
1. Tải tệp [`reports/paper_bundle.zip`](../paper_bundle.zip) lên Overleaf (`New Project` $\to$ `Upload Project`).
2. Chọn trình biên dịch là `pdfLaTeX` hoặc `XeLaTeX`.
3. Nhấn `Recompile`. Toàn bộ văn bản, công thức toán KaTeX/LaTeX, bảng biểu và 8 hình ảnh sẽ được kết xuất sạch sẽ không có lỗi đường dẫn.

### Cách 2: Biên dịch qua dòng lệnh máy trạm (Command Line)
```bash
cd reports/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 3. Tóm Tắt Các Bảng Biểu & Kết Quả Thực Nghiệm Cốt Lõi

* **Bảng I (15-Scene Crop PSNR Benchmark):**  
  So sánh đối chuẩn Crop PSNR trên 15 cảnh (8 pha quỹ đạo Quest 3S và 7 cảnh quốc tế Middlebury, KITTI, Flickr1024).  
  * Bicubic: $24.19\text{ dB}$  
  * DyT-EdgeINR (Ours, 4.47M params): **$24.74\text{ dB}$ ($+0.56\text{ dB}$ gain)**  
  * StereoINR Teacher (43.8M params): $27.32\text{ dB}$ ($+2.58\text{ dB}$ do quy mô tham số gấp $10\times$ và thời gian train dài).

* **Bảng II (Stereo Disparity Consistency - Stereo MAE $\downarrow$):**  
  Chứng minh mô hình của chúng ta đạt sai lệch thị sai thấp nhất trong số các mô hình đối đầu trực tiếp:  
  * Middlebury: **0.1290** (Giảm **-8.77%** lỗi so với Teacher 0.1414, tốt hơn Bicubic 0.1326).  
  * KITTI 2012: **0.0478** (Giảm **-14.34%** lỗi so với Teacher 0.0558, tốt hơn Bicubic 0.0492).  
  * KITTI 2015: **0.0502** (Giảm **-15.91%** lỗi so với Teacher 0.0597, tốt hơn Bicubic 0.0522).  
  * Flickr1024: **0.1348** (Giảm **-8.67%** lỗi so với Teacher 0.1476, tốt hơn Bicubic 0.1360).  
  * *Ý nghĩa vật lý:* Triệt tiêu hiện tượng lệch pha scanline, bảo vệ mắt người dùng khỏi hiện tượng Retinal Rivalry (tranh chấp võng mạc).

* **Bảng III (Qualcomm Hexagon V73 Hardware Execution & INT8 Quantization):**  
  * Phân bổ toán tử NPU: **1,495 / 1,495 (100% Native NPU)**, $0\%$ CPU fallback (StereoINR gốc crash với `MODEL_GRAPH_ERROR`).  
  * Độ trễ Foveated Patch ($256 \times 256$): **$4.20\text{ ms}$**.  
  * Tổng độ trễ Glass-to-Glass: **$6.16\text{ ms} \le 11.11\text{ ms}$** (Đạt **162.3 sustained FPS**, vượt ngân sách 90 FPS).  
  * Bộ nhớ RAM đỉnh: **$77.29\text{ MB}$** ($<1\%$ tổng 8GB RAM kính).

* **Bảng IV (Dual NVIDIA Tesla T4 Server Benchmark):**  
  * Tham số: Giảm từ $43.79\text{M}$ xuống **$4.47\text{M}$** ($9.8\times$ nhỏ hơn).  
  * Kích thước INT8: Giảm từ $41.8\text{ MB}$ xuống **$4.26\text{ MB}$**.  
  * Độ trễ GPU: Tăng tốc **$57.4\times$** (từ $7,500.00\text{ ms}$ xuống **$130.65\text{ ms}$**).  
  * VRAM: Giảm $4.3\times$ (từ $1,840.0\text{ MB}$ xuống **$426.0\text{ MB}$**).

---

## 4. Tinh Thần Học Thuật & Đánh Giá Đa Chiều (Academic Humility)

Báo cáo được trình bày với tinh thần khoa học khách quan và trung thực:
1. Thừa nhận sòng phẳng khoảng cách $-2.58\text{ dB}$ PSNR so với mô hình giáo viên khổng lồ 43.8M do thâm hụt thời gian huấn luyện (60 epochs / 1.45 giờ Kaggle so với 300K-800K iterations của mô hình tham chiếu).
2. Nêu bật sự đánh đổi kỹ thuật có chủ đích: Hy sinh một phần độ nét vi mô đơn mắt để đổi lấy tính ổn định thị giác lập thể hai mắt và khả năng thực thi thời gian thực trên chip di động.
3. Ghi nhận trung thực các rào cản nền tảng: Meta Horizon OS SELinux sandbox lockout (buộc ứng dụng thương mại bên thứ 3 phải chạy qua GPU Adreno 740 Unity Sentis / Vulkan) và giới hạn Fixed Foveation do kính Quest 3S không có camera Eye-Tracking phần cứng.
