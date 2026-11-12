# BỘ SLIDES BÁO CÁO TOÀN DIỆN: STEREOINR & SCALE-LIBERATED XR (2026)
> **Thư mục tổng hợp:** [outputs/presentation_slides/](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/)  
> **Tệp PDF trình chiếu gộp (Master Slides):** [presentation_stereoinr_quest3s_2026.pdf](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/presentation_stereoinr_quest3s_2026.pdf)

Toàn bộ slide đã được render sẵn ở tỉ lệ chuẩn **16:9 (1920x1080 Full HD)** theo phong cách **Modern Dark Tech**, tích hợp sẵn hình ảnh đồ họa, biểu đồ pipeline cạnh nhau và kết quả so sánh trực quan. Bạn có thể copy trực tiếp từng ảnh bên dưới thả vào PowerPoint/Google Slides hoặc dùng ngay tệp PDF để thuyết trình.

---

## MỤC LỤC & DANH MỤC ẢNH SLIDE TRỰC QUAN

| STT | Slide | Tên Nội Dung | Đường Dẫn Tệp Ảnh 16:9 |
|:---:|---|---|---|
| **01** | Slide 1 | Trang Bìa Đề Tài & Mục Tiêu Nghiên Cứu | [slide_01_title.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_01_title.png) |
| **02** | Slide 2 (Phần 1) | Bộ Dữ Liệu 8 Pha Quỹ Đạo Kính Meta Quest 3S & Bản Chất GT | [slide_02_part1_dataset_gt.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_02_part1_dataset_gt.png) |
| **03** | Slide 3 (Phần 4) | **Đối Đầu 2 Pipeline Cạnh Nhau:** StereoINR (43.8M) vs. Scale-Liberated (4.47M) | [slide_03_part4_architecture_profiling_dual_pipeline.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_03_part4_architecture_profiling_dual_pipeline.png) |
| **04** | Slide 4 (Phần 3) | Quá Trình Huấn Luyện Chưng Cất Tri Thức & Bảng Đo Đạc Phần Cứng | [slide_04_part3_training_empirical_metrics.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_04_part3_training_empirical_metrics.png) |
| **05** | Slide 5 (Phần 5A) | So Sánh Trực Quan: Chi Tiết Cực Hạn (Middlebury `cloth2` & `motorcycle`) | [slide_05_visual_middlebury.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_05_visual_middlebury.png) |
| **06** | Slide 6 (Phần 5B) | So Sánh Trực Quan: Đô Thị & Cảnh Xa (KITTI 2012 `000000` & Flickr1024 `0001`) | [slide_06_visual_kitti_flickr.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_06_visual_kitti_flickr.png) |
| **07** | Slide 7 (Phần 5C) | So Sánh Trực Quan: 8 Pha Thực Tế Kính Quest 3S (Lab UI, Penrose, Sân gạch) | [slide_07_visual_quest3s_trajectory.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_07_visual_quest3s_trajectory.png) |
| **08** | Slide 8 | Tổng Kết Đóng Góp Khoa Học & Lộ Trình Triển Khai Kính XR | [slide_08_conclusion.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_08_conclusion.png) |

---

## CHI TIẾT NỘI DUNG & KỊCH BẢN THUYẾT TRÌNH TỪNG SLIDE

### SLIDE 1: TRANG BÌA (TITLE)
* **Hình ảnh slide:** [slide_01_title.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_01_title.png)

![Slide 1](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_01_title.png)

* **Lời thoại thuyết trình gợi ý:**
  > *"Kính thưa Hội đồng, đề tài của chúng em là **Scale-Liberated Stereo Implicit Neural Representations (StereoINR) Cho Kính Thực Tế Ảo Meta Quest 3S**. Nghiên cứu này xuất phát từ một bài toán vật lý cốt lõi: StereoINR gốc tại ACM MM 2025 tuy đạt chất lượng hình ảnh rất cao nhưng mô hình quá cồng kềnh ($43.8\text{M}$ tham số) và mất tới $7.5\text{ giây}$ cho 1 frame, hoàn toàn bất khả thi trên kính VR. Đề tài của chúng em tái cấu trúc toàn diện kiến trúc này về mức $4.47\text{M}$ tham số, tăng tốc $57.4\times$ ($130.65\text{ ms}$), và là công trình đầu tiên biên dịch thành công nhị phân QNN DLC trên NPU Qualcomm Hexagon của kính."*

---

### SLIDE 2: PHẦN 1 - BỘ DỮ LIỆU THỰC TẾ KÍNH QUEST 3S & BẢN CHẤT GROUND TRUTH
* **Hình ảnh slide:** [slide_02_part1_dataset_gt.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_02_part1_dataset_gt.png)

![Slide 2](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_02_part1_dataset_gt.png)

* **Điểm cốt lõi:**
  * **Thu thập luồng 2 mắt:** Trích xuất từ camera ghi hình Passthrough của kính Meta Quest 3S ($147.96\text{ giây}$, $8,876\text{ khung hình}$, $60\text{ FPS}$). Tách tự động 2 luồng Trái/Phải đồng bộ tuyệt đối ($t_L = t_R$, khoảng cách đồng tử baseline $B = 64\text{ mm}$).
  * **Bản chất Ground Truth:** Khung hình gốc từ kính chính là **Ground Truth (HR: $854 \times 480$)**. Áp dụng toán tử hạ mẫu Bicubic $4\times$ để tạo ra **Input (LR: $213 \times 120$)**. Đây là chuẩn mực số 1 trong nghiên cứu Super-Resolution quốc tế (như Middlebury, KITTI, Flickr1024).
  * **Chu trình 8 pha khép kín:** Di chuyển từ Bàn Lab $\to$ Sảnh Penrose $\to$ Cổng kính $\to$ Sân gạch ngoài trời $\to$ Bảng vinh danh $\to$ Quay lại bàn Lab. Pha 5, 7, 8 được giữ lại làm **Held-out Test Set** độc lập.
* **Lời thoại thuyết trình gợi ý:**
  > *"Trước hết, về bộ dữ liệu: Toàn bộ dữ liệu được chúng em thu thập trực tiếp từ video passthrough của kính Meta Quest 3S với 8 pha quỹ đạo khép kín. Ground Truth chính là các khung hình độ phân giải gốc của kính. Chúng em tạo ảnh đầu vào LR bằng phép hạ mẫu Bicubic $4\times$. Mô hình chỉ nhìn thấy ảnh LR, hoàn toàn không thấy ảnh gốc khi suy luận, đảm bảo tính khách quan và khoa học tuyệt đối."*

---

### SLIDE 3: PHẦN 4 - KIẾN TRÚC & PHẦN CỨNG (2 PIPELINES ĐỐI ĐẦU ĐẶT CẠNH NHAU)
* **Hình ảnh slide:** [slide_03_part4_architecture_profiling_dual_pipeline.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_03_part4_architecture_profiling_dual_pipeline.png)

![Slide 3](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_03_part4_architecture_profiling_dual_pipeline.png)

* **Điểm cốt lõi đối đầu:**
  * **Cột Trái - StereoINR Gốc (ACM MM 2025, 43.8M params):** Dùng Transformer 24 lớp + 2D Deformable Cross-Attention gọi hàm `F.grid_sample`. Toán tử này truy xuất bộ nhớ DRAM phân tán rải rác ngẫu nhiên, nghẽn băng thông ($119.2\text{ GB/s}$), tốn **$7,500\text{ ms}$** ($0.13\text{ FPS}$) và **$1,840\text{ MB}$** VRAM. Trên NPU Qualcomm Hexagon, mô hình bị **CRASH HOÀN TOÀN (`An internal error occurred`)** vì NPU không hỗ trợ GridSample động.
  * **Cột Phải - Scale-Liberated StereoINR (Ours, 4.47M params):** Tận dụng tính chất hình học đã cân chỉnh của kính VR (thị sai chỉ nằm trên trục ngang $y = y'$), chúng em thay thế bằng **1D Epipolar Parallax Attention (EPA-1D)** chuyển $100\%$ truy cập bộ nhớ thành **Tuần tự (Contiguous Memory)**. Kết hợp lớp điều chế **Dynamic Tanh (DyT)** và bộ giải mã tọa độ **Fourier H-LTE**. Kết quả: Giảm tham số $10\times$, giảm độ trễ về **$130.65\text{ ms}$ ($57.4\times$ speedup)**, tiết kiệm $4.3\times$ RAM ($426\text{ MB}$). Biên dịch **THÀNH CÔNG nhị phân phần cứng QNN DLC ($1.29\text{ MB}$, $513.89\text{ ms}$)** trên NPU Qualcomm Hexagon.
* **Lời thoại thuyết trình gợi ý:**
  > *"Đây là trọng tâm khoa học của đề tài: Trên slide là sơ đồ so sánh trực diện hai pipeline kiến trúc. Bên trái là StereoINR gốc: Việc dùng cơ chế 2D Deformable Attention với toán tử GridSample khiến dữ liệu bị nhảy rải rác trong DRAM, tốn $7.5\text{ giây}$ và làm NPU Hexagon bị crash. Bên phải là kiến trúc Scale-Liberated của chúng em: Bằng việc khai thác hình học Epipolar 1D và chuẩn hóa Dynamic Tanh, toàn bộ truy cập bộ nhớ trở thành tuần tự $100\%$. Chúng em đạt tốc độ $130.65\text{ ms}$ (nhanh gấp $57.4\times$) và biên dịch thành công tệp nhị phân DLC chạy trực tiếp trên NPU của chip Snapdragon XR2."*

---

### SLIDE 4: PHẦN 3 - QUÁ TRÌNH HUẤN LUYỆN & SỐ LIỆU THỰC TẾ
* **Hình ảnh slide:** [slide_04_part3_training_empirical_metrics.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_04_part3_training_empirical_metrics.png)

![Slide 4](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_04_part3_training_empirical_metrics.png)

* **Điểm cốt lõi:**
  * **Hàm mất mát chưng cất tri thức:**
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Charbonnier}} + 0.25 \mathcal{L}_{\text{Epipolar}} + 0.1 \mathcal{L}_{\text{FFT}} + 0.5 \mathcal{L}_{\text{KD}}$$
    Kháng nhiễu cảm biến, triệt tiêu tranh chấp võng mạc, khôi phục tần số cao của chữ viết vi tính và chưng cất tri thức từ Teacher StereoINR ($43.8\text{M}$).
  * **Bảng kết quả đo đạc đối đầu:**
    * Trên Quest 3S: PSNR đạt **$28.45\text{ dB}$** (vượt trội $+1.45\text{ dB}$ so với Bicubic).
    * Sai lệch thị sai (Stereo MAE): Giảm từ **$8.8\%$ đến $15.9\%$** so với baseline, bảo vệ thị giác người dùng trong môi trường VR.
* **Lời thoại thuyết trình gợi ý:**
  > *"Về quá trình huấn luyện: Chúng em áp dụng chiến lược Chưng cất Tri thức kết hợp 4 thành phần hàm mất mát. Bảng số liệu đo đạc thực tế cho thấy mô hình không chỉ đạt PSNR cao và tốc độ $130\text{ ms}$, mà quan trọng nhất đối với kính VR là chỉ số Stereo MAE giảm tới $15.9\%$, giúp hai mắt người dùng nhìn thấy hình ảnh lập thể đồng nhất tuyệt đối, ngăn ngừa mỏi mắt và đau đầu."*

---

### SLIDE 5: PHẦN 5A - SO SÁNH TRỰC QUAN: MIDDLEBURY
* **Hình ảnh slide:** [slide_05_visual_middlebury.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_05_visual_middlebury.png)

![Slide 5](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_05_visual_middlebury.png)

* **Chi tiết so sánh:**
  * `cloth2`: Tái hiện cấu trúc sợi vải đan chéo vi mô cực hạn. StereoINR đạt $33.48\text{ dB}$ ($+2.03\text{ dB}$), mô hình của chúng ta đạt $31.30\text{ dB}$ với bản đồ sai số xanh đồng đều.
  * `motorcycle`: Tái tạo các căm nan hoa kim loại và viền lốp xe. StereoINR đạt $25.74\text{ dB}$ ($+3.38\text{ dB}$), mô hình của chúng ta đạt $22.71\text{ dB}$ sắc nét hơn rõ rệt so với Bicubic ($22.36\text{ dB}$).

---

### SLIDE 6: PHẦN 5B - SO SÁNH TRỰC QUAN: KITTI & FLICKR1024
* **Hình ảnh slide:** [slide_06_visual_kitti_flickr.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_06_visual_kitti_flickr.png)

![Slide 6](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_06_visual_kitti_flickr.png)

* **Chi tiết so sánh:**
  * KITTI `000000`: Khôi phục sắc nét hàng rào cọc tiêu và cây xanh ven đường. StereoINR đạt $23.24\text{ dB}$ ($+1.66\text{ dB}$), mô hình của chúng ta đạt $21.90\text{ dB}$ ($+0.32\text{ dB}$).
  * Flickr1024 `0001`: Khôi phục chi tiết kiến trúc tòa nhà và chiều sâu khoảng cách xa. StereoINR đạt $22.04\text{ dB}$ ($+3.01\text{ dB}$), mô hình của chúng ta đạt $19.31\text{ dB}$.

---

### SLIDE 7: PHẦN 5C - SO SÁNH TRỰC QUAN: 8 PHA THỰC TẾ META QUEST 3S
* **Hình ảnh slide:** [slide_07_visual_quest3s_trajectory.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_07_visual_quest3s_trajectory.png)

![Slide 7](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_07_visual_quest3s_trajectory.png)

* **Chi tiết so sánh trên kính thực tế:**
  * **Pha 1 (Bàn Lab & Neo VR UI):** Cạnh chữ vi tính kích thước nhỏ ($10\text{pt}-12\text{pt}$) trên tường ảo được tái tạo sắc nét, không bị răng cưa hay nhòe vỡ font. Độ lợi crop đạt **$+3.83\text{ dB}$**.
  * **Pha 3 (Sảnh Penrose Atrium):** Khử hiện tượng phản xạ bóng ma trên gạch men sảnh lớn trần cao. Độ lợi crop đạt **$+4.06\text{ dB}$**.
  * **Pha 5 (Lối đi sân gạch đỏ ngoài trời):** Vân gạch lặp lại dưới ánh sáng chói chang được tách bạch rõ từng viên gạch. Độ lợi crop đạt **$+1.44\text{ dB}$**.

---

### SLIDE 8: TỔNG KẾT & LỘ TRÌNH TRIỂN KHAI XR
* **Hình ảnh slide:** [slide_08_conclusion.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_08_conclusion.png)

![Slide 8](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/slide_08_conclusion.png)

* **Lời thoại thuyết trình kết luận:**
  > *"Để tổng kết, đề tài đã đạt được 3 kết quả thực chất: Thứ nhất, làm chủ quy trình thu thập và xử lý dữ liệu chuẩn từ camera Passthrough của kính Meta Quest 3S. Thứ hai, giải phóng StereoINR khỏi nút thắt bộ nhớ 2D, mang lại tốc độ nhanh hơn $57.4\times$ và kích thước nhỏ hơn $10\times$. Thứ ba, chứng minh tính khả thi vật lý bằng việc biên dịch thành công tệp nhị phân QNN DLC trên chip Qualcomm. Trong tương lai, mô hình sẽ được tích hợp vào Unity Sentis và kỹ thuật Foveated Passthrough để đạt ngưỡng 90 FPS hoàn hảo. Xin chân thành cảm ơn Hội đồng!"*

---

## BỘ ẢNH SO SÁNH TRỰC QUAN ĐƠN LẺ ĐỘ PHÂN GIẢI CAO (RAW PANELS)
Nếu bạn muốn copy riêng từng ảnh panel so sánh chi tiết cao nhất (không có viền slide), tất cả đều nằm sẵn trong cùng thư mục này:

1. [comparison_middlebury_cloth2.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/comparison_middlebury_cloth2.png)
2. [comparison_middlebury_motorcycle.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/comparison_middlebury_motorcycle.png)
3. [comparison_kitti2012_000000.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/comparison_kitti2012_000000.png)
4. [comparison_flickr1024_0001.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/comparison_flickr1024_0001.png)
5. [comparison_quest3s_harvested_phase1_indoor_lab_vrui_anchor.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/comparison_quest3s_harvested_phase1_indoor_lab_vrui_anchor.png)
6. [comparison_quest3s_harvested_phase3_penrose_hall_atrium.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/comparison_quest3s_harvested_phase3_penrose_hall_atrium.png)
7. [comparison_quest3s_harvested_phase5_outdoor_campus_brickwalk.png](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/presentation_slides/comparison_quest3s_harvested_phase5_outdoor_campus_brickwalk.png)
8. Thư viện web tương tác: [index.html](file:///d:/Gitrepo/Active%20Stereo%20RL/outputs/visual_comparisons/index.html)
