Họ tên: Nguyễn Đức Triệu
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | - | - | 0.0950 | [0.0681, 0.1246] | 0.7685 | 0.1457 | 0.0260 | 0.1920 |
| E1 | LoRA LLM | 33,030,144 | 6.55 | 41.58 | 0.2791 | [0.2408, 0.3188] | 1.2379 | 0.1117 | 0.2216 | 0.5462 |
| E2 | + projector | 142,135,296 | 6.13 | 51.22 | 0.3423 | [0.3035, 0.3817] | 1.0064 | 0.1347 | 0.2653 | 0.6456 |
| E3 | + LoRA ViT | 148,426,752 | 6.23 | 60.29 | 0.4385 | [0.3986, 0.4794] | 0.9196 | 0.3015 | 0.3501 | 0.7138 |
| E4 | rank: r=4 vs r=64 | 118,935,552 (r=4) / 266,391,552 (r=64) | 5.96 / 7.46 | 60.16 / 61.58 | 0.3993 / 0.4262 | [0.3610, 0.4425] / [0.3850, 0.4732] | 0.9357 / 0.8328 | 0.2613 / 0.2463 | 0.3179 / 0.3489 | 0.6644 / 0.7083 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Qwen3-VL-4B có tổng cộng khoảng 4.44B tham số, được phân bổ như sau:
- **LLM**: Chiếm khoảng 4.0B tham số (36 transformer layers, hidden size 2560, intermediate size 9728).
- **ViT (Vision Transformer)**: Chiếm khoảng 415M tham số (24 visual blocks).
- **Projector**: Chiếm khoảng 109.11M tham số gồm hai phần: `visual.merger` (~27.2M tham số) nằm ở cuối ViT và `visual.deepstack_merger_list` (~81.9M tham số) trích xuất đặc trưng đa tầng từ các lớp 5, 11, 17 của ViT.

Các khối được cập nhật trong từng thí nghiệm:
- **E1**: Chỉ gắn adapter LoRA trên LLM (36 layers × 7 modules `{q, k, v, o, gate, up, down}_proj` = 252 modules, cập nhật **33,030,144** tham số). ViT và projector bị đóng băng hoàn toàn.
- **E2**: Giữ nguyên LoRA LLM (33.03M) và mở huấn luyện toàn bộ tham số của Projector (`visual.merger` và `visual.deepstack_merger_list`, thêm 109,105,152 tham số ở float32) -> cập nhật tổng cộng **142,135,296** tham số. Các khối ViT blocks vẫn đóng băng.
- **E3**: Giữ nguyên LoRA LLM và Projector mở đầy đủ, đồng thời gắn thêm LoRA trên các khối attention và MLP của ViT (24 blocks × 4 modules `{qkv, proj, linear_fc1, linear_fc2}` = 96 modules, thêm 6,291,456 tham số) -> cập nhật tổng cộng **148,426,752** tham số.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- Tỷ lệ `parse_valid_rate` của mô hình gốc (E0) đạt **73.95%** (230/311 ảnh parse thành công, 81/311 ảnh không thể parse được JSON hợp lệ).
- Hai lỗi phổ biến nhất của mô hình gốc:
  1. **Sinh cấu trúc JSON không hoàn chỉnh / Lời thoại thừa**: Mô hình zero-shot thường sinh văn bản giải thích tự do (như *"Here is the detected damage..."*), quên đóng ngoặc `]` hoặc lặp từ liên tục dẫn đến chạm ngưỡng `max_new_tokens=256` và bị cắt cụt giữa chừng khiến cú pháp JSON bị hỏng.
  2. **Ảo giác toạ độ và sai lệch thang đo**: Mô hình chưa học được quy ước grounding chuẩn hoá 0–1000 cho bài toán; nó thường nhầm sang thang pixel thực tế (ví dụ box vượt quá 1000 hoặc toạ độ float trong khoảng [0, 1]), hoặc bịa ra các bounding box bao quanh các vật thể không liên quan như xe cộ, bóng râm hay vạch kẻ đường.
- Ví dụ output thô bị lỗi:
```text
```json
[{"label": "road_damage", "bbox_2d": [0.35, 0.62, 0.58, 0.81]}, {"label": "crack", "bbox_2d": [120, 340, 260
```
*(Output này vừa dùng nhãn không hợp lệ "road_damage", vừa dùng toạ độ chuẩn hoá [0,1], vừa bị cắt cụt không đóng ngoặc JSON do hết token).*

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- Nếu mô hình dự đoán theo thang 0–1000 nhưng trọng tài lại chấm theo toạ độ pixel tuyệt đối (ảnh gốc kích thước 640×360):
  - Một box chuẩn ở giữa ảnh có toạ độ 0–1000 là `[400, 300, 600, 500]` sẽ bị hiểu là pixel từ $x=400$ đến $600$ và $y=300$ đến $500$. Trong khi đó, toạ độ pixel thực tế của ground truth tương ứng phải là $x \in [256, 384]$ và $y \in [108, 180]$.
  - Hai vùng này nằm lệch hẳn nhau trên khung hình $640 \times 360$, dẫn tới độ giao diện tích Intersection over Union (IoU) rơi về xấp xỉ 0.0 (không đạt ngưỡng $IoU \ge 0.5$). Số True Positive (TP) sẽ bằng 0, khiến **F1 tụt dốc về đúng 0.00**.
- **Kiểm chứng**: Khi chạy kiểm thử `score.py` với cờ `--coord pixel` trên file dự đoán thang 0–1000 của E1, kết quả F1 đo được ngay lập tức rớt từ 0.2791 xuống **0.0000** do không khớp bất kỳ box nào.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- Ta bắt buộc phải gán `labels[:, :prompt_len] = -100` để hàm mất mát CrossEntropy chỉ tính đạo hàm trên các token thuộc chuỗi JSON trả lời của trợ lý (`assistant`).
- Nếu tính loss trên cả prompt và token ảnh:
  1. **Nhiễu loạn mục tiêu tối ưu**: Chuỗi câu lệnh prompt văn bản và 220 token ảnh thị giác chiếm hơn 80% tổng chiều dài token của mỗi mẫu. Nếu tính loss toàn bộ, phần lớn gradient sẽ bị lãng phí vào việc ép mô hình "học thuộc lòng" câu hỏi và tái tạo lại các vector đặc trưng của ảnh (vốn là đầu vào cố định).
  2. **Mô hình bị suy thoái khả năng grounding**: Trọng số cập nhật không tập trung vào quy luật sinh toạ độ `bbox_2d` và nhãn phân loại, dẫn tới mô hình sinh ra cú pháp JSON kém chuẩn xác và F1 giảm nghiêm trọng.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- Trong bitsandbytes, các tầng Linear lượng tử hóa 4-bit (NF4) lưu trọng số dưới dạng `Params4bit` (kiểu dữ liệu số nguyên `torch.uint8`). PyTorch quy định nghiêm ngặt: chỉ có tensor số thực (floating point) mới được phép bật `requires_grad=True`; nếu bật trên tensor `uint8` sẽ báo lỗi `RuntimeError: only Tensors of floating point and complex dtype can require gradients`. Do đó, nếu projector bị lượng tử hoá 4-bit, các ma trận trọng số không thể tính gradient và cập nhật được.
- **Cách xử lý**:
  1. Khi nạp mô hình, cấu hình `llm_int8_skip_modules=["visual", "lm_head"]` để giữ toàn bộ nhánh ViT và Projector ở định dạng FP16.
  2. Trong hàm `apply_setup`, đệ quy chuyển đổi toàn bộ các lớp tuyến tính bên trong `visual.merger` và `visual.deepstack_merger_list` thành `torch.nn.Linear` chuẩn ở kiểu dữ liệu `float32`.
- Projector phải ở `float32` vì đây là tầng ánh xạ then chốt biến đổi đặc trưng hình ảnh của ViT sang không gian nhúng của LLM. Độ nhạy gradient ở tầng này rất cao; giữ `float32` giúp tránh hiện tượng tràn số / tắt gradient (underflow) và giúp `GradScaler` không bị lỗi unscale FP16 gradient.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- Dữ liệu dashcam thực chất là các khung hình cắt từ video quay liên tục theo thời gian. Hai khung hình kề nhau (cùng 1 giây) có góc nhìn, bối cảnh mặt đường, vết nứt và điều kiện thời tiết gần như trùng khớp 100%. Nếu chia ngẫu nhiên (random split), các khung hình gần như giống nhau sẽ nằm ở cả tập train và tập test, gây ra hiện tượng **rò rỉ dữ liệu nghiêm trọng (data leakage)**. Mô hình chỉ cần ghi nhớ cảnh nền là đoán đúng, dẫn đến điểm F1 trên tập test cao giả tạo nhưng sẽ hoàn toàn mất tác dụng khi triển khai thực tế trên cung đường mới.
- Chia theo phiên quay (`session`) gom toàn bộ các frame của một chuyến đi vào duy nhất một tập split, phản ánh chính xác khả năng tổng quát hóa trên các phiên quay chưa từng gặp.
- Thống kê chi tiết từ `data/splits/meta.json`:
  - **Train**: 1,422 ảnh | 1,074 pothole, 1,750 crack, 691 manhole (Tổng cộng: 3,515 boxes).
  - **Val**: 276 ảnh | 66 pothole, 470 crack, 103 manhole (Tổng cộng: 639 boxes).
  - **Test**: 311 ảnh | 121 pothole, 299 crack, 163 manhole (Tổng cộng: 583 boxes).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **E1 (LoRA LLM)**: F1 = **0.2791** (CI95: [0.2408, 0.3188]), VRAM đỉnh 6.55 GB, thời gian train 41.58 phút.
- **E2 (+ Projector)**: F1 = **0.3423** (CI95: [0.3035, 0.3817]), VRAM đỉnh 6.13 GB, thời gian train 51.22 phút.
  - Điểm F1 tăng vọt **+6.32%** so với E1. Khoảng tin cậy [0.3035, 0.3817] tách biệt hoàn toàn và nằm trên cận trên của E1 (0.3188). Mở projector thích ứng rất tốt không gian biểu diễn thị giác vào LLM, hoàn toàn xứng đáng với 10 phút huấn luyện tăng thêm.
- **E3 (+ LoRA ViT)**: F1 = **0.4385** (CI95: [0.3986, 0.4794]), VRAM đỉnh 6.23 GB, thời gian train 60.29 phút.
  - Điểm F1 tăng thêm **+9.62%** so với E2 và tăng **+15.94%** so với E1! Khoảng tin cậy CI95 của E3 ([0.3986, 0.4794]) vượt xa cận trên của E2. Việc mở thêm LoRA trên ViT blocks giúp bộ trích xuất đặc trưng hình ảnh tinh chỉnh trực tiếp theo các hư hại nhỏ và mờ nhạt trên mặt đường.
- **Kết luận**: Mở cả Projector và LoRA ViT (E3) là phương án **hoàn toàn xứng đáng**, đem lại bước nhảy vọt về độ chính xác với mức tiêu thụ VRAM chỉ quanh mức 6.2 GB.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- Tập test có 311 ảnh (với 583 box), độ rộng của khoảng tin cậy bootstrap 95% (CI95) dao động trong khoảng $\pm 0.04$ (tương đương biên độ khoảng 0.08). Cụ thể ở E3, CI95 là `[0.3986, 0.4794]`.
- **Giới hạn kết luận**:
  - Với biên độ dao động ~0.08, nếu hai phương pháp chỉ chênh lệch nhau 1% – 2% F1 thì khoảng tin cậy của chúng sẽ chồng lấn nhau nặng nề, không thể khẳng định phương pháp nào vượt trội hơn một cách có ý nghĩa thống kê (statistically significant).
  - Tuy nhiên, độ chênh lệch giữa E3 (0.4385) so với E1 (0.2791) lên tới gần 16% và vượt xa biên độ CI95 (cận dưới của E3 là 0.3986 cao hơn hẳn cận trên của E1 là 0.3188). Do đó, kết luận "E3 vượt trội hơn E1 và E2" là hoàn toàn vững chắc và có ý nghĩa thống kê rõ ràng.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Lớp đạt hiệu quả cao nhất**: `manhole` (nắp cống) luôn dẫn đầu với F1 = **0.7138** ở E3 (Precision 0.7910, Recall 0.6503).
- **Lớp kém nhất**: `pothole` (ổ gà) đạt F1 thấp nhất với **0.3015** (Precision 0.3846, Recall 0.2479) và `crack` (vết nứt) đạt F1 = **0.3501**.
- **Nguyên nhân chi tiết**:
  1. **Hình học và ranh giới**: `manhole` có hình tròn/chữ nhật đồng nhất, kết cấu sắt/bê tông tương phản cao và ranh giới rất sắc nét. Ngược lại, `crack` là các đường nứt mảnh, zíc zắc và diện tích trung vị cực nhỏ (chỉ chiếm ~0.7% diện tích ảnh), rất dễ bị mất chi tiết khi nén qua 220 vision tokens.
  2. **Ngữ nghĩa dễ nhầm lẫn**: `pothole` có hình dạng méo mó bất kỳ, màu sắc hòa vào mặt nhựa đường và thường bị nhầm lẫn với vệt vá đường, bóng râm cây cối hoặc vũng nước phản chiếu ánh sáng (dẫn đến False Positive cao và Recall thấp).
- **Ảnh minh họa**: Đã được lưu và thể hiện trong file [`report/figures/prediction_examples.png`](file:///home/trieu/ai_in_action/track4_lab10/K4-Track-4-Day10-VisionLanguageModel/report/figures/prediction_examples.png), trực quan hóa rõ ràng các ca dự đoán đúng của nắp cống và các lỗi phát hiện nhầm bóng râm thành ổ gà.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến được chọn**: Thí nghiệm Ablation về **LoRA Rank ($r$)** trên setup tốt nhất (E3), so sánh giữa ba giá trị $r=4$ ($\alpha=8$), $r=16$ ($\alpha=32$), và $r=64$ ($\alpha=128$).
- **Kết quả đo được**:
  - **$r=4$** (118.9M tham số): F1 = **0.3993** (CI95: [0.3610, 0.4425]), VRAM 5.96 GB, thời gian train 60.16 phút.
  - **$r=16$ (Baseline E3)** (148.4M tham số): F1 = **0.4385** (CI95: [0.3986, 0.4794]), VRAM 6.23 GB, thời gian train 60.29 phút.
  - **$r=64$** (266.4M tham số): F1 = **0.4262** (CI95: [0.3850, 0.4732]), VRAM 7.46 GB, thời gian train 61.58 phút.
- **Đánh giá độ tin cậy**:
  - Khi giảm rank xuống $r=4$, F1 giảm từ 0.4385 xuống 0.3993 do năng lực biểu diễn của ma trận thích ứng bị hạn chế.
  - Khi tăng rank lên $r=64$, số tham số huấn luyện tăng gần gấp đôi (từ 148M lên 266M) và VRAM tăng lên 7.46 GB, nhưng F1 không tăng mà lại giảm nhẹ về 0.4262. Khoảng tin cậy CI95 của $r=64$ ([0.3850, 0.4732]) chồng lấn hoàn toàn với CI95 của $r=16$.
  - Kết luận: Với tập dữ liệu kích thước nhỏ (1.4k ảnh), rank $r=16$ là điểm tối ưu cân bằng nhất; việc tăng rank lên $r=64$ không mang lại lợi ích thống kê nào mà chỉ làm tăng chi phí tính toán và nguy cơ overfitting.

## Nhìn lại

Qua quá trình thực nghiệm, bài lab giúp hiểu rõ cơ chế tương tác đa phương thức trong VLM (từ ViT patch tokens qua DeepStack projector đến LLM). Điều thú vị là các lỗi dự đoán thực tế của mô hình: với các vết nứt dài liên tục, mô hình thường dự đoán thành nhiều box rời rạc nối tiếp nhau hoặc một box lớn bao phủ, dẫn đến độ đo IoU@0.5 phạt nhầm mặc dù mô hình đã định vị đúng vùng tổn hại mặt đường. Về mặt kỹ thuật, việc xử lý đồng bộ kiểu dữ liệu float32 cho các tham số cần huấn luyện là bài học quan trọng để tối ưu hóa mixed-precision trên GPU T4 mà không gặp lỗi số học.
