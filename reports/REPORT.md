# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Lưu Thị Lan Anh

Công cụ gán nhãn đã dùng: CVAT v2.76.0 chạy bằng Docker trên máy cá nhân; import/export định dạng
Ultralytics YOLO Detection 1.0, đóng gói bằng `tools/pack_labels.py`.

Nguồn số liệu: `reports/rounds_table.md`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json`,
`outputs/selection_round1.csv`, `outputs/round1_diff.md`, `outputs/compare_round0.jpg`,
`outputs/compare_round1.jpg` và lần chạy đối chứng trong `outputs/experiments/round1_baseline_e50_b16/`.
Nhãn test do một mô hình khác tạo và chưa được người rà, nên mọi số AP50/P/R dưới đây là **mức khớp
với bộ tham chiếu này**, không phải độ chính xác tuyệt đối.

## 1. Dữ liệu và cách chia tập

Video quay bằng camera cố định, trích 2.5 frame/giây. Hai frame liền nhau chỉ cách 0.4 s và một
chiếc xe nằm trong khung hình vài giây, nên các frame gần nhau gần như là cùng một ảnh. Nếu chia
ngẫu nhiên, cùng một chiếc xe ở cùng vị trí sẽ xuất hiện ở cả pool (được gán nhãn rồi train) và
test. Mô hình khi đó được chấm trên chính những chiếc xe nó đã học, nên số đo test sẽ **cao hơn
thực tế** (lệch lạc quan, do rò rỉ dữ liệu), và không cho biết model có tổng quát sang dòng xe mới
hay không.

Vì vậy dữ liệu được chia theo trục thời gian: 20 ảnh test lấy ở 4 đoạn quanh giây 20, 60, 100 và
140; 112 ảnh vùng đệm (±4 s quanh mỗi đoạn test) bị bỏ; 268 ảnh còn lại là pool. Ảnh pool gần test
nhất vẫn cách 4.4 s (`data/DATA.md`). Tôi không sửa `data/test/labels/` và không đưa ảnh test vào
`labels/round1/`; `tools/check_submission.py` kiểm hash nhãn test khớp bản phát hành.

## 2. Mô hình khởi đầu lạnh (cold start)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Cold start rất **thận trọng**: precision 0.925 (16 FP) nhưng recall chỉ 0.489 (bỏ sót 206/403 box
tham chiếu, `metrics_round0.json`). Trên `compare_round0.jpg`, các box vàng (bỏ sót) tập trung ở:

- **Xe nhỏ ở xa**, gần đường chân trời, chỉ còn cụm đèn đỏ: recall small = 0.182 (12/66).
- **Xe lớn ở gần bị lóa đèn pha**, sát mép dưới (ví dụ frame_0050 và frame_0250 góc dưới trái).
  Recall large chỉ 0.561, gần bằng medium: ảnh đêm với đèn pha quá sáng làm mất đường viền thân
  xe, điều COCO (chủ yếu ảnh ban ngày) ít gặp.
- **Xe nối đuôi nhau**: ở frame_0350, cold start vẽ một box đỏ lớn gộp nhiều xe ở làn trái (FP),
  đồng thời bỏ sót từng xe bên trong.

Recall theo kích thước cho thấy điểm yếu không chỉ là xe nhỏ; model chưa quen với miền ảnh đêm ở
mọi kích thước.

Ca cần người rà nhãn tham chiếu trước khi kết luận model sai: **frame_0250, mép phải**. Tham chiếu
có một box rất hẹp ngay mép ảnh (khoảng x 1260 trên ảnh so sánh) và một box cho xe nhoè bên dưới.
Box hẹp này có thể chỉ là một phần xe bị cắt ở mép, hoặc là vệt đèn; sau fine-tune model vẽ một box
lớn hơn ôm cả vùng nhoè và bị tính là FP. Nếu tham chiếu vẽ sai phạm vi thì FP này là lỗi của bộ
tham chiếu chứ không phải của model.

## 3. Chiến lược chọn mẫu

`score = W_U·U + W_A·A + W_D·D` với trọng số mặc định 0.5 / 0.3 / 0.2:

- **U (độ bất định)**: với mỗi box, `u = 1 − |2·conf − 1|`, bằng 1 khi conf = 0.5 (model phân vân
  nhất) và bằng 0 khi conf gần 0 hoặc 1. U là trung bình 5 giá trị u lớn nhất của frame, tức là đo
  các box khó nhất.
- **A (box mơ hồ)**: số box có 0.15 ≤ conf < 0.50, chia cho giá trị lớn nhất trong pool. Frame có
  nhiều xe "nửa thấy nửa không" được ưu tiên.
- **D (đa dạng thời gian)**: khoảng cách tới frame đã gán gần nhất, chặn ở 10 s. Vòng 1 chưa có
  frame nào được gán nên D = 1 với mọi frame.
- **`MIN_GAP_S = 2.0`**: khi chọn tham lam theo score, bỏ frame cách frame đã chọn dưới 2 s, vì
  camera cố định nên hai frame sát nhau gần như trùng lặp: gán cả hai tốn gấp đôi công mà model
  học thêm rất ít.

Chi tiết trong `reports/SELECTION.md`. Tóm tắt:

- Ba frame trong lô: **frame_0182** (hạng 1, score 0.959, A = 1.0, bản cuối thêm 12 xe),
  **frame_0331** (hạng 5, A = 1.0, xoá 5 box AI sai và thêm 11 xe), **frame_0392** (hạng 15,
  U = 0.975 cao nhất top 50, thêm 14 xe).
- Frame bị loại vì gần trùng: **frame_0372** (hạng 6, score 0.910) cao điểm hơn 7 ảnh được chọn
  nhưng chỉ cách frame_0369 1.2 s.
- Với ngân sách chỉ năm ảnh, tôi chọn 0182, 0369, 0326, 0099, 0227 để trải đều video thay vì lấy
  đúng năm điểm cao nhất (bốn trong số đó nằm trong đoạn 130–152 s).

Điểm bất định **không chứng minh** ảnh đó sẽ cải thiện model. U đo mức model phân vân, không đo
mức model sai: một chiếc xe bị bỏ sót hoàn toàn (không có box nào) không đóng góp gì vào U. Ví dụ
frame_0099 không có box gợi ý nào cho SUV lớn sát mép dưới, frame_0312 bỏ sót cả xe tải trắng. Ngoài
ra lô 12 ảnh không có ảnh nào trong 39 giây đầu video, vì ở vòng 1 thành phần D không phân biệt
được frame nào.

## 4. Các vòng học chủ động (active learning)

### Bảng kết quả

`reports/rounds_table.md` (lần chạy đã nộp):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 290 | 0.925 | +0.154 | 0.646 | 0.950 | 0.769 | 0.818 | 0.976 | 0.976 |

### Mức độ sửa nhãn gợi ý (`outputs/round1_diff.md`)

Model đề xuất 169 box (conf ≥ 0.25) trên 12 ảnh; bản cuối có **290 box**: accepted 121, edited 32,
deleted 16 (FP của model), **added 137** (FN của model). Accept rate 72% nhưng con số dễ gây hiểu
nhầm: trong 290 box cuối, chỉ 121 box (42%) là giữ nguyên từ AI. Lỗi chính của pre-label là
**bỏ sót**, không phải vẽ sai: mỗi ảnh được thêm 8–15 xe.

### Hai lần fine-tune trên cùng lô nhãn

Lần chạy đầu dùng cấu hình mặc định của notebook và cho kết quả **tệ hơn cold start**
(`outputs/experiments/round1_baseline_e50_b16/metrics_round1.json`):

| Lần chạy | epochs | batch | freeze | Số bước cập nhật | AP50 | P@0.25 | R@0.25 | R small / medium / large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Đối chứng (mặc định) | 50 | 16 | 0 | 50 | 0.380 | 1.000 | 0.109 | 0.000 / 0.088 / 0.439 |
| Đã nộp | 100 | 4 | 10 | 300 | **0.925** | 0.646 | 0.950 | 0.818 / 0.976 / 0.976 |

Nguyên nhân tôi xác định: với 12 ảnh và batch 16, mỗi epoch chỉ có 1 bước cập nhật trọng số, nên
50 epoch là 50 bước. Ultralytics mặc định warmup ít nhất 100 bước, nên cả quá trình huấn luyện nằm
trong giai đoạn learning rate còn đang tăng. Thêm vào đó, đầu phân lớp được khởi tạo lại cho 1 lớp
`car`, nên kiến thức lớp xe từ COCO bị mất. Kết quả là model gần như chưa học: không có FP nào nhưng
bỏ sót 359/403 xe.

Tôi đổi cấu hình **một lần**, theo lập luận trên chứ không thử nhiều cấu hình rồi chọn số đẹp nhất:
batch 4 (3 bước/epoch) và 100 epoch cho 300 bước, vượt xa warmup; `freeze = 10` đóng băng backbone
để giữ đặc trưng đã học từ COCO và giảm nguy cơ quá khớp vào 12 ảnh. Cấu hình được ghi trong
`metrics_round1.json` (`epochs`, `train_batch`, `freeze`).

### AP50 và các nhóm xe thay đổi thế nào

- **AP50**: 0.771 → 0.925 (**+0.154** so với cold start; vòng 1 là vòng đầu nên so với vòng trước
  cũng là +0.154). Mức chênh lớn hơn nhiều so với ngưỡng ~0.01 mà `DATA.md` cảnh báo là nhiễu.
- **Recall tăng ở mọi kích thước**: small 0.182 → 0.818, medium 0.547 → 0.976, large 0.561 → 0.976.
  Nhóm được lợi nhất là xe nhỏ ở xa và xe lớn bị lóa, đúng hai loại tôi thêm nhiều nhất khi sửa nhãn.
- **Precision giảm**: 0.925 → 0.646. FP tăng từ 16 lên 210, FN giảm từ 206 xuống 20. Model chuyển
  từ quá thận trọng sang "thấy xe ở khắp nơi".

### Một ca kết quả đổi sau fine-tune (`compare_round1.jpg`)

- **Tốt hơn, frame_0350**: cold start TP 9 / FP 2 / FN 14, có một box đỏ gộp nhiều xe ở làn trái.
  Vòng 1: TP 23 / FP 15 / **FN 0**. Các xe nối đuôi ở làn trái được tách thành box riêng, đúng với
  luật "hai xe sát nhau vẽ hai box" mà tôi áp dụng khi xoá box gộp ở frame_0099 và frame_0312.
- **Xấu hơn về FP, cùng frame_0350**: 15 FP, phần lớn là box đỏ nhỏ ở cụm xe xa gần chân trời (góc
  trên trái). Có thể kiểm: tôi đã thêm nhiều xe xa cao 14–20 px khi sửa nhãn (ví dụ 6/14 box thêm ở
  frame_0392), còn bộ tham chiếu do model tạo lại yếu nhất ở xe nhỏ. Nhiều FP này có thể là xe thật
  mà tham chiếu thiếu, cần người rà trước khi kết luận model sai.

### Phân biệt quan sát độc lập, lỗi pre-label và kết quả model

- **Quan sát độc lập** (`BLIND_SCAN.md`, khoá trước khi xem pre-label, frame_0099): tôi đếm 26 xe
  và dự đoán hai chỗ AI dễ sai: xe bị che/cắt ở góc khung hình, và xe nhỏ ở xa nối đuôi nhau bị gộp
  thành một.
- **Lỗi pre-label đã sửa** (`REVIEW_LOG.csv`, `round1_diff.md`): ở frame_0099 AI đề xuất 13 box,
  bản cuối 21 box (thêm 9, sửa 3, xoá 1). Dự đoán thứ hai trong bản quét đúng: box xoá duy nhất là
  một box ôm hai xe nối đuôi phía xa bên trái, được thay bằng hai box. Bản quét **không** đoán được
  lỗi lớn nhất: AI bỏ sót chiếc SUV sát mép dưới dù xe rất rõ. Bản cuối có 21 box, ít hơn 26 xe tôi
  đếm; phần chênh là các xe ở rất xa chỉ còn chấm đèn mà tôi quyết định không vẽ box.
- **Kết quả model sau train** (`metrics_round1.json`): đánh giá trên 20 ảnh test khác thời điểm,
  không phải trên 12 ảnh đã sửa. Nhãn tốt hơn không tự động cho model tốt hơn: cùng bộ 290 box, lần
  chạy mặc định cho AP50 0.380 còn lần chạy chỉnh tham số cho 0.925.

### Một ca khó theo guideline

**Xe nhoè chuyển động bị cắt ở mép phải, frame_0331.** AI vẽ một box hẹp chỉ ôm một dải sáng của
xe. Guideline có hai luật liên quan: "xe bị nhoè vẫn gán, box ôm vùng nhoè của thân xe" và "xe bị
cắt ở mép ảnh chỉ vẽ phần nằm trong ảnh". Tôi xoá box hẹp và vẽ lại box ôm toàn bộ vùng nhoè của
thân xe nhưng dừng ở mép ảnh. Cách xử lý này tôi áp dụng cho mọi xe nhoè ở mép trong lô (ví dụ xe
ở mép dưới frame_0099 và frame_0331) để model không học hai kiểu box khác nhau cho cùng một tình huống.

## 5. Kết luận và giới hạn

**So với cold start**: AP50 tăng từ 0.771 lên 0.925 và recall từ 0.489 lên 0.950 chỉ với 12 ảnh
đã sửa, nhưng precision giảm từ 0.925 xuống 0.646. Kết quả này chỉ đạt được sau khi sửa cấu hình
huấn luyện; với cấu hình mặc định, cùng bộ nhãn làm model kém đi (AP50 0.380).

**Quyết định**: tôi **dừng ở vòng 1** trong buổi lab này, và đề xuất trước khi gán thêm vòng 2 cần
**rà lại FP trên tập test**. Lý do: 210 FP hiện là vấn đề lớn nhất, nhưng tôi chưa biết bao nhiêu
trong số đó là lỗi thật của model và bao nhiêu là xe thật mà tham chiếu thiếu. Nếu phần lớn là xe
thật, gán thêm ảnh sẽ không làm precision tăng mà chỉ làm lệch thêm so với tham chiếu.

**Hai ca còn yếu hoặc bất định cho vòng sau:**

1. **Cụm xe nhỏ ở xa gần chân trời** (góc trên trái, frame_0050 và frame_0350 trên
   `compare_round1.jpg`): nhiều box đỏ FP. Cần thống nhất luật cho xe cao 14–20 px (sát ngưỡng bỏ
   qua 16 px) giữa nhãn của tôi và nhãn tham chiếu. Chi phí rà cao: vòng 1 tôi mất công thêm
   trung bình ~11 box/ảnh (137 box added / 12 ảnh), và các ảnh `selection_round2.csv` xếp đầu có 50–87
   box dự đoán (conf ≥ 0.05) mỗi ảnh, nhiều hơn mức 28–47 box của lô vòng 1.
2. **Xe nhoè hoặc bị cắt ở mép ảnh, và xe đèn hậu đỏ nối đuôi nhau** (frame_0250 mép phải, frame_0150
   làn phải): model vẽ box chồng hoặc lớn hơn tham chiếu, bị tính FP dù có thể trúng xe.

**Ảnh gần trùng ở vòng 2**: `selection_round2.csv` chọn frame_0324 (D = 0.08, cách frame_0326 đã gán
0.8 s) và frame_0372 (D = 0.12, cách frame_0369 1.2 s). Hai ảnh này có điểm cao nhờ U và A nhưng gần
như trùng ảnh đã gán; nếu làm vòng 2 tôi sẽ thay bằng các ảnh 0–39 s (ví dụ frame_0023, frame_0072
đã có trong lô) hoặc tăng `W_D`.

**Giới hạn ảnh hưởng tới kết luận:**

- **Test chỉ 20 ảnh, 403 box** từ 4 đoạn video: +0.154 AP50 là lớn, nhưng một cảnh giao thông khác
  (mưa, ban ngày, camera khác) có thể cho kết quả rất khác.
- **Nhãn tham chiếu do model tạo, chưa có người rà**: precision 0.646 có thể thấp hơn thực tế nếu
  tham chiếu bỏ sót xe xa; AP50 đo mức khớp với một model khác chứ không phải chân lý.
- **Luật bỏ qua box dưới 16 px**: 14 box tham chiếu không được tính; xe xa nhỏ hơn ngưỡng này không
  ảnh hưởng số đo, nên số đo không phản ánh đúng khả năng phát hiện xe rất xa.
- **Không có tập validation riêng** (`val = False`), và tôi đổi cấu hình sau khi đã thấy kết quả
  test của lần chạy đầu. Tôi chỉ đổi một lần dựa trên lập luận về số bước huấn luyện, nhưng quyết
  định này vẫn được gợi ý bởi số đo test, nên mức +0.154 có thể hơi lạc quan.
- **Không có đối chứng chọn ngẫu nhiên** nên chưa tách được phần cải thiện do cách chọn mẫu và
  phần do có thêm 290 box ảnh đêm bất kỳ.

**Nếu AP50 giảm, tôi sẽ kiểm tra theo thứ tự**: (1) cấu hình train: số bước cập nhật so với warmup,
learning rate, freeze, như đã gặp ở lần chạy đầu; (2) nhãn: class id, định dạng, box gộp hoặc bỏ
sót, cách xử lý không nhất quán giữa các ảnh; (3) ảnh so sánh test: lỗi là FP hay FN, ở nhóm kích
thước nào; (4) nhãn tham chiếu của những ca bị tính sai, trước khi kết luận model kém đi.
