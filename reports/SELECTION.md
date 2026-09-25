# Vì sao chọn lô này?

Nguồn số liệu: `outputs/selection_round1.csv` (điểm do mô hình cold start `yolov8n` COCO tính trên
268 ảnh pool) và `outputs/selection_round1.jpg`. Ở vòng 1 chưa có ảnh nào được gán nên `D = 1.0`
với mọi frame; thứ hạng chỉ do `U` (độ bất định) và `A` (số box mơ hồ 0.15 ≤ conf < 0.50) quyết định.
Không frame nào trong top 50 bị đánh dấu `empty` (model luôn thấy ít nhất vài chục box).

## Top 5 nếu chỉ có ngân sách rà năm ảnh

Năm frame đứng đầu theo điểm (0182, 0369, 0380, 0326, 0331) thì có bốn frame nằm trong đoạn
130–152 s của video. Camera cố định và dòng xe ở các giây liền nhau gần giống nhau, nên nếu chỉ có
năm ảnh, tôi ưu tiên **trải đều theo thời gian** hơn là lấy đúng năm điểm cao nhất:

| Thứ tự | Frame | Hạng CSV | Score | U | A | t (s) | Lý do |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | frame_0182.jpg | 1 | 0.959 | 0.918 | 1.000 | 72.8 | Điểm cao nhất; 18/28 box mơ hồ, nhiều nhất pool (cùng 0331, 0312) |
| 2 | frame_0369.jpg | 2 | 0.932 | 0.932 | 0.889 | 147.6 | Đại diện đoạn cuối video (140–157 s), 43 box dự đoán |
| 3 | frame_0326.jpg | 4 | 0.916 | 0.931 | 0.833 | 130.4 | Đại diện đoạn 124–133 s; chọn thay cho 0331 vì 0331 chỉ cách 2.0 s |
| 4 | frame_0099.jpg | 8 | 0.906 | 0.946 | 0.778 | 39.6 | Đoạn đầu video; U cao nhất trong top 10 |
| 5 | frame_0227.jpg | 11 | 0.892 | 0.916 | 0.778 | 90.8 | Lấp khoảng 75–124 s chưa có ảnh nào trong top 5 |

Quyết định về ảnh gần trùng: **bỏ frame_0380** (hạng 3, score 0.917) vì chỉ cách 0369 4.4 s và
cùng một đoạn dòng xe; **bỏ frame_0331** (hạng 5) vì cách 0326 đúng 2.0 s và có 47 box dự đoán
(nhiều nhất top 5, chi phí rà cao nhất). Với ngân sách năm ảnh, mỗi ảnh gần trùng là một ảnh mất đi
cho một đoạn thời gian khác.

## Ba frame thuộc lô 12 ảnh model chọn và bằng chứng

- **frame_0182.jpg** (hạng 1, score 0.959, U 0.918, A 1.000, 28 box, 18 box mơ hồ). A = 1.0 nghĩa
  là số box mơ hồ bằng mức cao nhất pool. Khi rà thực tế, AI đề xuất 13 box (conf ≥ 0.25) nhưng bản
  cuối có 24 box: tôi thêm 12 xe và chỉnh 5 box (`outputs/round1_diff.md`). Điểm bất định cao ở đây
  trùng với việc model thực sự bỏ sót nhiều xe.
- **frame_0331.jpg** (hạng 5, score 0.915, U 0.831, A 1.000, 47 box). U thấp hơn 0182 nhưng A tối
  đa nên vẫn vào top 5. Đây là ảnh có nhiều box AI sai nhất lô: xoá 5 box (box gộp hai xe, box hẹp
  trên xe nhoè ở mép phải) và thêm 11 box.
- **frame_0392.jpg** (hạng 15, score 0.887, U 0.975, A 0.667, 35 box). U cao nhất trong top 50
  (các box khó nhất có conf gần 0.5) nhưng ít box mơ hồ nên hạng thấp hơn. Bản cuối thêm 14 box,
  nhiều nhất lô: 6/14 box thêm là xe ở xa cao 14–20 px, hai xe bị cắt ở mép dưới, còn lại là xe cỡ vừa.

Trên contact sheet `selection_round1.jpg`, 12 ảnh trải từ giây 39.6 đến 156.8 nhưng dồn thành cụm
(39–43 s, 72–75 s, 91 s, 108 s, 124–132 s, 147–157 s); không có ảnh nào ở 0–39 s.

## Một frame điểm cao nhưng không chọn, và một frame điểm thấp vẫn nên xem

- **frame_0372.jpg** (hạng 6, score 0.910, U 0.920) có điểm cao hơn 7 ảnh được chọn nhưng bị
  `MIN_GAP_S = 2.0` loại vì chỉ cách 0369 1.2 s. Tương tự 0368 (hạng 9, cách 0369 0.4 s), 0330
  (hạng 12, cách 0331 0.4 s) và 0271 (hạng 16, cách 0270 0.4 s). Luật khoảng cách đã làm đúng việc:
  hai ảnh cách 0.4 s gần như là một ảnh, gán cả hai tốn gấp đôi công mà model học thêm rất ít.
- **frame_0002.jpg** (hạng 18, score 0.866, t = 0.8 s) và **frame_0020.jpg** (hạng 29, t = 8.0 s)
  nên được xem dù điểm thấp hơn: lô 12 ảnh không có ảnh nào trong 39 giây đầu video. Ở vòng 1,
  `D = 1.0` cho mọi frame nên thành phần đa dạng không có tác dụng; chỉ `MIN_GAP_S` ngăn trùng lặp
  cục bộ chứ không đảm bảo phủ toàn video. Vòng 2 xác nhận điều này: `selection_round2.csv` xếp
  frame_0030 (t = 12.0 s) hạng 1.

## Điều phép chọn này chưa chứng minh về chất lượng mô hình

- Điểm bất định chỉ đo **model phân vân** (conf gần 0.5), không đo model **sai**. Một xe bị bỏ sót
  hoàn toàn (không có box nào) không làm tăng U. Ví dụ ở frame_0099 không có box gợi ý (conf ≥ 0.25)
  nào cho SUV lớn ở mép dưới, và ở frame_0312 bỏ sót cả xe tải trắng; điểm cao của hai ảnh này đến từ các xe khác.
- Điểm cao không chứng minh ảnh đó sẽ giúp AP50 tăng. Mức tăng sau fine-tune (xem `REPORT.md`)
  phụ thuộc cả vào cách huấn luyện: cùng lô ảnh này, lần chạy mặc định (50 epoch, batch 16) làm
  AP50 giảm còn 0.380, lần chạy chỉnh tham số làm AP50 tăng lên 0.925.
- Không có đối chứng `STRATEGY = "random"` nên chưa biết chọn theo độ bất định có hơn chọn ngẫu
  nhiên với cùng 12 ảnh hay không.
