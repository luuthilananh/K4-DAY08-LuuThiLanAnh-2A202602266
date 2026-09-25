# Vòng 1 — lần chạy đối chứng (cấu hình mặc định của notebook)

Các file trong thư mục này là bản sao **nguyên vẹn** của kết quả Colab lần chạy vòng 1 đầu tiên,
chưa chỉnh sửa. Chúng được giữ lại để so sánh với lần chạy sau khi đổi tham số huấn luyện.
Bản ZIP đầy đủ nằm ngoài repo: `day8_round1_out_baseline_e50_b16.zip`.

| File | Nội dung |
| --- | --- |
| `metrics_round1.json` | Số đo trên 20 ảnh test sau fine-tune (tạo lúc 2026-09-25T07:19:28, Tesla T4) |
| `compare_round1.jpg` | Ảnh so sánh nhãn tham chiếu / cold start / vòng 1 trên 4 frame test |
| `rounds_table.md` | Bảng vòng 0 và vòng 1 của lần chạy này |

## Cấu hình huấn luyện

| Tham số | Giá trị |
| --- | --- |
| Model gốc | `yolov8n.pt` |
| Dữ liệu train | 12 ảnh, 290 box (`labels/round1/`) |
| `EPOCHS` | 50 |
| `batch` | 16 |
| `freeze` | 0 (không đóng băng lớp nào) |
| `imgsz` | 960 |
| `seed` | 8 |

## Số đo chính (từ `metrics_round1.json`)

| | Cold start (vòng 0) | Vòng 1 lần chạy này |
| --- | ---: | ---: |
| AP50 | 0.771 | 0.380 |
| Precision @0.25 | 0.925 | 1.000 |
| Recall @0.25 | 0.489 | 0.109 |
| Recall xe nhỏ / trung bình / lớn | 0.182 / 0.547 / 0.561 | 0.000 / 0.088 / 0.439 |

## Nhận định về nguyên nhân (giả thuyết, chưa kiểm chứng)

- 12 ảnh với batch 16 nghĩa là mỗi epoch chỉ có 1 bước cập nhật, nên 50 epoch chỉ là 50 bước.
  Ultralytics mặc định dành ít nhất 100 bước đầu để warmup, nên toàn bộ quá trình huấn luyện
  vẫn nằm trong giai đoạn learning rate đang tăng.
- Khi train với 1 lớp `car`, đầu phân lớp được khởi tạo lại từ đầu, nên kiến thức lớp xe đã học
  từ COCO bị mất.
- Model sau train rất thận trọng: không có box sai (FP = 0) nhưng bỏ sót 359/403 xe tham chiếu.

Lần chạy tiếp theo đổi `EPOCHS = 100`, `TRAIN_BATCH = 4` và `FREEZE = 10`. Cấu hình này được chọn
từ lập luận trên, không phải từ việc thử nhiều lần trên tập test.
