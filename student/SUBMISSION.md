# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Đỗ Mạnh Nghĩa
- MSSV: 2A202602971
- Email: domanhnghiaforwork@gmail.com
- Link repo (fork): https://github.com/domanhnghiaforwork-cloud/K4-L2L3-DAY23-DoManhNghia-2A202602971-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): (xem output từ `git rev-parse HEAD` sau khi commit CP6)

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: compare, [0, 198], training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9701, 0.7004, 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1503, 502, 11.3436, 0, 239, 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1359, 502, 9.2668, 0, 239, 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  Ở cả hai chế độ `lidar` và `fused`, số cặp ghép confirmed track với ground truth là bằng nhau tuyệt đối (`matches = 502`), không có bất kỳ track ảo nào sinh ra (`ghost_track_frames = 0`), và số lượng nhãn xe chưa bám được là đồng nhất (`missed_gt_frames = 239`, `mean_confirmed_tracks = 2.5226`). Điều này hoàn toàn nhất quán với thiết kế kiến trúc track-then-fuse: camera chỉ tinh chỉnh trạng thái EKF mà không tham gia tạo, xoá hay quyết định xác nhận track.
  Về mặt độ chính xác vị trí 3D, chế độ `fused` đạt RMSE thấp hơn rõ rệt so với `lidar` (0.1359 m so với 0.1503 m, giảm sai số 0.0144 m, tương ứng tổng bình phương sai số `sum_sq_err` giảm từ 11.3436 m² xuống 9.2668 m²). Việc kết hợp thêm thông tin góc đo 2D từ camera FRONT thông qua mô hình pinhole và cập nhật EKF với Kalman Gain tối ưu đã giúp tinh chỉnh và làm mượt thêm ước lượng toạ độ của xe, chứng minh camera cải thiện tracking mà không làm xuất hiện ghost track hay bỏ sót detection.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - Đo LiDAR: Vector đo z gồm 3 toạ độ không gian [x, y, z]^T (mét) trong hệ toạ độ cảm biến; hàm đo là tuyến tính với ma trận H = [I_3, 0_3x3] (cỡ 3x6); ma trận hiệp phương sai nhiễu đo R = diag(sigma_x^2, sigma_y^2, sigma_z^2) mang đơn vị mét vuông (mặc định sigma = 0.1 m).
   - Đo Camera: Vector đo z gồm 2 toạ độ điểm ảnh [u, v]^T (pixels) trên mặt phẳng ảnh; hàm đo h(x) là phi tuyến theo mô hình chiếu pinhole (u = c_i - f_i * y_s / x_s, v = c_j - f_j * z_s / x_s sau khi đổi qua hệ toạ độ camera p_s = R*p + t), ma trận H là Jacobian của phép chiếu kết hợp xoay extrinsic (cỡ 2x6); ma trận hiệp phương sai nhiễu đo R = diag(sigma_cam_i^2, sigma_cam_j^2) mang đơn vị pixel bình phương (mặc định sigma = 5.0 pixels).

2. Vì sao cần gating Mahalanobis trước khi gán?
   - Khoảng cách Mahalanobis d^2 = gamma^T * S^-1 * gamma tính toán khoảng cách sai lệch giữa phép đo và dự báo có chuẩn hoá theo ma trận hiệp phương sai bất định tổng hợp S = H*P*H^T + R. Khác với khoảng cách Euclid thông thường bỏ qua phân phối sai số, Mahalanobis xét đến hình elip bất định của từng track.
   - Cổng chi-bình-phương loại bỏ các phép đo nằm ngoài vùng phân phối thống kê hợp lý (d^2 > chi2.ppf(0.995, dim)), giúp loại bỏ nhiễu/ngoại lai (clutter), giảm tải chi phí tính toán cho thuật toán gán Greedy, và ngăn ngừa việc cập nhật sai lệch làm hỏng ước lượng trạng thái EKF.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - Pipeline là track-then-fuse: Hệ thống duy trì một danh sách track duy nhất; mỗi frame dự báo EKF một lần, sau đó lần lượt cập nhật EKF theo từng modality (AssocL với LiDAR, tiếp theo là AssocC với Camera). Hệ thống không gộp dữ liệu thô hay detection trước khi tracking (fuse-then-track).
   - Minh chứng trên log `grade_run.log` và code:
     + Ở cùng một frame, số lượng confirmed tracks, matches, ghosts và misses hoàn toàn giống nhau giữa hai mode (ví dụ frame 4 ở cả hai mode đều có confirmed: 2, matches: 2, ghosts: 0, misses: 0). Camera không làm thay đổi số lượng track.
     + Trong code `run_lab.py`: Trong một frame, hàm gọi KF.predict() một lần, sau đó gọi associate_and_update(..., lidar) rồi mới gọi associate_and_update(..., camera) trên cùng danh sách manager.track_list.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Nếu calibration camera bị lệch (extrinsic hoặc intrinsic sai lệch so với thực tế), giá trị hình chiếu dự báo h(x) sẽ bị lệch khỏi vị trí quan sát thực z trên ảnh.
   - Triệu chứng: Innovation gamma = z - h(x) sẽ xuất hiện thành phần sai số hệ thống (bias) lớn, không còn phân phối chuẩn kỳ vọng 0. Khoảng cách Mahalanobis d^2 = gamma^T * S^-1 * gamma tăng cao vượt qua ngưỡng cổng Chi-bình-phương, khiến các phép đo camera bị cổng gating loại bỏ (coi là outlier). Kết quả là camera không thể ghép cặp với track, không thực hiện được EKF update, hoặc nếu chưa vượt cổng sẽ kéo lệch ước lượng vị trí track.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   - Cần sensor tường minh ở frame rỗng: Ngay cả khi meas_list rỗng (không có detection nào trong frame), hàm vẫn phải gọi manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor). Module quản lý vòng đời cần biết rõ sensor của pass hiện tại là "lidar" hay "camera" để trừ điểm (miss trong FOV) nếu là LiDAR, hoặc bỏ qua nếu là Camera. Nếu không truyền sensor tường minh, hệ thống không thể xử lý chính xác vòng đời ở frame rỗng.
   - LiDAR quyết định score/init/delete: LiDAR quét toàn cảnh 360 độ và cung cấp toạ độ không gian 3D trực tiếp, độ tin cậy hình học cao, đủ thông tin để khởi tạo vật thể 3D mới và kiểm soát sự tồn tại.
   - Camera chỉ EKF update: Camera trong bài chỉ quan sát góc hẹp phía trước (FRONT) và đo góc 2D trên ảnh (mất chiều sâu x_s). Camera không đủ thông tin 3D để tự tạo track, và không được trừ điểm hay xoá track khi xe đi ra ngoài tầm nhìn camera (nhưng vẫn còn trong tầm LiDAR), do đó chỉ dùng camera để tinh chỉnh trạng thái EKF.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - Điều kiện xác nhận: Track được khởi tạo với score = 1/window (1/6). Mỗi frame có LiDAR hit, score tăng +1/window (tối đa 1.0). Khi score > confirmed_threshold (0.8, tức cần ít nhất 5 hits), track được xác nhận (state = "confirmed").
   - Giữ confirmed sau miss: Khi bị miss trong FOV LiDAR, score bị trừ -1/window, nhưng trạng thái track vẫn được giữ nguyên là "confirmed" (không bị hạ cấp), giúp duy trì ổn định danh tính đối tượng qua các lần che khuất ngắn.
   - Điều kiện xoá track (toán tử OR):
     1. Phương sai vị trí ngang quá lớn: P[0, 0] > max_P (9.0 m²) hoặc P[1, 1] > max_P.
     2. Track đã confirmed có score < delete_threshold (0.6).
     3. Track chưa confirmed (tentative/initialized) có score <= 0.0.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity IDE (Gemini)
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ phân tích mã nguồn theo các chỉ dẫn TODO trong Part E, G, F, H; tự động hoá chạy pytest và runner tích hợp Waymo `fusion-run-lab`; giải thích chi tiết các câu hỏi kỹ thuật và đối chiếu Rubric chấm điểm.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Đã chạy toàn bộ bộ kiểm thử với lệnh `pytest student/tests -q` (128/128 passed); chạy kiểm thử tích hợp Waymo với lệnh `fusion-run-lab` trên 198 frames đạt RMSE LiDAR 0.1503m, RMSE Fused 0.1359m; chạy công cụ kiểm định nộp bài `python tools/check_submission.py` xác nhận đạt 100%.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
