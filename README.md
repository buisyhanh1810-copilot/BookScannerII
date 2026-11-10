# Book Scanner (Android)

Chụp/chọn ảnh một trang sách mở (2 trang liền nhau) → app tự tìm viền sách, nắn phẳng,
tìm rãnh gáy, tách thành 2 trang theo thứ tự (trái → phải), xoá bóng, làm trắng nền,
chỉnh nghiêng, lưu JPEG vào `Pictures/BookScanner`. Có nút gộp PDF (`Download/BookScanner`).

## Build bằng GitHub Actions
1. Tạo repo mới trên GitHub, đẩy toàn bộ thư mục này lên nhánh `main`.
2. Vào tab **Actions** → workflow **Build APK** chạy tự động (hoặc bấm *Run workflow*).
3. Chạy xong, mở lần chạy đó → mục **Artifacts** → tải `BookScanner-debug-apk`, giải nén lấy `app-debug.apk` cài lên máy.
4. Muốn có bản Release: `git tag v1.0 && git push --tags`, APK sẽ được đính vào Releases.

Không cần Gradle wrapper: workflow dùng `gradle/actions/setup-gradle` với Gradle 8.7.
Build cục bộ: Android Studio mở thư mục này, hoặc `gradle assembleDebug`.

## Chụp / nhập nhiều trang
- **Chụp liên tục**: chụp xong một ảnh app tự mở lại camera; thoát camera (nút Back) để xử lý cả loạt.
- **Chọn nhiều ảnh**: chọn nhiều ảnh trong thư viện, app xử lý lần lượt.
- Kiểu xuất mặc định là **Đen trắng**.

## Thuật toán tìm viền & gáy (`EdgeDetector.kt`)
Kết hợp 3 đặc trưng thay vì một ngưỡng sáng/tối duy nhất:
1. **Giấy so với nền**: mô hình giấy học từ giữa ảnh (độ sáng + sắc độ Lab), đo bước nhảy "giống giấy -> không giống" ở nhiều mức làm mờ nên bắt được cả cạnh mờ. Bóng gáy không còn làm vùng giấy bị tách đôi.
2. **Vùng chữ**: cạnh trang không được cắt vào chữ (cổng text) và có tiên nghiệm lề; nếu nét chữ lọt ra ngoài thì cạnh được đẩy ra.
3. **Gáy**: bóng gáy + khoảng trống giữa hai khối chữ + vị trí gần giữa. Vết cắt luôn bị kẹp vào khoảng không có chữ, cách chữ mỗi bên một lề an toàn nên không lẫn trang bên kia. Nét chữ mờ gần gáy được bắt bằng ngưỡng yếu hơn.

Mỗi cạnh dò bằng ~28 tia, khớp đường thẳng RANSAC có trọng số tin cậy, đoạn cong/mờ tự bị loại. Lỗi thì quay về thuật toán cũ.
`tools/prototype/` là bản Python để thử thuật toán trên ảnh giả lập (`python3 run.py`, `python3 gut.py`).

## Ghi chú
- minSdk 29 (Android 10+), dùng OpenCV 4.9.0 từ Maven Central.
- Sách xoay dọc trong ảnh: chọn "Luôn tách đôi" (app tự xoay 90°).
- Nền quanh sách có vật sáng dính vào sách làm sai viền: bỏ tick "Tự phát hiện viền" rồi chụp sát sách.
