# USAGE
# python main.py --model handwrite.model --image images/image2.png

# Import các thư viện cần thiết
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2

# Xây dựng trình phân tích cú pháp và phân tích các đối số
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained handwriting recognition model")
args = vars(ap.parse_args())

# Nạp mô hình OCR chữ viết tay
print("[INFO] loading handwriting OCR model...")
model = load_model(args["model"])

# Đọc hình ảnh đưa vào, resize ảnh, chuyển đổi thành ảnh xám và giảm nhiễu làm mờ
image = cv2.imread(args["image"])
resized = cv2.resize(image,(700, 734))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Thực hiện phát hiện biên, tìm các đường viền trong ảnh 
# sắp xếp các đường viền kết quả từ trái sang phải
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

# Khởi tạo danh sách các hộp chứa đường viền 
# ký tự tương ứng sẽ nhận dạng OCR
chars = []

# Lặp qua các coutour
for c in cnts:
	# tính bounding box của contour
	(x, y, w, h) = cv2.boundingRect(c)

    # Lọc ra các bouding box, đảm bảo rằng không quá nhỏ hoặc quá lớn
	if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
        # Trích xuất ký tự và ngưỡng để làm cho ký tự trắng trên nền đen
        # sau đó lấy chiều rộng và chiều cao của hình ảnh đã ngưỡng
		roi = gray[y:y + h, x:x + w]
		thresh = cv2.threshold(roi, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		(tH, tW) = thresh.shape

		# Nếu chiều rộng lớn hơn chiều cao, thay đổi kích thước theo chiều rộng
		if tW > tH:
			thresh = imutils.resize(thresh, width=32)

		# ngược lại, thay đổi kích thước theo chiều cao
		else:
			thresh = imutils.resize(thresh, height=32)

		# Lấy lại kích thước hình ảnh (sau khi đã thay đổi kích thước)
        # sau đó xác định cần phải thêm vào chiều rộng và chiều cao
        # để đảm bảo rằng hình ảnh có kích thước 32x32
		(tH, tW) = thresh.shape
		dX = int(max(0, 32 - tW) / 2.0)
		dY = int(max(0, 32 - tH) / 2.0)

		# thêm pad vào đảm bảo rằng kích thước là 32x32
		padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
			left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
			value=(0, 0, 0))
		padded = cv2.resize(padded, (32, 32))

		#  Chuẩn bị hình ảnh đã thêm pad để phân loại qua mô hình OCR chữ viết tay
		padded = padded.astype("float32") / 255.0
		padded = np.expand_dims(padded, axis=-1)

		# Cập nhật danh sách các ký tự sẽ được nhận dạng OCR
		chars.append((padded, (x, y, w, h)))

# Trích xuất vị trí boudingbox và ký tự đã thêm pad
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")

# Nhận dạng ký tự bằng cách sử dụng mô hình nhận dạng chữ viết tay
preds = model.predict(chars)

# Xác định danh sách tên nhãn
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# vòng lặp cho các dự đoán và vị trí của boudingbox 
for (pred, (x, y, w, h)) in zip(preds, boxes):
	# tìm index của nhãn có xác suất tương ứng lớn nhất
    # trích xuất xác suất và nhãn
	i = np.argmax(pred)
	prob = pred[i]
	label = labelNames[i]

	# Vẽ kết quả nhận dạng lên hình ảnh
	print("[INFO] {} - {:.2f}%".format(label, prob * 100))
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.putText(image, label, (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

	# Hiển thị hình ảnh 
	cv2.imshow("Image", image)
	cv2.waitKey(0)