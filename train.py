# USAGE
# Sử dụng: python train.py --az a_z_handwritten_data.csv --model handwriting.model

# Thiết lập matplotlib để có thể lưu các hình ảnh nền trong quá trình thực hiện
import matplotlib
matplotlib.use("Agg")

# Import các thư viện cần thiết
from cnn.models import ResNet
from cnn.az_dataset import load_mnist_dataset
from cnn.az_dataset import load_az_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# Tạo các đối số dòng lệnh và phân tích các đối số
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True,
    help="Đường dẫn đến tập dữ liệu A-Z")
ap.add_argument("-m", "--model", type=str, required=True,
    help="Đường dẫn đến mô hình nhận diện chữ viết tay đã huấn luyện")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="Đường dẫn đến tệp lưu biểu đồ lịch sử huấn luyện")
args = vars(ap.parse_args())

# Khởi tạo số epoch cho quá trình huấn luyện, tốc độ học ban đầu,
# và kích thước batch
EPOCHS = 50
INIT_LR = 1e-1
BS = 128

# Tải tập dữ liệu A-Z và MNIST, tương ứng
print("[INFO] Đang tải tập dữ liệu...")
(azData, azLabels) = load_az_dataset(args["az"])
(digitsData, digitsLabels) = load_mnist_dataset()

# Tập dữ liệu MNIST sử dụng các nhãn từ 0 đến 9, vì vậy cộng 10 cho
# mỗi nhãn A-Z để đảm bảo các ký tự A-Z không bị nhầm lẫn
# thành các chữ số
azLabels += 10

# Gộp dữ liệu và nhãn từ tập dữ liệu A-Z và tập dữ liệu chứa các chữ số của MNIST
data = np.vstack([azData, digitsData])
labels = np.hstack([azLabels, digitsLabels])

# Mỗi hình ảnh trong tập dữ liệu A-Z và tập dữ liệu chứa chữ số của MNIST có kích thước 28x28 pixel;
# tuy nhiên, kiến trúc mạng mà sử dụng được thiết kế cho các hình ảnh có kích thước 32x32 pixel,
# vì vậy cần phải thay đổi kích thước thành 32x32
data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

# Thêm một chiều kênh vào mỗi hình ảnh trong tập dữ liệu và chia giá trị pixel của các hình ảnh từ [0, 255] xuống [0, 1]
data = np.expand_dims(data, axis=-1)
data /= 255.0

# Chuyển đổi các nhãn từ dạng số nguyên sang dạng vector
le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)

# Điều chỉnh cho độ lệch trong dữ liệu được gán nhãn
classTotals = labels.sum(axis=0)
classWeight = {}

# Duyệt qua tất cả các lớp và tính toán trọng số của lớp
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# Phân chia dữ liệu thành các tập huấn luyện và kiểm tra, sử dụng 80% dữ liệu cho huấn luyện và 20% cho kiểm tra
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.20, stratify=labels, random_state=42)

# Xây dựng trình tạo hình ảnh cho quá trình mở rộng dữ liệu
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest")

# Khởi tạo và biên dịch mạng neural sâu 
print("[INFO] Biên dịch mô hình...")
opt = Adam(lr=INIT_LR)
model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
	(64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Huấn luyện mạng
print("[INFO] Huấn luyện mạng...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1)

# Xác định danh sách các tên nhãn
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# Đánh giá mạng
print("[INFO] Đánh giá mạng...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=labelNames))

# Lưu mô hình 
print("[INFO] Lưu mạng...")
model.save(args["model"], save_format="h5")

# Xây dựng biểu đồ lịch sử huấn luyện và lưu biểu đồ
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Biểu đồ Lỗi và Độ chính xác trong Quá trình Huấn luyện")
plt.xlabel("Epoch #")
plt.ylabel("Lỗi/Độ chính xác")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Khởi tạo danh sách hình ảnh kết quả
images = []

# Ngẫu nhiên chọn một số ký tự trong tập kiểm tra
for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
    # Dự đoán ký tự
    probs = model.predict(testX[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]

    # Trích xuất hình ảnh từ dữ liệu kiểm tra và khởi tạo màu chữ ký tất cả là màu xanh (đúng)
    image = (testX[i] * 255).astype("uint8")
    color = (0, 255, 0)

    # Nếu dự đoán lớp không đúng với nhãn thực tế
    if prediction[0] != np.argmax(testY[i]):
        color = (0, 0, 255)

    # Gộp các kênh màu thành một hình ảnh, thay đổi kích thước hình ảnh từ 32x32
    # thành 96x96 để có thể nhìn rõ hơn và sau đó vẽ nhãn dự đoán lên hình ảnh
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        color, 2)

    # Thêm hình ảnh vào danh sách hình ảnh kết quả
    images.append(image)

# Xây dựng hình ảnh tổng hợp cho các ký tự
montage = build_montages(images, (96, 96), (7, 7))[0]

# Hiển thị hình ảnh tổng hợp kết quả
cv2.imshow("Kết quả OCR", montage)
cv2.waitKey(0)
