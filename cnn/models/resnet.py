from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Conv2D, Activation, Dense, Flatten, Input, AveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf

class ResNet:
    @staticmethod
    def residual_block(data, K, stride, chan_dim, red,reg=0.0001, bn_eps=2e-5, bn_momentum=0.9, use_dropout=False):
        # Phần shortcut của mô-đun ResNet
        shortcut = data

        # Phần đầu tiên của mô-đun ResNet: 1x1 CONV
        bn1 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_momentum)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K*0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # Phần thứ hai của mô-đun ResNet: 3x3 CONV
        bn2 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_momentum)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K*0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act2)

        # Phần thứ ba của mô-đun ResNet: 1x1 CONV
        bn3 = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_momentum)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # Nếu cần giảm kích thước không gian, áp dụng một lớp CONV cho shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)
        # Cộng shortcut và lớp CONV cuối cùng
        x = add([conv3, shortcut])

        # Thêm lớp Dropout kiểm soát overfitting
        if use_dropout:
            x = Dropout(0.5)(x)
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bn_eps=2e-5, bn_momentum=0.9):
        input_shape = (height, width, depth)
        chan_dim = -1

        # Kiểm tra định dạng dữ liệu hình ảnh (channels last hoặc channels first)
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1

        # Tạo đầu vào cho mạng
        inputs = Input(shape=input_shape)

        # Áp dụng BatchNormalization sau đó Conv2D
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_momentum)(inputs)
        x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)

        # Lặp qua số lớp residual (các stages)
        for i in range(0, len(stages)):
            # Khởi tạo bước đi (stride), sau đó áp dụng mô-đun residual để giảm kích thước không gian của đầu vào
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_block(x, filters[i + 1], stride, chan_dim, reg, bn_eps, bn_momentum)

            # Lặp qua số lớp trong mỗi stage
            for _ in range(0, stages[i] - 1):
                # Áp dụng mô-đun residual
                x = ResNet.residual_block(x, filters[i + 1], (1, 1), chan_dim, reg, bn_eps, bn_momentum)

        # Áp dụng BatchNormalization => Activation (ReLU) => AveragePooling2D
        x = BatchNormalization(axis=chan_dim, epsilon=bn_eps, momentum=bn_momentum)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # Flatten đầu ra
        x = Flatten()(x)

        # Thêm lớp fully connected (Dense) với kernel_regularizer
        x = Dense(classes, kernel_regularizer=l2(reg))(x)

        # Áp dụng hàm kích hoạt softmax
        x = Activation("softmax")(x)

        # Tạo mô hình hoàn chỉnh
        model = Model(inputs, x, name="resnet")
        return model

class CNN:
    @staticmethod
    def cnn_model(input_shape, num_classes):
        # Tạo một mô hình Sequential
        model = keras.Sequential()

        # Lớp convolutional 1
        model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=input_shape))
        model.add(layers.Dropout(0.25))
        model.add(layers.MaxPooling2D((2, 2)))
        # Lớp convolutional 2
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.25))
        model.add(layers.MaxPooling2D((2, 2)))

        # Lớp convolutional 3
        model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.25))
        model.add(layers.MaxPooling2D((2, 2)))

        # Dàn phẳng các đặc trưng
        model.add(layers.Flatten())

        # Thêm một lớp kết nối đầy đủ (fully connected layer) với lớp dropout
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(layers.Dropout(0.5))

        # Lớp đầu ra với số lớp tương ứng với số lớp phân loại
        model.add(layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01)))
        return model
