#import library
import os
import numpy as np
np.warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from google_drive_downloader import GoogleDriveDownloader as gdd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib.eager.python import tfe
tf.enable_eager_execution()
tf.set_random_seed(0)
np.random.seed(0)

#import data
gdd.download_file_from_google_drive(file_id='1ABhwNb5ioRzUEV9iLDpVSEIV_76yofNm', dest_path='./zaloai_landmark_20k.npz', unzip=False)

data = np.load("zaloai_landmark_20k.npz")
X, y = data['X'], data['y']

num_classes = len(np.unique(y))
y_ohe = tf.keras.utils.to_categorical(y, num_classes=num_classes, dtype='int')
#chia du lieu de huan luyen
x_train, x_test, y_train_ohe, y_test_ohe = train_test_split(X, y_ohe, test_size=0.25)
print("Train size: {} - Test size: {}".format(x_train.shape, x_test.shape))

#xay dung mo hinh
class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel, strides, padding):
        super(ConvBlock, self).__init__()
        ## TODO 1 ##
        
        ## END TODO 1 ##
        self.cnn = tf.keras.layers.Conv2D(filters, (kernel, kernel), strides=(strides, strides), kernel_initializer='he_normal', padding=padding)
        self.pool = tf.keras.layers.MaxPool2D((2,2), strides=(2,2))
        self.bn = tf.keras.layers.BatchNormalization()
        
    def call(self, inputs, training=None, mask=None):
        ## TODO 2 ##
        
        ## END TODO 2 ##
        x = self.cnn(inputs)
        x = self.bn(x)
        x = tf.nn.relu(x)   
        x = self.pool(x)

        return x
    
class CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        ## TODO 3 ##
        ## END TODO 3 ##
        self.block1 = ConvBlock(64, kernel=3, strides=1, padding='same')
        self.block2 = ConvBlock(128, kernel=3, strides=1, padding='same')
        self.block3 = ConvBlock(256, kernel=3, strides=1, padding='same')
        self.block4 = ConvBlock(512, kernel=3, strides=1, padding='same')
        self.block5 = ConvBlock(512, kernel=3, strides=1, padding='same')
        self.block6 = ConvBlock(1024, kernel=3, strides=1, padding='same')
        self.flatten = tf.layers.Flatten()
        
        ## TODO 4 ##
        ## END TODO 4 ##
        self.dense2 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        
        ## TODO 5 ##
        ## END TODO 5 ##
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.flatten(x)
        
        ## TODO 6 ##
        ## END TODO 6 ##   
        x = self.dense2(x)
        
        # softmax op does not exist on the gpu, so always use cpu
        with tf.device('/cpu:0'):
            output = tf.nn.softmax(x)

        return output
# training
device = '/cpu:0' if tfe.num_gpus() == 0 else '/gpu:0'
batch_size = 32
epochs = 16

with tf.device(device):
    # build model and optimizer
    model = CNN(num_classes)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train
    model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test_ohe), verbose=1)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)

    model.save_weights('./check_points/my_model')
# du doan mo hinh    
model = CNN(num_classes)
model.load_weights('./check_points/my_model')
print("Model đã được load")

x_new = x_test[0]
pred = model.predict(x_new[None, :])
pred_label = np.argmax(pred)
plt.imshow(x_new)
print("Mô hình dự đoán nhãn của bức ảnh: {}".format(pred_label))