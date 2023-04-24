import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.datasets import mnist
import numpy as np
import scipy.io as sio
from keras import backend as K
from keras.layers import Layer
from keras.utils import conv_utils
from keras import initializers
from keras import activations
import tensorflow as tf
from keras.utils import np_utils
from Layers_Train import HidLayerMP
from Layers_Train import customConvMP
from tensorflow.keras import layers
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization, Softmax
import matplotlib.pyplot as plt
from keras.layers import GaussianNoise
import os 

os.environ["CUDA_VISIBLE_DEVICES"]="1"

#configuring GPUS
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)

#Building model
def build_Custommodel():
    model = Sequential()
    #3x3 conv layer with 16 channels followed by maxpool and bn
    model.add(customConvMP(16,3,8,kernel_initializer='he_normal', activation = 'relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    #3x3 conv layer with 32 channels followed by maxpool and bn
    model.add(customConvMP(32,3,8,kernel_initializer='he_normal', activation = 'relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())


    model.add(Flatten())
    #500 nodes FC layer followed by BN
    model.add(HidLayerMP(500,11))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    #10 nodes FC output layer followed by BN
    model.add(HidLayerMP(10,50))
    model.add(BatchNormalization())
    #model.add(Softmax())

    #defining optimizers and loss function
    sgd = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print(model.summary())
    return model

#defing time based lr decay
epochs = 50
initial_learning_rate = 0.001
decay = initial_learning_rate / epochs

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)

from tensorflow.keras.callbacks import LearningRateScheduler


if __name__ == '__main__':
    #pre-processing of MNIST Dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    
    # build network
    model = build_Custommodel()
    print(model.summary())
    batch_size = 16
    
    #initialize model
    model.load_weights('Models/Batch16_OldW_CNN16and32_CNNHidDO0p1_Hid500.hdf5')
    
    #Evaluate model
    _, acc = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc))

    out = model.predict(x_test)
    out = np.argmax(out,axis=1)


    from sklearn.metrics import accuracy_score
    print(accuracy_score(np.argmax(y_test,axis=1), out))
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(np.argmax(y_test,axis=1), out))
