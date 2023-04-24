import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,Dropout
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
import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Softmax
import matplotlib.pyplot as plt
import os 

os.environ["CUDA_VISIBLE_DEVICES"]="1"

#defining custom loss function
def custom_loss_function(y_true, y_pred):
    batch_mean, batch_var = tf.nn.moments(y_pred, [0])
    y_pred = tf.keras.activations.softmax(y_pred)
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(y_true, y_pred)
    return loss

#build model
def build_Custommodel():
    model = Sequential()
    #conv layer with 3x3 kernels and 6 channels followed by maxpooling
    model.add(customConvMP(6,3,8,kernel_initializer='he_normal', activation = 'relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    #FC layer with 15 and 10 hidden nodes
    model.add(HidLayerMP(15,22))
    model.add(HidLayerMP(10,1))
    #BN layer at the output node
    model.add(BatchNormalization())

    #defining optimizer
    sgd = tf.keras.optimizers.Adam(learning_rate=0.0005)
    sgd = tf.keras.mixed_precision.LossScaleOptimizer(sgd)
    model.compile(loss=custom_loss_function, optimizer=sgd, metrics=['accuracy'])

    print(model.summary())
    return model

#initializing LR scheduler
epochs = 30
initial_learning_rate = 0.00005
decay = initial_learning_rate / epochs

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)

from tensorflow.keras.callbacks import LearningRateScheduler

if __name__ == '__main__':
    #Pre-processing MNIST dataset
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
    #filename='Analysis/Batch32_Hid15_oldW_oneBN.hdf5'
    #history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
    batch_size = 256

    #loading trained model
    model.load_weights('Models/model_Batch32_Hid15_oldW_oneBN.hdf5') 
   
    
    #evaluate model
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
