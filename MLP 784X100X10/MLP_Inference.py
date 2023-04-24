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
from tensorflow.keras import layers
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Softmax
import matplotlib.pyplot as plt
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def custom_loss_function(y_true, y_pred):
    batch_mean, batch_var = tf.nn.moments(y_pred, [0])
    y_pred = tf.nn.batch_normalization(y_pred,
                batch_mean, batch_var, 0, 1, 0.001)
    y_pred = tf.keras.activations.softmax(10*y_pred)
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(y_true, y_pred)
    return loss

def build_Custommodel():
    model = Sequential()

    #Define 784x100x10 network
    model.add(HidLayerMP(100,22,input_shape=(784,)))
    model.add(HidLayerMP(10,10))

    #define optimizer
    sgd = tf.keras.optimizers.Adam(learning_rate=0.001)
    sgd = tf.keras.mixed_precision.LossScaleOptimizer(sgd)
    model.compile(loss=custom_loss_function, optimizer=sgd, metrics=['accuracy'])

    print(model.summary())
    return model

epochs = 30
initial_learning_rate = 0.001
decay = initial_learning_rate / epochs
def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)

from tensorflow.keras.callbacks import LearningRateScheduler

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train.reshape((x_train.shape[0], 784))
    x_test = x_test.reshape((x_test.shape[0], 784))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    
    # build network
    model = build_Custommodel()
    print(model.summary())

    filename='Plots/temp.csv'
    history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
 
    batch_size = 128
    
    #load weights of saved model
    model.load_weights('Models/model_MLP_100hid.hdf5')
    
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
