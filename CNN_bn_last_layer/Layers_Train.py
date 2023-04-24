from tensorflow import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.datasets import mnist
import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.utils import conv_utils
from keras import initializers
from keras import activations
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.python.ops import math_ops

#Defining TEMP-based Fully connected layer
class HidLayerMP(Layer): 
   def __init__(self, output_dim, gamma, **kwargs): 
      self.output_dim = output_dim 
      self.gamma = gamma
      super(HidLayerMP, self).__init__(**kwargs) 
   def build(self, input_shape): 
      self.weight = self.add_weight(name = 'kernelPlus', shape = (input_shape[1], self.output_dim), trainable = True, initializer = tf.keras.initializers.GlorotNormal(),constraint=lambda x: tf.clip_by_value(x, -3, 3)) 
      super(HidLayerMP, self).build(input_shape) # 

   #Defining TEMP Computation
   def spikeMP(self, inMat, gamma):
       inMat = tf.sort(inMat,axis=1)
       shape = tf.shape(inMat)
       batch_num = shape[0]
       in_size = shape[1]
       out_size = shape[2]
       print(batch_num,in_size,out_size)
       cs_t = tf.cumsum(inMat,axis=1)
       cs_t = cs_t+gamma
       d = tf.ones([batch_num,in_size,out_size],dtype=tf.dtypes.float32)
       d = tf.cumsum(d,axis=1)
       cs_t = tf.divide(cs_t,d)

       row_to_add = 999*tf.ones([batch_num,1,out_size],dtype=tf.dtypes.float32)
       arr_1 = tf.concat((row_to_add, cs_t), axis=1)

       row_to_add = tf.zeros([batch_num,1,out_size],dtype=tf.dtypes.float32)
       arr_2 = tf.concat((inMat, row_to_add), axis=1)

       #print(tf.keras.backend.eval(arr_1))
       out = tf.where(arr_1 > arr_2, 999 * tf.ones_like(arr_1), arr_1)
       #print(out.shape)
       out = tf.reduce_min(out, axis=1)
       out = tf.where(out == 999, arr_1[:,-1], out)
       return out

   def call(self, inputs): 
      inputs = tf.expand_dims(inputs,axis=-1)
      filters = self.weight.shape[1]
      for i in range(filters):
          if (i==0):
             newInputs = tf.concat([inputs],axis=-1)
          else:
             newInputs = tf.concat([newInputs,inputs],axis=-1)
      
      #define the inputs and weights in differential form
      plusIn = (1+newInputs)+1
      minusIn = (1-newInputs)+1
      plusW = 0.5*(3+self.weight)
      minusW = 0.5*(3-self.weight)

      #define x+x, -x-w,-x+w,-w+x
      plusXplusW = tf.add(plusIn,plusW)
      minusXminusW = tf.add(minusIn,minusW)
      plusXminusW = tf.add(plusIn,minusW)
      minusXplusW = tf.add(minusIn,plusW)

      zPlus = tf.concat([plusXplusW,minusXminusW],axis=1)
      zMinus = tf.concat([plusXminusW,minusXplusW],axis=1)
      print(zPlus.shape)
      
      #differential outputs of spikeMP - zplus and zminus
      zPlus = self.spikeMP(zPlus,self.gamma)
      print(zPlus.shape)
      print(zPlus.shape)
      zMinus = self.spikeMP(zMinus,self.gamma)
      print(zMinus.shape)

      
      #final output = relu(zplus - zminus)
      output = tf.nn.relu(zPlus-zMinus)
      print(output.shape)
      return output

   def compute_output_shape(self, input_shape): return (input_shape[0], self.output_dim)


class HidLayerFuncMP():
   def __init__(self, weight, in_size, output_dim, gamma):
      self.output_dim = output_dim
      self.gamma = gamma
      self.weight = weight

   #defining TEMP operation 
   def spikeMP(self, inMat, gamma):
       inMat = tf.sort(inMat,axis=1)
       shape = tf.shape(inMat)
       batch_num = shape[0]
       in_size = shape[1]
       out_size = shape[2]
       print(batch_num,in_size,out_size)
       cs_t = tf.cumsum(inMat,axis=1)
       cs_t = cs_t+gamma
       d = tf.ones([batch_num,in_size,out_size],dtype=tf.dtypes.float32)
       d = tf.cumsum(d,axis=1)
       cs_t = tf.divide(cs_t,d)

       row_to_add = 999*tf.ones([batch_num,1,out_size],dtype=tf.dtypes.float32)
       arr_1 = tf.concat((row_to_add, cs_t), axis=1)

       row_to_add = tf.zeros([batch_num,1,out_size],dtype=tf.dtypes.float32)
       arr_2 = tf.concat((inMat, row_to_add), axis=1)

       #print(tf.keras.backend.eval(arr_1))
       out = tf.where(arr_1 > arr_2, 999 * tf.ones_like(arr_1), arr_1)
       #print(out.shape)
       out = tf.reduce_min(out, axis=1)
       out = tf.where(out == 999, arr_1[:,-1], out)
       return out


   def forward(self, inputs):
      inputs = tf.expand_dims(inputs,axis=-1)
      filters = self.weight.shape[1]
      for i in range(filters):
          if (i==0):
             newInputs = tf.concat([inputs],axis=-1)
          else:
             newInputs = tf.concat([newInputs,inputs],axis=-1)


      #define the inputs and weights in differential form
      plusIn = (1+newInputs)+1
      minusIn = (1-newInputs)+1
      plusW = 0.5*(3+self.weight)
      minusW = 0.5*(3-self.weight)

      #define x+x, -x-w,-x+w,-w+x
      plusXplusW = tf.add(plusIn,plusW)
      minusXminusW = tf.add(minusIn,minusW)
      plusXminusW = tf.add(plusIn,minusW)
      minusXplusW = tf.add(minusIn,plusW)
      
      #differential outputs of spikeMP - zplus and zminus
      zPlus = tf.concat([plusXplusW,minusXminusW],axis=1)
      zMinus = tf.concat([plusXminusW,minusXplusW],axis=1)
      zPlus = self.spikeMP(zPlus,self.gamma)
      zMinus = self.spikeMP(zMinus,self.gamma)

      tp = tf.expand_dims(zPlus,axis=2)
      tm = tf.expand_dims(zMinus,axis=2)

      output = tf.nn.relu(zPlus-zMinus)
      print(output.shape)
      return output



#define TEMP based convolution layer
class customConvMP(Layer):
   def __init__(self, filters, kernel_size, gamma, kernel_initializer='kernel_initializer', activation=None, bias_initializer='zeros', strides=1, padding='valid', **kwargs):
      self.filters = filters

      self.gamma = gamma
      self.kernel_size = kernel_size#conv_utils.normalize_tuple(kernel_size, 2,
      #                                                'kernel_size')
      self.strides = strides#conv_utils.normalize_tuple(strides, 2, 'strides')
      self.padding = conv_utils.normalize_padding(padding)
      self.kernel_initializer = initializers.get(kernel_initializer)
      self.bias_initializer = initializers.get(bias_initializer)
      self.data_format = "channels_last"
      self.activation = activations.get(activation)
      super(customConvMP, self).__init__(**kwargs)

   def build(self, input_shape):
      self.input_dim = input_shape[-1]
      kernel_shape = conv_utils.normalize_tuple(self.kernel_size,2,'kernel_size') + (self.input_dim, self.filters)
      self.output_dim = (input_shape[0],input_shape[1],input_shape[2],self.filters)
      self.kernel = self.add_weight(name = 'kernelPlus', shape = (self.input_dim*self.kernel_size*self.kernel_size, self.filters), trainable = True, initializer = tf.keras.initializers.GlorotNormal(),constraint=lambda x: tf.clip_by_value(x, -3, 3))

      super(customConvMP, self).build(input_shape) #

   def call(self, inputs):
      patches = tf.image.extract_patches(images=inputs, sizes=[1, self.kernel_size, self.kernel_size, 1],
                                           strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                           padding="SAME")



      hidlayer = HidLayerFuncMP(self.kernel,self.kernel_size*self.kernel_size,self.filters,self.gamma)

      print(patches.shape)
      input_size = tf.shape(inputs)
      patches_flatten = tf.reshape(patches,[input_size[0], -1, self.input_dim * self.kernel_size * self.kernel_size])
      print(patches_flatten.shape)
      img_raw = tf.map_fn(hidlayer.forward, patches_flatten)

      size = inputs.shape
      img_reshaped = tf.reshape(img_raw,
                                  [input_size[0], tf.cast(tf.math.ceil(size[1] / self.strides), tf.int32),
                                   tf.cast(tf.math.ceil(size[2] / self.strides), tf.int32),
                                   self.filters])

      print(img_reshaped.shape)
      return img_reshaped

   def compute_output_shape(self, input_shape): return self.output_dim

