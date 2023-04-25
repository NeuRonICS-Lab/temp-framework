# Time-to-Event-Margin Propagation (TEMP)-framework 
Training and inference codes on the MNIST Dataset to accompany the paper titled "Neuromorphic Computing with AER using Time-to-Event-Margin Propagation"

Contents of the Repository
1. MLP784X100X10 folder - 

    a. MLP_Train.py - Tensorflow code to train a fully-connected network with layers implementing TEMP computation
    
    b. MLP_Inference.py - Tensorflow code for running inference on a saved model
    
    c. Layers_Train.py - Tensorflow code to create custom layers (Fully connected, Convolution layers) that incorporate TEMP based computations

    d. Models folder -
    
        a. model_MLP_100hid.hdf5 - saved weights of the 784x100x10 network that give a test accuracy of 97.8%
        
        b. model_MLP_100hid_initialW.hdf5 - weight initialization

2. CNN_bn_last_layer folder - 

    a. MP_CNN_Train_one_bn.py - Tensorflow code to train a convolution network with layers implementing TEMP computation
    
    b. MP_CNN_Inference_one_bn.py - Tensorflow code for running inference on a saved model
    
    c. Layers_Train.py - Tensorflow code to create custom layers (Fully connected, Convolution layers) that incorporate TEMP based computations

    d. Models folder -
    
        a. model_Batch32_Hid15_oldW_oneBN.hdf5 - saved weights of the CNN network that give a test accuracy of 97.8%
        
        b. model_Batch32_Hid15_initialW_oneBN.hdf5 - weight initialization

3. CNN_2_conv_layer folder - 

    a. MP_CNN_Train.py - Tensorflow code to train a 2-convolution network with layers implementing TEMP computation
    
    b. MP_CNN_Inference.py - Tensorflow code for running inference on a saved model
    
    c. Layers_Train.py - Tensorflow code to create custom layers (Fully connected, Convolution layers) that incorporate TEMP based computations

    d. Models folder -
    
        a. Batch16_OldW_CNN16and32_CNNHidDO0p1_Hid500_W.hdf5 - saved weights of the CNN network that give a test accuracy of 99.2%
        
        b. Batch16_OldW_CNN16and32_CNNHidDO0p1_Hid500_initialW.hdf5 - weight initialization

Instructions
1. To obtain the results reported in the paper, run the inference codes, by loading the trained weight values saved in the Models folder. 
        eg. For the MLP Model, download the MLP_Inference.py, Layers_Train.py files and the Models Folder, to a Python environment, and run the command:  
        python MLP_Inference.py
2. To train the models, download the respective folders, and run the *_Train.py files.
        eg. For training the MLP model, run the command: python MLP_Train.py. 
        
Environment Settings

    Package                     Version  
    
1. tensorflow-gpu               2.8.0

2. keras                        2.8.0

3. Keras-Preprocessing          1.1.2

4. keras-tuner                  1.1.2

5. ipykernel                    6.16.2

6. ipython                      8.5.0

7. pandas                       1.5.0

