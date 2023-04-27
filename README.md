# Time-to-Event-Margin Propagation (TEMP)-framework 
Training and inference codes on the MNIST Dataset to accompany the paper titled "Neuromorphic Computing with AER using Time-to-Event-Margin Propagation."

Contents of the Repository

1. Consists of 3 folders to implement the following fully connected and convolution network architectures. 

    a. 3-layer fully connected network (MLP784X100X10)

    b. CNN network with BN in the output layer (CNN_bn_last_layer)

    c. CNN with BN after every successive layer (CNN_2_conv_layer)  

2. Each folder consists of the following files:

    a. Layers_Train.py -  Code to create custom layers (Fully connected, Convolution layers) that incorporate TEMP-based computations

    b. *_Train.py - Train a TEMP-based neural network on the MNIST Dataset

    c. *_Inference.py -  Code for running inference on a trained TEMP-based network.

    d. Models Folder - consists of two files that save the initial and trained weight configurations for different network architectures.
    
Instructions
1. To obtain the results reported in the paper, run the inference codes by loading the trained weight values saved in the Models folder. 

        E.g., For the MLP Model, download the MLP_Inference.py, Layers_Train.py files and the Models Folder, and run the command:  
        python MLP_Inference.py
        
2. To train the models, download the respective folders, and run the *_Train.py files.

         E.g., For training the MLP model, run the command: python MLP_Train.py. 
        
Environment Settings
    
1. tensorflow-gpu               2.8.0

2. keras                        2.8.0

3. Keras-Preprocessing          1.1.2

4. keras-tuner                  1.1.2

5. ipykernel                    6.16.2

6. ipython                      8.5.0

7. pandas                       1.5.0


​
