# OCR
Optical Character Recognition Tensorflow Models

This is a Tensorflow implementation of an Optical Character Recognition architecture I implemented during my master’s thesis. It was applied on the IAM dataset.  
http://www.fki.inf.unibe.ch/databases/iam-handwriting-database  
The architecture consists of an encoder module in form of a CNN, a GRU based RNN module which contains an attention mechanism to attend to different parts of the input selectively. To reduce the memory requirements and to handle various wide images a sliding window-based approach was taken. The image is encoded by a CNN and then cut up into several pieces that is determined by the width of each window, and the offset of each window (They can overlap. It is similar on how the size of a convolution result is determined). An example word and the first four windows can be found in the following figures:  
![](images/29.png)  
![](images/29_0.png)
![](images/29_1.png)
![](images/29_2.png)
![](images/29_3.png)  
Each image window is then inserted into the RNN module which can attend to specific spots in the image. The RNN outputs a character at each time step and a Connectionist Temporal Classification loss is used to count repeated strings as one. During application a greedy scheme is applied were repeated characters count as one if there is not a separation symbol between them. An image of the architecture can be found in the following figure:  
![](images/Model.png)
  
More information can be found in the master’s thesis at docs/Thesis_Kappen.pdf  
##  Dependencies
The code requires the following libraries in order to work:    
Tensorflow 1.0.0.  
It does not work with Tensorflow versions starting from 2.0 without change.    
Numpy  
Scipy  
Random  
Math  
OS  

##  Installation
In order to use the project installation of all requirements in a virtual environment is necessary and download of the src folder is necessary. It can then be executed with interpreter of the virtual enrionment.  
##Example
The software has,to be installed as stated in installation. In order to use the example, the IAM dataset must be downloaded and the ConvertToFile in the src/utility.py file must be executed on the respective files of the IAM dataset. It will create a train set data file, a validation set data file, a test set data file for training and a meta data file for meta information. This step was necessary to improve the training speed. The in src contained file encoder_decoder_model_1_translator.txt. must also be downloaded. It translates characters to numbers and adds additional characters. In the main file all LOCATION constant variables must be set to the respective file locations. It can then be executed. It will store a checkpoint every 2000 steps and create a Tensorboard summary every 10 steps in a subdirectory of the directory containing the main.py file. Results of a run can be found in the following figure:  
![](images/Attention_Model_Results.png)  
The tensorboard file can be found in the Attention_Model_Trace folder.

