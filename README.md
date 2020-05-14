# Udacity - Create Your Own Image Classifier
This is part of "Intro to Machine Learning" Nanodegree.

This project is to implement an image classifier with PyTorch, then convert it into a command line applications: train.py, predict.py.

## Prequisite:
* Python 3.6
* Pytorch


## Project Part 1
The image classifier uses a pretrained network like densenet121 to train on 102 different species of flowers and to predict the flower in the given image and its probability. It uses ReLU(Recitified Linear Unit) activations and LogSoftmax to predict the image. It also uses dropout strategy to prevent overfitting. The training loss, validation loss and accuracy were tracked to identify optimal hyperparameters and then the model is used on the test set to determine the testing loss and its accuracy.

## Project Part 2
Command line applications includes train.py and predict.py

Following are various command line arguments available for train.py:

* 'data_dir', help = 'Provide data directory'
* '--save_dir', help = 'Optional: Provide saving directory. Default is current directory'
* '--arch', help = 'Optional: Currently it supports only densenet121, densenet169, densenet161, densenet201 and alexnet pretrained model. Default pretrained model is densenet121'
* '--learning_rate', help = 'Optional: Learning rate. Default value is 0.003'
* '--input_units', help = 'Optional: Input units in Classifier. Default value is 1024'
* '--hidden_units', help = 'Optional: Hidden units in Classifier. Default value is 512'
* '--output_units', help = 'Optional: Hidden units in Classifier. Default value is 102'
* '--epochs', help = 'Optional: Number of epochs. Default value is 5'
* '--GPU', help = 'Optional: Option to use GPU. Default is GPU'
    
Following are various command line arguments available for predict.py

* 'imagefile', help = 'Provide path of the image file'
* 'checkpoint', help = 'Provide path of the checkpoint. Default is current directory'
* '--top_k', help = 'Optional: Return top K most likely classes. Default value is 5'
* '--category_names', help = 'Optional: Provide path of file that has mapping of categories to real names. Default is ./cat_to_name.json'
* '--GPU', help = 'Optional: Option to use GPU. Default is GPU'
