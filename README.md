# Breast-Cancer-Classification-using-Neural-Network
Breast Cancer Classification using Neural Network

This repository contains code for a breast cancer classification model built using a neural network with the Keras API and TensorFlow backend. The model takes in 30 input features and predicts the binary classification of benign or malignant breast tumors.

Model Architecture

The model architecture consists of three layers: a flatten layer as the input layer, a dense layer with 20 nodes and a ReLU activation function as the first hidden layer, and a dense layer with 2 nodes and a sigmoid activation function as the output layer.

Model Compilation

The model is compiled using the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy metric.

Usage

The main script, "breast_cancer_classification.py", contains code for loading and preprocessing the data, training the model, and evaluating its performance on a test set.

Requirements

The code requires the following packages:

TensorFlow
Keras
scikit-learn
numpy
pandas
Credits

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Data Set from the UCI Machine Learning Repository.

Thanks!!
