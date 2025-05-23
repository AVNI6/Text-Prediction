# Text-Prediction

This project is a Machine Learning application that performs **next-word prediction** using a trained **LSTM (Long Short-Term Memory)** neural network model. When a user enters a word or phrase, the model predicts the most likely next words based on the training data.

## Features

- Predicts next words based on user input
- Trained on a large text dataset
- Built with TensorFlow/Keras and LSTM architecture
- Tokenization and sequence padding handled using Keras tools
- Easy to train and extend for your own dataset

## Model Overview

- **Embedding Layer** – Turns words into dense vectors
- **LSTM Layer** – Learns long-term dependencies from sequences
- **Dense Layer** – Predicts the next word using softmax activation
