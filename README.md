# CIFAR-10 Image Classification with CNN

This notebook contains my implementation of an image classifier using a Convolutional Neural Network (CNN) in TensorFlow/Keras. The goal is to classify 32x32 pixel color images from the CIFAR-10 dataset into 10 different categories.

## Project Workflow

### 1. Setup and Preprocessing
* **Loading Data:** Imported the dataset directly from `keras.datasets`.
* **Normalization:** Scaled the pixel values by dividing by 255.0 to bring them between 0 and 1.
* **Target Encoding:** Converted integer labels to one-hot encoded vectors.
* **Data Augmentation:** Used `ImageDataGenerator` for random rotations, shifts, and flips to improve generalization.

### 2. Model Architecture
Built a Sequential CNN model with the following structure:
* **Conv Blocks:** 3 main blocks, each using `Conv2D` with ReLU activation followed by `BatchNormalization`.
* **Regularization:** Added `MaxPooling2D` and `Dropout` (ranging from 0.25 to 0.4) after each block to reduce overfitting.
* **Output:** Replaced the traditional Flatten layer with `GlobalAveragePooling2D`, leading into a Final Dense layer (Softmax activation).

### 3. Training & Optimization
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Callbacks Used:** 
  * `EarlyStopping` to prevent overfitting.
  * `ReduceLROnPlateau` to drop the learning rate when loss stalls.
  * `ModelCheckpoint` to save the best weights based on validation accuracy.

## Results
* **Test Accuracy:** 86.68%
* **Test Loss:** 0.4075

The repository also includes code for a confusion matrix heatmap and a custom inference function to predict random single images from the test set.

## Installation
All required dependencies are listed in the `requirements.txt` file. You can install them using:
```bash
pip install -r requirements.txt
```
