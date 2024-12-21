# Hand Gesture Recognition Model

## **Objective**  
This project aims to develop a machine learning model that can accurately identify and classify different hand gestures from image data. The model can be used for intuitive human-computer interaction, enabling gesture-based control systems.

---

## **Dataset**  
- **Source**: [LeapGestRecog Dataset](https://www.kaggle.com/gti-upm/leapgestreco)
- **Features**: The dataset contains images of hand gestures classified into various categories.

---

## **Workflow**  

1. **Data Preprocessing**:  
   - Load images from the dataset and resize them to a consistent size (e.g., 64x64).
   - Normalize the pixel values to range [0, 1] by dividing by 255.
   - Split the dataset into training and validation sets.

2. **Model Architecture**:  
   - We use a Convolutional Neural Network (CNN) for gesture recognition. The architecture includes several convolutional layers followed by fully connected layers for classification.

3. **Model Training**:  
   - Train the model using the training dataset.
   - Use cross-entropy loss and an optimizer (e.g., Adam or SGD) to minimize the loss.
   - Track training and validation accuracy and loss during training.

4. **Model Evaluation**:  
   - Evaluate the model on the validation set to determine its accuracy and identify areas of improvement.
  
## **Technologies Used**:
- Language: Python
- Libraries:
  - PyTorch: For defining and training the neural network.
  - torchvision: For image transformations and data loading.
