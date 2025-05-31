# Team Members : Rania,Aasiya & Uroosa 

# ***************Mini Project 1**********************
"""
**This program performs handwritten digit recognition
using Logistic Regression. It uses the digits dataset from Scikit-learn,
which consists of 8x8 pixel grayscale images of handwritten digits
(0 through 9). The data is first normalized and then split into training and testing sets. 
A logistic regression model is trained on the training data to learn the patterns that distinguish different digits.
 After training, the model is used to predict the labels of the test set. 
The program then evaluates the model’s performance using a classification report and confusion matrix.
 Finally, it visualizes a few test images along with their predicted and actual digit labels to demonstrate how well the model performs.
"""

# Importing libraries

import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization/ plotting
from sklearn.datasets import load_digits  # for loading of the digits dataset
from sklearn.model_selection import train_test_split  #for splitting data into training/testing sets
from sklearn.linear_model import LogisticRegression # for logistic regression model
from sklearn.metrics import classification_report, confusion_matrix  # For model performance



#Loading and preparing the data 

digits = load_digits() # Load the digits dataset 
X = digits.data       # the images dataset 8x 8 64 pixel images
y = digits.target    #the labels dataset to train the model 

X = X / 16.0 # simply normalizing the original pixel values which is  from 0 to 16 to range 0–1 for better model performance

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 #Spliting the dataset into training and testing sets (80% train, 20% test)


model = LogisticRegression(max_iter=1000) # Creating a logistic regression model with more iterations
model.fit(X_train, y_train)   # Train the model using the training data



y_pred = model.predict(X_test) # Predict the labels of the test set , it returns a list of predicted labels for each item in X_test.



print("Classification Report:\n", classification_report(y_test, y_pred))


print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualizing predictions on random test images
num_images_to_show = 12
# Select 12 random indices from the test set
random_indices = np.random.choice(len(X_test), num_images_to_show, replace=False)

# Create a 3x4 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(12, 9))


axes = axes.flatten()
# Loop through each subplot and plot the corresponding image
for ax, idx in zip(axes, random_indices): 
    ax.matshow(X_test[idx].reshape(8, 8), cmap="gray")  # Reshape flat array back to 8x8 image
    ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")# Show true and predicted labels
    ax.axis("off") # Hide axes ticks for cleaner display

plt.tight_layout()
plt.show()
