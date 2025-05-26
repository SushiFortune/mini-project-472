import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#Aasiya
digits = load_digits()
X = digits.data       
y = digits.target    

X = X / 16.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

#Rania

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
#Aasiya
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

#Uroosa
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


num_images_to_show = 12
random_indices = np.random.choice(len(X_test), num_images_to_show, replace=False)


fig, axes = plt.subplots(3, 4, figsize=(12, 9))


axes = axes.flatten()

for ax, idx in zip(axes, random_indices):
    ax.matshow(X_test[idx].reshape(8, 8), cmap="gray")  
    ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
    ax.axis("off")

plt.tight_layout()
plt.show()
