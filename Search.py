from keras.models import load_model
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the saved model
model = load_model('Dog_Breed_Identifier.py/dog_breed_classifier.keras')

# Load the LabelEncoder (if necessary)
# Make sure to fit it to the training labels before saving the model

# Load the labels DataFrame
df = pd.read_csv('Dog_Breed_Identifier.py/labels.csv')

# Assuming 'breed' column contains the labels
Y_train = df['breed'].values

# Function to preprocess the image
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [128, 128])
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Function to predict the breed of the dog
def predict_breed(image_path, model):
    # Load the LabelEncoder and fit it to the training labels
    le = LabelEncoder()
    le.fit(Y_train)  # Make sure Y_train is defined
    
    # Preprocess the image
    img = preprocess_image(image_path)
    img = tf.expand_dims(img, axis=0)
    
    # Use the model to predict the breed
    predictions = model.predict(img)
    
    # Get the predicted breed index
    predicted_breed_index = np.argmax(predictions)
    
    # Inverse transform the predicted breed index to get the label
    predicted_breed_label = le.inverse_transform([predicted_breed_index])[0]
    
    return predicted_breed_label

# Example usage
image_path = 'Dog_Breed_Identifier.py/00cc68a50b2d016a6b29af628ea4e04b.jpg'
predicted_breed = predict_breed(image_path, model)
print("Predicted breed:", predicted_breed)