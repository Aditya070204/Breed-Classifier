import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 

import cv2 
import tensorflow as tf 
from tensorflow import keras 
from keras import layers 
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from functools import partial 
import albumentations as A 

import warnings 
warnings.filterwarnings('ignore') 
AUTO = tf.data.experimental.AUTOTUNE


# from zipfile import ZipFile 
# data_path = 'dog-breed-identification.zip'
  
# with ZipFile(data_path, 'r') as zip: 
#     zip.extractall() 
#     print('The data set has been extracted.') 

df = pd.read_csv('Dog_Breed_Identifier.py/labels.csv') 
# print(df.head())
# print(df['breed'].nunique())

plt.figure(figsize=(10, 5)) 
df['breed'].value_counts().plot.bar() 
plt.axis('off') 
plt.show() 

df['filepath'] = 'Dog_Breed_Identifier.py/train/' + df['id'] + '.jpg'
print(df.head())
# print(df.head())

# fig,ax = plt.subplots(figsize = (10,10),nrows=3,ncols=4)
# for row in ax:
#     for col in row:
#         k = np.random.randint(0,len(df))
#         img = cv.imread(df.loc[k,'filepath'])
#         col.imshow(img)
#         col.set_title(df.loc[k,'breed'])
#         col.axis('off')
# plt.show()
# k = 0
# filepath = df.loc[k, 'filepath']
# print("Reading image from:", filepath)
# img = cv2.imread(filepath)
# if img is None:
#     print("Error: Unable to read image from:", filepath)
# else:
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     cv2.imshow("Image", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# plt.subplots(figsize=(10, 10)) 
# for i in range(12): 
#     plt.subplot(4, 3, i+1) 

#     # Selecting a random image index from the dataframe.
#     k = np.random.randint(0, len(df))
#     img = cv2.imread(df.loc[k, 'filepath'])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img) 
#     plt.title(df.loc[k, 'breed']) 
#     plt.axis('off') 
# plt.show()

le = LabelEncoder() 
df['breed'] = le.fit_transform(df['breed']) #Converts breed name to numbers so that machine can process them properly
df.head() 

features = df['filepath'] 
target = df['breed'] 

X_train, X_val,Y_train, Y_val = train_test_split(features, target, 
									test_size=0.15, 
									random_state=10) 
# print(X_train.head())#contains features i.e, image
# print(y_train.head())#contains breed number but same index is used as in X_train
# print(X_test.head())#Contains features
# print(y_test.head())#contains breed number but same index is used as in X_test

# print(X_train.shape)
# print(X_test.shape)



transforms_train = A.Compose([ 
    A.VerticalFlip(p=0.5),  # Flip vertically with a 50% chance
    A.HorizontalFlip(p=0.5),  # Flip horizontally with a 50% chance
    A.CoarseDropout(p=0.5),  # Apply CoarseDropout with a 50% chance
    A.RandomGamma(p=0.5),  # Adjust gamma with a 50% chance
    A.RandomBrightnessContrast(p=0.5),  # Adjust brightness and contrast with a 50% chance
    A.ToGray(p=0.5),  # Convert to grayscale with a 50% chance
    A.AdvancedBlur(p=0.5),  # Apply advanced blur with a 50% chance
    # A.BasicTransform(p=0.5)  # Apply basic transformations with a 50% chance
    A.HueSaturationValue(p=0.5),  # Adjust hue, saturation, and value with a 50% chance
    A.RGBShift(p=0.5),  # Shift RGB channels with a 50% chance
    A.ChannelShuffle(p=0.5)  # Shuffle channels with a 50% chance
    # A.GaussianBlur(p=0.5),  # Apply Gaussian blur with a 50% chance
    # A.ColorJitter(p=0.5)  # Jitter color with a 50% chance
])

# img = cv.imread(df.loc[10,'filepath'])
# plt.imshow(img)
# plt.show()

# transform = [A.VerticalFlip(p=1),
#                       A.HorizontalFlip(p=1),
#                       A.CLAHE(p=1),
#                       A.RandomBrightnessContrast(p=1),
#                       A.RandomGamma(p=1),
#                       A.MedianBlur(p=1)]

# fig,ax = plt.subplots(figsize = (15,15),nrows=2,ncols=3)
# i = 0
# for row in ax:
#     for col in row:
#         image = transform[i](image=img)["image"]
#         i+=1
#         col.imshow(image)
#         col.axis('off')
# plt.show()

def aug_fn(img): 
	aug_data = transforms_train(image=img) 
	aug_img = aug_data['image'] 

	return aug_img 


@tf.function
def process_data(img, label):
    aug_img = tf.numpy_function(aug_fn, [img], Tout=tf.float32)
    aug_img.set_shape(img.shape)  
    return aug_img, label


def decode_image(filepath, label=None): 

	img = tf.io.read_file(filepath) 
	img = tf.image.decode_jpeg(img) 
	img = tf.image.resize(img, [128, 128]) 
	img = tf.cast(img, tf.float32) / 255.0

	if label == None: 
		return img 

	return img, tf.one_hot(indices=label, 
						depth=120, 
						dtype=tf.float32) 

# img = cv.imread(df.loc[10,'filepath'])
# plt.imshow(img)
# plt.show()
# img = aug_fn(img)
# plt.imshow(img)
# plt.show()

train_ds = ( 
	tf.data.Dataset 
	.from_tensor_slices((X_train, Y_train)) 
	.map(decode_image, num_parallel_calls=AUTO) 
	.map(partial(process_data), num_parallel_calls=AUTO) 
	.batch(32) 
	.prefetch(AUTO) 
) 

val_ds = ( 
	tf.data.Dataset 
	.from_tensor_slices((X_val, Y_val)) 
	.map(decode_image, num_parallel_calls=AUTO) 
	.batch(32) 
	.prefetch(AUTO) 
) 

for img, label in train_ds.take(1): 
    print(img.shape, label.shape)



pre_trained_model = InceptionV3( 
	input_shape=(128, 128, 3), 
	weights='imagenet', 
	include_top=False
) 
# print(len(pre_trained_model.layers))
# print(pre_trained_model.layers)

for layer in pre_trained_model.layers: 
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7') 
# print('last layer output shape: ', last_layer.output.shape) 
last_output = last_layer.output

# Model Architecture 
x = layers.Flatten()(last_output) 
# Rectified Linear Unit
x = layers.Dense(256, activation='relu')(x) 
x = layers.BatchNormalization()(x) 
x = layers.Dense(256, activation='relu')(x) 
x = layers.Dropout(0.3)(x) 
x = layers.BatchNormalization()(x) 
output = layers.Dense(120, activation='softmax')(x) 

model = keras.Model(pre_trained_model.input, output) 

# Model Compilation 
model.compile( 
      # Adaptive Moment Estimation
	optimizer='adam', 
	loss=keras.losses.CategoricalCrossentropy(from_logits=True), 
      # Area under the ROC Curve
	metrics=['AUC'] 
) 



class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        val_auc = logs.get('val_auc')
        if val_auc is not None and val_auc > 0.99: 
            print('\n Validation accuracy has reached up to 99%, so stopping further training.') 
            self.model.stop_training = True

es = EarlyStopping(patience=3, 
				monitor='val_auc', 
				restore_best_weights=True,
				mode='max') 

lr = ReduceLROnPlateau(monitor='val_loss', 
					patience=2, 
					factor=0.5, 
					verbose=1) 

# Model Training

history = model.fit(train_ds, 
					validation_data=val_ds, 
					epochs=50, 
					verbose=1, 
					callbacks=[es, lr, myCallback()]) 

#Saving the trained model
model.save('dog_breed_classifier.keras')

history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['auc','val_auc']].plot()
plt.show()

# Load an image you want to classify
image_path = 'D:/Notes/ML Projects/Dog_Breed_Identifier.py/0dee0e6895ebcc7c6d0fe88772619b38.jpg'
img = decode_image(image_path)

# Reshape the image to match the input shape of the model
img = tf.expand_dims(img, axis=0)

# Use the trained model to predict the breed
predictions = model.predict(img)

# Get the predicted breed
predicted_breed_index = np.argmax(predictions)
predicted_breed = le.inverse_transform([predicted_breed_index])[0]

print("Predicted breed:", predicted_breed)
