
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import pathlib
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input


# Define data directory
data_dir = "/content/drive/MyDrive/Data for model/Train"
validation_dir = "/content/drive/MyDrive/Data for model/Validation"
data_dir = pathlib.Path(data_dir)
data_dir = pathlib.Path(validation_dir)

# Define image directories per week
week_image_direct = {
    'week1': list(data_dir.glob("week 1/*")),
    'week2': list(data_dir.glob("week 2/*")),
    'week3': list(data_dir.glob("week 3/*")),
    'week4': list(data_dir.glob("week 4/*")),
    'week5': list(data_dir.glob("week 5/*")),
    'week6': list(data_dir.glob("week 6/*"))
}

# Define label mapping
week_label_dirc = {
    'week1': 0,
    'week2': 1,
    'week3': 2,
    'week4': 3,
    'week5': 4,
    'week6': 5
}

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers
for layer in model.layers:
    layer.trainable = False

# Add custom dense layers
x = layers.Flatten()(model.output)
x = layers.Dense(512, activation='relu')(x)
output = layers.Dense(6, activation='softmax')(x)

# Create new model
model = models.Model(model.input, output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 244),
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 244),
    class_mode='categorical')

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=2,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=2)


# save model
mode.save("wood_model.h5")

# predicte in new image
img = load_img(r'/content/drive/MyDrive/Data for model/Train/week 6/horizontal_flip_1.jpg', target_size=(224, 224))
img = np.array(img)
img1= img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
pro_img = preprocess_input(img1)

model.predict(pro_img)
