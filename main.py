import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Load and Augment the Dataset
dataset_dir = 'water_spinach_dataset'

img_size = (224, 224)
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Load the dataset with labels from subfolders (fresh, wilted, aging_dying, rotten)
train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Prefetch for performance
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y)).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# 2️⃣ Build the Transfer Learning Model
base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

global_avg_pool = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(4, activation='softmax')  # 4 classes

model = models.Sequential([
    base_model,
    global_avg_pool,
    layers.Dropout(0.2),
    output_layer
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3️⃣ Train the Model
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Optional: Unfreeze some layers of base_model and fine-tune
# base_model.trainable = True
# model.compile(optimizer=Adam(learning_rate=1e-5),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# fine_tune_epochs = 5
# history_fine = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs)

# 4️⃣ Save the Model
model.save("water_spinach_freshness_model.h5")
print("Model saved to water_spinach_freshness_model.h5")

