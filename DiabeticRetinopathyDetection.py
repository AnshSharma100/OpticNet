import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

data_dir = r'C:\Users\sharm\Documents\organized images'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='sparse',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='sparse',
    subset='validation'
)

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

base_model = tf.keras.models.load_model('diabetic_retinopathy_model.h5')
input_shape = base_model.input_shape[1:]

for layer in base_model.layers:
    layer.trainable = False

new_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

new_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

def lr_schedule(epoch, lr):
    return lr * 0.1 if epoch > 20 else lr

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),
    ModelCheckpoint(filepath='best_model.h5', save_best_only=True, verbose=1),
    LearningRateScheduler(lr_schedule, verbose=1)
]

history = new_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights
)

new_model.save('new_diabetic_retinopathy_model.h5')
new_model.summary()
