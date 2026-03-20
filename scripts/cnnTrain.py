import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

train_dir = "data/processed/train"
val_dir = "data/processed/val"

datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
train_gen = datagen.flow_from_directory(train_dir, target_size=(128,128), batch_size=16, class_mode='binary')
val_gen = datagen.flow_from_directory(val_dir, target_size=(128,128), batch_size=16, class_mode='binary')

base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(128,128,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=5)
model.save("models/cnn_mobilenetv3.keras")