#import numpy as np
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping


# Define the batch size
BATCH_SIZE = 32
NUM_EPOCHS = 1  # Increase the number of epochs
NUM_CLASSES = 2  # Binary task

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# Load the VGG19 with pretrained weights
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3))

for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of ResNet50
x = base_model.output
x = Flatten()(x)  # Add a flatten layer
x = Dropout(0.2)(x)  # Add a dropout layer with 50% dropout rate
x = Dense(NUM_CLASSES, activation='softmax')(x)  # Add a dense softmax layer with 10 output units

model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators with green channel extraction and ResNet50 preprocessing
image_datagen = ImageDataGenerator(
    preprocessing_function=(lambda x: x / 127.5 - 1.),
    # Other augmentation and parameters here if needed
)

train_generator = image_datagen.flow_from_directory(
    './data/train',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = image_datagen.flow_from_directory(
    './data/validation',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
)


# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Train the model with oversampling and early stopping
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping]
)

model_path = './models/classifier/vgg_model.h5'

# Save the model
model.save(model_path)

test_generator = image_datagen.flow_from_directory(
    './data/test',
    batch_size=BATCH_SIZE,
    target_size=(512, 512),
    class_mode='categorical',
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator)

print(f'test_acc: {test_acc}')
