#import numpy as np
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping

# Define the batch size
BATCH_SIZE = 16
NUM_EPOCHS = 1  # Increase the number of epochs
NUM_CLASSES = 2  # Binary task

# Load the ResNet50 model with pretrained weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512,512,3))

# Add custom layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators with green channel extraction and ResNet50 preprocessing
image_datagen = ImageDataGenerator(
    preprocessing_function=(lambda x: x / 127.5 - 1.),
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = image_datagen.flow_from_directory(
    './data/train',
    target_size=(512, 512),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
)

validation_generator = image_datagen.flow_from_directory(
    './data/validation',
    target_size=(512, 512),
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

model_path = './models/classifier/resnet_model.h5'

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
