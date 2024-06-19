import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

def create_train_val_datasets():
    def load_data(data_path, categories):
        data = []
        labels = []
        
        for category in categories:
            folder_path = os.path.join(data_path, category)
            label = categories.index(category)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_path.endswith('.npy'):
                    data.append(np.load(file_path))
                    labels.append(label)
        
        return np.array(data), np.array(labels)

    # Define paths and categories
    data_path = './3d_dataset'
    categories = ['normal', 'not_normal']

    # Load data
    data, labels = load_data(data_path, categories)

    # Reshape data to fit the input shape of the model (width, height, depth, 1)
    data = data.reshape(data.shape[0], 64, 64, 64, 1)

    # Normalize the data
    data = data / 127.5 - 1.

    # Split the data into training, validation, and testing sets
    train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.3, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

    # Create TensorFlow datasets
    batch_size = 4
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(buffer_size=250).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset