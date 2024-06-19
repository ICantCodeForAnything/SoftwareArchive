import os
import numpy as np

def create_normal_class(volume_shape, gray_box_shape):
    volume = np.zeros(volume_shape, dtype=np.uint8)
    x = np.random.randint(0, volume_shape[0] - gray_box_shape[0])
    y = np.random.randint(0, volume_shape[1] - gray_box_shape[1])
    z = np.random.randint(0, volume_shape[2] - gray_box_shape[2])
    volume[x:x+gray_box_shape[0], y:y+gray_box_shape[1], z:z+gray_box_shape[2]] = 128
    return volume

def create_not_normal_class(volume_shape, gray_box_shape, white_box_shape):
    volume = np.zeros(volume_shape, dtype=np.uint8)
    x = np.random.randint(0, volume_shape[0] - gray_box_shape[0])
    y = np.random.randint(0, volume_shape[1] - gray_box_shape[1])
    z = np.random.randint(0, volume_shape[2] - gray_box_shape[2])
    volume[x:x+gray_box_shape[0], y:y+gray_box_shape[1], z:z+gray_box_shape[2]] = 128

    x_inner = x + np.random.randint(0, gray_box_shape[0] - white_box_shape[0])
    y_inner = y + np.random.randint(0, gray_box_shape[1] - white_box_shape[1])
    z_inner = z + np.random.randint(0, gray_box_shape[2] - white_box_shape[2])
    volume[x_inner:x_inner+white_box_shape[0], y_inner:y_inner+white_box_shape[1], z_inner:z_inner+white_box_shape[2]] = 255

    return volume

def save_volumes(volumes, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    for i, volume in enumerate(volumes):
        np.save(os.path.join(save_dir, f"{prefix}_volume_{i}.npy"), volume)

# Parameters
volume_shape = (64, 64, 64)
gray_box_shape = (16, 16, 16)
white_box_shape = (8, 8, 8)
num_samples = 250

# Generate dataset
normal_volumes = [create_normal_class(volume_shape, gray_box_shape) for _ in range(num_samples)]
not_normal_volumes = [create_not_normal_class(volume_shape, gray_box_shape, white_box_shape) for _ in range(num_samples)]

# Save datasets
save_dir = '3d_dataset'
save_volumes(normal_volumes, os.path.join(save_dir, 'normal'), 'normal')
save_volumes(not_normal_volumes, os.path.join(save_dir, 'not_normal'), 'not_normal')
