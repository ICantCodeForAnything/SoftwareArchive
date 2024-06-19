import os
import shutil
import random
from PIL import Image

# Define paths
source_folder = './other_data/normal'
destination_folder_normal = './DR_box/normal'
destination_folder_not_normal = './DR_box/not_normal'

# Create destination folders if they don't exist
os.makedirs(destination_folder_normal, exist_ok=True)
os.makedirs(destination_folder_not_normal, exist_ok=True)

# List all images in the source folder
all_images = os.listdir(source_folder)

# Shuffle the list of images
random.shuffle(all_images)

# Select 1000 random images
selected_images = random.sample(all_images, 1000)

# Process each selected image
for image_name in selected_images:
    # Load the image
    image_path = os.path.join(source_folder, image_name)
    image = Image.open(image_path)
    width, height = image.size
    
    # Generate random coordinates for the box
    box_size = min(width, height) // 4
    left = random.randint(0, width - box_size)
    top = random.randint(0, height - box_size)
    right = left + box_size
    bottom = top + box_size
    
    # Crop the image to create a box
    box_image = image.crop((left, top, right, bottom))
    
    # Remove the box from the original image
    image.paste((255, 255, 255), (left, top, right, bottom))  # Fills the box with white color
    
    # Save the modified image to the destination folder
    if len(os.listdir(destination_folder_normal)) < 500:
        # Copy to 'normal' folder
        shutil.copy(image_path, destination_folder_normal)
    else:
        # Copy to 'not_normal' folder
        image.save(os.path.join(destination_folder_not_normal, image_name))
