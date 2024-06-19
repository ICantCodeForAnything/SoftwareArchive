import os
import numpy as np
from PIL import Image, ImageDraw
import random

def generate_image_with_gray_square(image_size, square_size_percentage):
    # Create a fully black image
    image = Image.new("L", image_size, color=0)  # 'L' mode means grayscale
    draw = ImageDraw.Draw(image)

    # Calculate the square size
    square_width = int(image_size[0] * square_size_percentage)
    square_height = int(image_size[1] * square_size_percentage)
    
    # Determine the top-left corner of the gray square
    x = random.randint(0, image_size[0] - square_width)
    y = random.randint(0, image_size[1] - square_height)
    
    # Define the gray color (value between 0 and 255)
    gray_color = 128  # Mid-level gray
    
    # Draw the gray square
    draw.rectangle([x, y, x + square_width, y + square_height], fill=gray_color)
    
    return image

def generate_image_with_gray_and_white_square(image_size, square_size_percentage):
    # Create a fully black image
    image = Image.new("L", image_size, color=0)
    draw = ImageDraw.Draw(image)

    # Calculate the square size
    square_width = int(image_size[0] * square_size_percentage)
    square_height = int(image_size[1] * square_size_percentage)
    
    # Determine the top-left corner of the gray square
    x = random.randint(0, image_size[0] - square_width)
    y = random.randint(0, image_size[1] - square_height)
    
    # Define the gray color
    gray_color = 128
    
    # Draw the gray square
    draw.rectangle([x, y, x + square_width, y + square_height], fill=gray_color)
    
    # Calculate the white square size (smaller than gray square)
    white_square_width = int(square_width * 0.5)
    white_square_height = int(square_height * 0.5)
    
    # Determine the top-left corner of the white square inside the gray square
    white_x = x + (square_width - white_square_width) // 2
    white_y = y + (square_height - white_square_height) // 2
    
    # Define the white color
    white_color = 255
    
    # Draw the white square
    draw.rectangle([white_x, white_y, white_x + white_square_width, white_y + white_square_height], fill=white_color)
    
    return image

def create_dataset(num_images, image_size, square_size_percentage, output_dir):
    class1_dir = os.path.join(output_dir, 'class1')
    class2_dir = os.path.join(output_dir, 'class2')
    
    # Create the directories if they don't exist
    os.makedirs(class1_dir, exist_ok=True)
    os.makedirs(class2_dir, exist_ok=True)
    
    # Generate class1 images
    for i in range(num_images):
        image = generate_image_with_gray_square(image_size, square_size_percentage)
        image_path = os.path.join(class1_dir, f'image_{i}.png')
        image.save(image_path)
    
    # Generate class2 images
    for i in range(num_images):
        image = generate_image_with_gray_and_white_square(image_size, square_size_percentage)
        image_path = os.path.join(class2_dir, f'image_{i}.png')
        image.save(image_path)

# Specify parameters
num_images = 10  # Number of images per class
image_size = (512, 512)  # Image dimensions
square_size_percentage = 0.25  # Square size as a percentage of the image size
output_dir = 'toy_dataset'  # Output directory

# Create the dataset
create_dataset(num_images, image_size, square_size_percentage, output_dir)