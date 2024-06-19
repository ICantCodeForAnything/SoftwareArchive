import numpy as np
import os
import cv2

def create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path, normal_class=True):
    # Create a black image
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate the center of the image
    center_x = width // 2
    center_y = height // 2
    
    # Calculate the length of the line
    length = min(width, height) // 2
    
    # Calculate the starting and ending coordinates of the straight line
    x1 = int(center_x - length * np.cos(np.radians(angle)))
    y1 = int(center_y - length * np.sin(np.radians(angle)))
    x2 = int(center_x + length * np.cos(np.radians(angle)))
    y2 = int(center_y + length * np.sin(np.radians(angle)))
    
    # Calculate the center point of the line
    center_line_x = (x1 + x2) // 2
    center_line_y = (y1 + y2) // 2
    
    # Calculate the direction vector of the line
    line_dx = x2 - x1
    line_dy = y2 - y1
    # Calculate the length of the line
    line_length = np.sqrt(line_dx ** 2 + line_dy ** 2)
    # Normalize the direction vector
    line_dx /= line_length
    line_dy /= line_length
    
    if normal_class:
        cv2.line(image, (x1, y1), (x2, y2), 255, line_thickness)
    else:
        # Calculate the perpendicular direction vector
        perp_dx = -line_dy
        perp_dy = line_dx
        
        # Calculate the peak of the triangular bump
        peak_x = int(center_line_x + triangle_height * perp_dx)
        peak_y = int(center_line_y + triangle_height * perp_dy)
        
        # Calculate the base points of the triangular bump
        base_x1 = int(center_line_x - triangle_base_width / 2 * line_dx)
        base_y1 = int(center_line_y - triangle_base_width / 2 * line_dy)
        base_x2 = int(center_line_x + triangle_base_width / 2 * line_dx)
        base_y2 = int(center_line_y + triangle_base_width / 2 * line_dy)
        
        # Draw the lines forming the sides of the triangle bump
        cv2.line(image, (base_x1, base_y1), (peak_x, peak_y), 255, line_thickness)
        cv2.line(image, (base_x2, base_y2), (peak_x, peak_y), 255, line_thickness)
        
        cv2.line(image, (base_x1, base_y1), (x1, y1), 255, line_thickness)
        cv2.line(image, (base_x2, base_y2), (x2, y2), 255, line_thickness)
    
    cv2.imwrite(save_path, image)

# Parameters
width = 512
height = 512
line_thickness = 5  # Thickness of the lines
angle = 15  # Angle in degrees (positive for counter-clockwise rotation)
triangle_base_width = 100  # Width of the base of the triangular bump
triangle_height = 20  # Height of the triangular bump
num_images = 1000
seed = 33

rng = np.random.default_rng(seed)

os.makedirs(os.path.join('data', 'train', 'normal'), exist_ok=True)
os.makedirs(os.path.join('data', 'train', 'not_normal'), exist_ok=True)
os.makedirs(os.path.join('data', 'validation', 'normal'), exist_ok=True)
os.makedirs(os.path.join('data', 'validation', 'not_normal'), exist_ok=True)
os.makedirs(os.path.join('data', 'test', 'normal'), exist_ok=True)
os.makedirs(os.path.join('data', 'test', 'not_normal'), exist_ok=True)

os.makedirs(os.path.join('cyclegan_data', 'train', 'normal', 'normal'), exist_ok=True)
os.makedirs(os.path.join('cyclegan_data', 'train', 'not_normal', 'not_normal'), exist_ok=True)
os.makedirs(os.path.join('cyclegan_data', 'validation', 'normal', 'normal'), exist_ok=True)
os.makedirs(os.path.join('cyclegan_data', 'validation', 'not_normal', 'not_normal'), exist_ok=True)
os.makedirs(os.path.join('cyclegan_data', 'test', 'normal', 'normal'), exist_ok=True)
os.makedirs(os.path.join('cyclegan_data', 'test', 'not_normal', 'not_normal'), exist_ok=True)


for i in range(num_images):
    angle = rng.uniform(-90, 90)
    save_path = os.path.join('data', 'train', 'normal', f'normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path)

for i in range(num_images):
    angle = rng.uniform(-40, 40)
    save_path = os.path.join('data', 'train', 'not_normal', f'not_normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path, normal_class=False)

for i in range(int(0.2 * num_images)):
    angle = rng.uniform(-40, 40)
    save_path = os.path.join('data', 'validation', 'normal', f'normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path)

for i in range(int(0.2 * num_images)):
    angle = rng.uniform(-40, 40)
    save_path = os.path.join('data', 'validation', 'not_normal', f'not_normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path, normal_class=False)
    
for i in range(int(0.1 * num_images)):
    angle = rng.uniform(-40, 40)
    save_path = os.path.join('data', 'test', 'normal', f'normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path)

for i in range(int(0.1 * num_images)):
    angle = rng.uniform(-40, 40)
    save_path = os.path.join('data', 'test', 'not_normal', f'not_normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path, normal_class=False)
    
    
    

for i in range(num_images):
    angle = rng.uniform(-40, 40)
    save_path = os.path.join('cyclegan_data', 'train', 'normal', 'normal', f'normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path)

for i in range(num_images):
    angle = rng.uniform(-40, 40)
    save_path = os.path.join('cyclegan_data', 'train', 'not_normal', 'not_normal', f'not_normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path, normal_class=False)

for i in range(int(0.2 * num_images)):
    angle = rng.uniform(-40, 40)
    save_path = os.path.join('cyclegan_data', 'validation', 'normal', 'normal', f'normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path)

for i in range(int(0.2 * num_images)):
    angle = rng.uniform(-40, 40)
    save_path = os.path.join('cyclegan_data', 'validation', 'not_normal', 'not_normal', f'not_normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path, normal_class=False)
    
for i in range(int(0.1 * num_images)):
    angle = rng.uniform(-40, 40)
    save_path = os.path.join('cyclegan_data', 'test', 'normal', 'normal', f'normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path)

for i in range(int(0.1 * num_images)):
    angle = rng.uniform(-40, 40)
    save_path = os.path.join('cyclegan_data', 'test', 'not_normal', 'not_normal', f'not_normal_{i}.png')
    # Create the image
    image = create_image(width, height, line_thickness, angle, triangle_base_width, triangle_height, save_path, normal_class=False)