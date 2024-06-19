import numpy as np
import matplotlib.pyplot as plt

# Load the 3D numpy array from the .npy file
array_3d = np.load('./3d_dataset/not_normal/not_normal_volume_20.npy')

# Initial slice index
current_slice = 0

# Create a figure and axis
fig, ax = plt.subplots()

# array_3d = (array_3d + 1) * 127.5

print(array_3d.max())

# Display the initial slice
slice_2d = array_3d[current_slice, :, :]
img = ax.imshow(slice_2d, cmap='gray', vmin=0, vmax=255)
ax.set_title(f'Slice {current_slice}')
ax.axis('off')

# Function to update the slice displayed
def update_slice(new_slice):
    global current_slice
    current_slice = new_slice
    slice_2d = array_3d[current_slice, :, :]
    img.set_data(slice_2d)
    ax.set_title(f'Slice {current_slice}')
    fig.canvas.draw_idle()

# Function to handle scroll events
def on_scroll(event):
    global current_slice
    if event.button == 'up':
        new_slice = (current_slice + 1) % array_3d.shape[0]
    elif event.button == 'down':
        new_slice = (current_slice - 1) % array_3d.shape[0]
    update_slice(new_slice)

# Connect the scroll event to the handler
fig.canvas.mpl_connect('scroll_event', on_scroll)

plt.show()