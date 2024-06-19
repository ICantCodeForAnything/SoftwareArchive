# This is similar to view_single_volume except we can load the 3 volumes side by side and do a comparison directly
# The 3 volumes being the normal, translated, and reconstructed volume

import numpy as np
import matplotlib.pyplot as plt

# Load the 3D numpy arrays from the .npy files

epoch = 19
batch_number = 0


# original_volume = np.load(f'./dense_1_old_images/original_not_normal_{epoch}_{batch_number}.npy')
# translated_volume = np.load(f'./dense_1_old_images/translated_not_normal_{epoch}_{batch_number}.npy')
# recons_volume = np.load(f'./dense_1_old_images/recon_not_normal_{epoch}_{batch_number}.npy')

# original_volume = np.load(f'./dense_1_old_images/original_normal_{epoch}_{batch_number}.npy')
# translated_volume = np.load(f'./dense_1_old_images/translated_normal_{epoch}_{batch_number}.npy')
# recons_volume = np.load(f'./dense_1_old_images/recon_normal_{epoch}_{batch_number}.npy')

# for i in range(100):
#     epoch = i

#     original_volume = np.load(f'./images/original_normal_{epoch}_{batch_number}.npy')
#     translated_volume = np.load(f'./images/translated_normal_{epoch}_{batch_number}.npy')
#     recons_volume = np.load(f'./images/recon_normal_{epoch}_{batch_number}.npy')

#     original_volume = 0.5 * original_volume + 0.5
#     translated_volume = 0.5 * translated_volume + 0.5
#     recons_volume = 0.5 * recons_volume + 0.5
    
#     print(i, translated_volume.min(), translated_volume.max())

original_volume = np.load(f'./images/original_normal_{epoch}_{batch_number}.npy')
translated_volume = np.load(f'./images/translated_normal_{epoch}_{batch_number}.npy')
recons_volume = np.load(f'./images/recon_normal_{epoch}_{batch_number}.npy')

    
# for i in range(100):
#     epoch = i

#     original_volume = np.load(f'./images/original_not_normal_{epoch}_{batch_number}.npy')
#     translated_volume = np.load(f'./images/translated_not_normal_{epoch}_{batch_number}.npy')
#     recons_volume = np.load(f'./images/recon_not_normal_{epoch}_{batch_number}.npy')

#     original_volume = 0.5 * original_volume + 0.5
#     translated_volume = 0.5 * translated_volume + 0.5
#     recons_volume = 0.5 * recons_volume + 0.5
    
#     print(i, translated_volume.min(), translated_volume.max())
    
# epoch = 5
    
# original_volume = np.load(f'./images/original_not_normal_{epoch}_{batch_number}.npy')
# translated_volume = np.load(f'./images/translated_not_normal_{epoch}_{batch_number}.npy')
# recons_volume = np.load(f'./images/recon_not_normal_{epoch}_{batch_number}.npy')
    


original_volume = 0.5 * original_volume + 0.5
translated_volume = 0.5 * translated_volume + 0.5
recons_volume = 0.5 * recons_volume + 0.5
# Initial slice index
current_slice = 0

print(epoch, translated_volume.min(), translated_volume.max())

# Create a figure and axes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display the initial slice from each file
img1 = axes[0].imshow(original_volume[current_slice, :, :], cmap='gray', vmin=0, vmax=1)
axes[0].set_title(f'Original Volume - Slice {current_slice}')
axes[0].axis('off')

img2 = axes[1].imshow(translated_volume[current_slice, :, :], cmap='gray', vmin=0, vmax=1)
axes[1].set_title(f'Translated Volume - Slice {current_slice}')
axes[1].axis('off')

img3 = axes[2].imshow(recons_volume[current_slice, :, :], cmap='gray', vmin=0, vmax=1)
axes[2].set_title(f'Recons Volume - Slice {current_slice}')
axes[2].axis('off')

# Function to update the slices displayed
def update_slices(new_slice):
    global current_slice
    current_slice = new_slice
    
    img1.set_data(original_volume[current_slice, :, :])
    axes[0].set_title(f'Original Volume - Slice {current_slice}')
    
    img2.set_data(translated_volume[current_slice, :, :])
    axes[1].set_title(f'Translated Volume - Slice {current_slice}')
    
    img3.set_data(recons_volume[current_slice, :, :])
    axes[2].set_title(f'Recons Volume - Slice {current_slice}')
    
    fig.canvas.draw_idle()

# Function to handle scroll events
def on_scroll(event):
    global current_slice
    if event.button == 'up':
        new_slice = (current_slice + 1) % original_volume.shape[0]
    elif event.button == 'down':
        new_slice = (current_slice - 1) % original_volume.shape[0]
    update_slices(new_slice)

# Connect the scroll event to the handler
fig.canvas.mpl_connect('scroll_event', on_scroll)

plt.show()