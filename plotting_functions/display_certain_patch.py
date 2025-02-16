import numpy as np
import matplotlib.pyplot as plt

def display_patch(gal_patch, star_patch, title=""):
    """Helper function to display galaxy and star patches together with value ranges."""
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))  # 6 columns for 5 bands + label
    fig.suptitle(title)
    
    band_names = ['i', 'r', 'g', 'z', 'u']
    
    # Plot the galaxy patch (top row)
    axes[0, 0].text(0.5, 0.5, "Galaxy", ha='center', va='center', fontsize=16, color='red')  # Label for Galaxy row
    axes[0, 0].axis('off')  # Hide the axes for the label
    for i, ax in enumerate(axes[0, 1:]):  # Start from the second column
        img = gal_patch[:, :, i]
        vmin, vmax = np.min(img), np.max(img)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{band_names[i]}-band\n(min: {vmin:.2f}, max: {vmax:.2f})")
        ax.axis('off')
    
    # Plot the star patch (bottom row)
    axes[1, 0].text(0.5, 0.5, "Star", ha='center', va='center', fontsize=16, color='blue')  # Label for Star row
    axes[1, 0].axis('off')  # Hide the axes for the label
    for i, ax in enumerate(axes[1, 1:]):  # Start from the second column
        img = star_patch[:, :, i]
        vmin, vmax = np.min(img), np.max(img)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{band_names[i]}-band\n(min: {vmin:.2f}, max: {vmax:.2f})")
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the suptitle
    plt.show()

def display_some_patches(rerun, run, camcol, patch_size, field, gal_id, star_id):
    """Display some sample patches from the galaxies and stars."""
    gal_patches = np.load(f"./preprocessed_data/rerun_{rerun}/run_{run}/camcol_{camcol}/patches{patch_size}_gals.npy", allow_pickle=True).item()
    star_patches = np.load(f"./preprocessed_data/rerun_{rerun}/run_{run}/camcol_{camcol}/patches{patch_size}_stars.npy", allow_pickle=True).item()

    gal_patch = gal_patches[str(field)][gal_id]
    star_patch = star_patches[str(field)][star_id] 
        
    # Display both galaxy and star patches together
    display_patch(gal_patch, star_patch, title=f"Field {field} - Galaxy vs Star")