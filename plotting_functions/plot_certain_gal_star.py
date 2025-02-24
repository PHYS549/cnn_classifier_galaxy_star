import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os

def load_star_gal_data(path):
    assert os.path.exists(path), "Path to load from does not exist"
    files = os.listdir(path)
    assert files, "No files in target path found"

    frames = {}
    stars = []
    gals = []
    for file in files:
        if file == "frames_aligned.npy":
            frames = np.load(path + file, allow_pickle=True).item()
        if file == "gal.fits.gz":
            gals = fits.open(path + file)[1]
        if file == "star.fits.gz":
            stars = fits.open(path + file)[1]
    return frames, gals, stars

def plot_object_on_frame(rerun, run, camcol, field, filter_brightness, bright_ones, band_num, object_type, ax=None, color='red', marker='x'):
    """
    Plots the position of a specific object (star or galaxy) on a specific frame.
    
    Parameters:
    rerun (int): The rerun number
    run (int): The run number
    camcol (int): The camcol number
    field (int): The field number
    band_num (int): The band number (0 for u, 1 for g, 2 for r, etc.)
    object_type (str): Type of object ('star' or 'galaxy')
    ax (matplotlib axis, optional): Axis to plot on (useful for overlaying multiple objects)
    color (str, optional): Color of the marker (default is 'red')
    marker (str, optional): Marker style for the object (default is 'x')
    """
    
    # Load the aligned frames and target coordinates
    path = f"./preprocessed_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    frames, gals, stars = load_star_gal_data(path)

    # Get the frame data and WCS for the specified field
    frame_data, frame_wcs = frames[str(field)]
    frame_data = frame_data[band_num]
    
    # Get the target coordinates (stars or galaxies)
    if filter_brightness:
        if bright_ones:
            gals = np.load(path + "bright-target_gals.npy", allow_pickle=True).item()
            stars = np.load(path + "bright-target_stars.npy", allow_pickle=True).item()
        else:
            gals = np.load(path + "dark-target_gals.npy", allow_pickle=True).item()
            stars = np.load(path + "dark-target_stars.npy", allow_pickle=True).item()
    else:
        gals = np.load(path + "no_filter-target_gals.npy", allow_pickle=True).item()
    
    if object_type == 'star':
        targets = stars
    elif object_type == 'galaxy':
        targets = gals
    else:
        raise ValueError("Invalid object_type. Choose 'star' or 'galaxy'.")
    
    
    # Get the pixel position of the object (star or galaxy) on the frame
    object_pixels = targets[str(field)]
    
    # Check if the index is valid
    object_index = np.random.randint(len(object_pixels))
    
    
    # Get the coordinates of the specific object
    object_coord = object_pixels[object_index]
    
    # Plot the frame
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(frame_data, cmap='gray', origin='lower', interpolation='none')
    
    # Plot the object position
    ax.scatter(object_coord[0], object_coord[1], color=color, marker=marker, label=f"{object_type.capitalize()} {object_index}")
    
    return ax

def plot_a_star_and_a_galaxy(rerun, run, camcol, field, filter_brightness, bright_ones, band_num):
    # Plotting both objects (star and galaxy) on the same frame
    ax = plot_object_on_frame(rerun, run, camcol, field, filter_brightness, bright_ones, band_num, 'star', color='red', marker='x')

    # Now plot the galaxy on the same axis with different color and marker
    plot_object_on_frame(rerun, run, camcol, field, filter_brightness, bright_ones, band_num, 'galaxy', ax=ax, color='blue', marker='o')

    # Add labels and title
    ax.set_title(f"Star and Galaxy Positions on Frame {field}")
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    ax.legend()

    # Show the plot
    plt.show()