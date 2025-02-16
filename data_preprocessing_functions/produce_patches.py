import os
import numpy as np
from tqdm import tqdm

def assert_patch_sizegth_is_odd(patch_size):
    """ Ensure the patch length is odd """
    assert patch_size % 2, "Patch length should be odd."


def create_dir(path):
    """ Helper function to create the directory if it doesn't exist. """
    os.makedirs(path, exist_ok=True)


def extract_padded_patches(frame, data, patch_size, patch_half_width):
    """
    Extracts patches from the frame at the specified data points (gals or stars).

    Args:
        frame (np.array): The frame from which patches are extracted.
        data (np.array): The coordinates of galaxies/stars.
        patch_size (int): The length of the patch (width and height).
        patch_half_width (int): Half the width of the patch.

    Returns:
        np.array: Array of patches extracted from the frame.
    """
    channels, max_height, max_width = frame.shape

    # Padding the frame for extracting patches at the borders
    padded_frame = np.pad(frame, patch_half_width + 1, 'constant', constant_values=0)[patch_half_width + 1:- patch_half_width - 1]

    # Get patch centers and adjust for padding
    patch_centers = np.rint(data).astype(int) + patch_half_width + 1
    right_bottom = patch_centers + patch_half_width + 1
    left_top = patch_centers - patch_half_width

    # Extract patches
    # Define Patch Size
    patches = np.zeros((len(patch_centers), channels, patch_size, patch_size))
    for idx in range(len(patch_centers)):
        patches[idx] = padded_frame[:, left_top[idx, 1]:right_bottom[idx, 1], left_top[idx, 0]:right_bottom[idx, 0]]

    return patches



def produce_patches_per_frame(frame, gals, stars, patch_size):
    """ Generate patches for galaxies and stars in the given frame. """
    assert_patch_sizegth_is_odd(patch_size)
    patch_half_width = patch_size // 2

    # Extract galaxy and star patches
    gal_patches = np.moveaxis(np.array(extract_padded_patches(frame, gals, patch_size, patch_half_width)), 1, 3)
    star_patches = np.moveaxis(np.array(extract_padded_patches(frame, stars, patch_size, patch_half_width)), 1, 3)
    
    return gal_patches, star_patches


def load_data(path):
    """ Load frames, galaxies, and stars from the specified path. """
    frames = np.load(path + "frames_aligned.npy", allow_pickle=True).item()
    gals = np.load(path + "target_gals.npy", allow_pickle=True).item()
    stars = np.load(path + "target_stars.npy", allow_pickle=True).item()
    return frames, gals, stars


def produce_patches_all_frames(rerun, run, camcol, fields, patch_size):
    """ Produce patches for all frames of the specified fields. """
    path = f"./preprocessed_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    frames, gals, stars = load_data(path)

    gal_patches, star_patches = {}, {}
    print("Extracting patches for fields...")
    for field in tqdm(fields, desc="producing patches for all astronomical object images", unit="frame"):
        field_str = str(field)
        if field_str in frames.keys():
            frame = frames[field_str][0]
            gal = gals[field_str]
            star = stars[field_str]
            gal_patches[field_str], star_patches[field_str] = produce_patches_per_frame(frame, gal, star, patch_size)
    
    print("Saving patches...")
    np.save(path + f"patches{patch_size}_gals.npy", gal_patches)
    np.save(path + f"patches{patch_size}_stars.npy", star_patches)


def produce_set_data(gal_patches, star_patches, fields, patch_size, set_name="training set"):
    """ Prepare the data and target labels for training. """
    data = np.zeros((0, patch_size, patch_size, 5))
    
    # Add galaxy patches
    for field in fields:
        data = np.append(data, gal_patches[str(field)], axis=0)
    
    gal_count = len(data)

    # Add star patches
    for field in tqdm(fields, desc="producing " + set_name + " data", unit="frame"):
        data = np.append(data, star_patches[str(field)], axis=0)

    # Create targets: 0 for galaxy, 1 for star
    targets = np.ones(len(data))
    targets[:gal_count] = 0

    return data, targets


def save_ml_data(train_data, train_targets, test_data, test_targets, val_data, val_targets, identifier):
    """ Save the machine learning data to disk. """
    ml_data_path = f"./ml_data/{identifier}/"
    create_dir(ml_data_path)

    print("Saving ML data...")
    np.save(ml_data_path + "train_data.npy", train_data)
    np.save(ml_data_path + "train_targets.npy", train_targets)
    np.save(ml_data_path + "test_data.npy", test_data)
    np.save(ml_data_path + "test_targets.npy", test_targets)
    np.save(ml_data_path + "val_data.npy", val_data)
    np.save(ml_data_path + "val_targets.npy", val_targets)
    


def produce_cnn_data(rerun, run, camcol, train_fields, test_fields, val_fields, patch_size, identifier):
    """ Generate and save CNN data (train, validation, and test sets). """
    path = f"./preprocessed_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    gal_patches = np.load(path + f"patches{patch_size}_gals.npy", allow_pickle=True).item()
    star_patches = np.load(path + f"patches{patch_size}_stars.npy", allow_pickle=True).item()

    print("Preparing ML data...")
    train_data, train_targets = produce_set_data(gal_patches, star_patches, train_fields, patch_size, set_name="training set")
    test_data, test_targets = produce_set_data(gal_patches, star_patches, test_fields, patch_size, set_name="test set")
    val_data, val_targets = produce_set_data(gal_patches, star_patches, val_fields, patch_size, set_name="validation set")

    save_ml_data(train_data, train_targets, test_data, test_targets, val_data, val_targets, identifier)
