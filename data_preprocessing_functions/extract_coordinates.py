import os
import bz2
import numpy as np
import shutil
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.units import deg
from reproject import reproject_exact
from tqdm import tqdm

def align_and_save_frames(rerun, run, camcol, fields):
    """
    Aligns frames for multiple fields and saves the results to disk.
    """
    preprocessed_path = f"./preprocessed_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    os.makedirs(preprocessed_path, exist_ok=True)

    print(f"- Loading frames...", end=" ")
    frames = load_frames(rerun, run, camcol, fields)
    print("done!")

    aligned_frames = {}
    
    # Use tqdm to display a progress bar as frames are aligned
    for frame in tqdm(frames, desc="Aligning frames", unit="frame"):
        aligned_frames[frame] = align_frame(frames[frame])#main: remove[0]

    # Save the aligned frames to disk
    np.save(f"{preprocessed_path}frames_aligned", aligned_frames)
    print("Frames alignment completed and saved!")


def load_frames(rerun, run, camcol, fields):
    """
    Loads the frames for the given rerun, run, camcol, and fields.
    """
    raw_path = f"./raw_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    preprocessed_path = f"./preprocessed_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    os.makedirs(preprocessed_path, exist_ok=True)

    frames_data_wcs = {}
    files = load_data_from_path(raw_path, file_type=".fits.bz2", decompress=True)
    
    for file in files:
        field = file[6:-11]
        if field not in frames_data_wcs:
            frames_data_wcs[field] = {}
        hdul = files[file]
        frames_data_wcs[field][file[-10]] = (np.array(hdul[0].data), WCS(hdul[0].header))

    assert frames_data_wcs, f"Files for fields {fields} not found in {raw_path}"
    return frames_data_wcs



def align_frame(frames):
    """
    Aligns frames for different bands (i, r, g, z, u) using the reference band 'r'.
    """
    ref_band = "r"
    ref_data, ref_wcs = frames[ref_band]

    aligned_frame = np.zeros((5, *ref_data.shape))
    
    for idx, band in enumerate(["i", "r", "g", "z", "u"]):
        if band == ref_band:
            aligned_frame[idx], _ = frames[band]
        else:
            aligned_frame[idx] = reproject_exact(frames[band], ref_wcs, shape_out=ref_data.shape, return_footprint=False)
    
    return np.nan_to_num(aligned_frame), ref_wcs



def load_data_from_path(path, file_type=None, decompress=False):
    """
    Helper function to load data from the specified path.
    Optionally decompresses and filters files by type.
    """
    assert os.path.exists(path), f"Path {path} does not exist"
    files = os.listdir(path)
    assert files, f"No files found in {path}"
    
    if file_type:
        files = [f for f in files if f.endswith(file_type)]
    
    data = {}
    for file in files:
        if decompress:
            bz2_file = bz2.BZ2File(path + file)
            hdul = fits.open(bz2_file)
        else:
            hdul = fits.open(path + file)
        
        data[file] = hdul
    return data


def save_target_coords(rerun, run, camcol):
    """
    Extracts and saves the target coordinates (for galaxies and stars) for all fields.
    """
    preprocessed_path = f"./preprocessed_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    raw_path = f"./raw_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    frames, gals, stars = load_star_gal_data(raw_path, preprocessed_path)

    gal_targets, star_targets = {}, {}

    # Use tqdm to display a progress bar as we loop through the frames
    for frame_num in tqdm(frames, desc="Processing frames", unit="frame"):
        gal_targets[frame_num], star_targets[frame_num] = target_coords_for_field(frames[frame_num], int(frame_num), gals, stars)
    
    # Save the targets to files after processing
    np.save(f"{preprocessed_path}target_gals", gal_targets)
    np.save(f"{preprocessed_path}target_stars", star_targets)
    print("\nTarget coordinates saved!")


def load_star_gal_data(raw_path, preprocessed_path):
    """
    Loads the star and galaxy data from the specified path.
    """

    frames = np.load(preprocessed_path + "frames_aligned.npy", allow_pickle=True).item()
    gals = fits.open(raw_path + "gal.fits.gz")[1]
    stars = fits.open(raw_path + "star.fits.gz")[1]

    return frames, gals, stars


def target_coords_for_field(frame, field, gals, stars):
    """
    Retrieves the pixel coordinates for galaxies and stars for a specific field.
    """
    wcs = frame[1]
    _, height_ref, width_ref = frame[0].shape
    

    gal_pixels = extract_field_coords(gals.data, wcs, field, height_ref, width_ref)
    star_pixels = extract_field_coords(stars.data, wcs, field, height_ref, width_ref)
    
    return np.array(gal_pixels), np.array(star_pixels)


def extract_field_coords(data, wcs, field, height_ref, width_ref):
    """
    Extracts pixel coordinates for a specific field from galaxy or star data.
    """
    pixels = []

    for entry in data:
        if field - 1 <= entry["FIELD"] <= field + 1:
            coords = SkyCoord(entry["RA"], entry["DEC"], unit=deg)
            width, height = wcs.world_to_pixel(coords)

            if -0.5 < width < width_ref + 0.5 and -0.5 < height < height_ref + 0.5:
                pixels.append(np.array([width, height]))
    
    return pixels

