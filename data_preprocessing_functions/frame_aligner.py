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
import pandas as pd

class FrameAligner:
    def __init__(self, rerun, run, camcol):
        self.rerun = rerun
        self.run = run
        self.camcol = camcol
        self.preprocessed_path = f"./preprocessed_data/rerun_{self.rerun}/run_{self.run}/camcol_{self.camcol}/"
        os.makedirs(self.preprocessed_path, exist_ok=True)

    def align_and_save_frames(self, fields):
        """
        Aligns frames for multiple fields and saves the results to disk.
        Skips processing if aligned frames already exist.
        """
        aligned_frames_path = f"{self.preprocessed_path}frames_aligned.npy"
        
        # Check if the aligned frames file already exists
        if os.path.exists(aligned_frames_path):
            print(f"Aligned frames already exist at {aligned_frames_path}. Skipping alignment.")
            return

        print(f"- Loading frames...", end=" ")
        frames = self.load_frames(fields)
        print("done!")

        aligned_frames = {}

        # Use tqdm to display a progress bar as frames are aligned
        for frame in tqdm(frames, desc="Aligning frames", unit="frame"):
            aligned_frames[frame] = self.align_frame(frames[frame])

        # Save the aligned frames to disk
        np.save(aligned_frames_path, aligned_frames)
        print("Frames alignment completed and saved!")

    def load_frames(self, fields):
        """
        Loads the frames for the given rerun, run, camcol, and fields.
        """
        raw_path = f"./raw_data/rerun_{self.rerun}/run_{self.run}/camcol_{self.camcol}/"
        preprocessed_path = f"./preprocessed_data/rerun_{self.rerun}/run_{self.run}/camcol_{self.camcol}/"
        os.makedirs(preprocessed_path, exist_ok=True)

        frames_data_wcs = {}
        files = self.load_data_from_path(raw_path, file_type=".fits.bz2", decompress=True)
        
        for file in files:
            field = file[6:-11]
            if field not in frames_data_wcs:
                frames_data_wcs[field] = {}
            hdul = files[file]
            frames_data_wcs[field][file[-10]] = (np.array(hdul[0].data), WCS(hdul[0].header))

        assert frames_data_wcs, f"Files for fields {fields} not found in {raw_path}"
        return frames_data_wcs

    def align_frame(self, frames):
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

    def load_data_from_path(self, path, file_type=None, decompress=False):
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