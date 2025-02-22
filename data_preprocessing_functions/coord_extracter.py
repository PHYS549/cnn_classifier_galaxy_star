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

class CoordExtracter:
    def __init__(self, rerun, run, camcol):
        self.rerun = rerun
        self.run = run
        self.camcol = camcol
        self.preprocessed_path = f"./preprocessed_data/rerun_{self.rerun}/run_{self.run}/camcol_{self.camcol}/"
        os.makedirs(self.preprocessed_path, exist_ok=True)

    def filter_and_save_target_coords(self, filter_brightness):
        """
        Extracts and saves the target coordinates (for galaxies and stars) for all fields.
        """
        self.filter_brightness = filter_brightness
        raw_path = f"./raw_data/rerun_{self.rerun}/run_{self.run}/camcol_{self.camcol}/"
        frames, gals, stars = self.load_star_gal_data(raw_path)

        gal_targets, star_targets = {}, {}
        gal_num = 0
        star_num = 0
        # Use tqdm to display a progress bar as we loop through the frames
        for frame_num in tqdm(frames, desc="Processing frames", unit="frame"):
            gal_targets[frame_num], star_targets[frame_num] = self.target_coords_for_field(frames[frame_num], int(frame_num), gals, stars)
            gal_num = gal_num + len(gal_targets[frame_num])
            star_num = star_num + len(star_targets[frame_num])
        
        # Save the targets to files after processing
        np.save(f"{self.preprocessed_path}target_gals", gal_targets)
        np.save(f"{self.preprocessed_path}target_stars", star_targets)
        print("\nTarget coordinates saved!")
        print("Star number:" + str(star_num))
        print("Galaxy number:" + str(gal_num))

    def load_star_gal_data(self, raw_path):
        """
        Loads the star and galaxy data from the specified path.
        """
        preprocessed_path = f"./preprocessed_data/rerun_{self.rerun}/run_{self.run}/camcol_{self.camcol}/"
        frames = np.load(preprocessed_path + "frames_aligned.npy", allow_pickle=True).item()
        gals = fits.open(raw_path + "gal.fits.gz")[1]
        stars = fits.open(raw_path + "star.fits.gz")[1]

        return frames, gals, stars

    def target_coords_for_field(self, frame, field, gals, stars):
        """
        Retrieves the pixel coordinates for galaxies and stars for a specific field.
        """
        wcs = frame[1]
        _, height_ref, width_ref = frame[0].shape
        
        gal_pixels = self.filtering_and_extract_pixel(gals.data, wcs, field, height_ref, width_ref)
        star_pixels = self.filtering_and_extract_pixel(stars.data, wcs, field, height_ref, width_ref)
        
        return np.array(gal_pixels), np.array(star_pixels)

    def Asinh_Magnitude(self, flux, b):
        """
        Calculate the asinh magnitude for a 1D flux array given a band-specific b parameter.
        """
        f_f0 = flux * 1e-9  # Convert from nanomaggy to maggy
        logb = np.log(b)
        return -2.5 / np.log(10) * (np.arcsinh(f_f0 / (2 * b)) + logb)

    def filtering_and_extract_pixel(self, data, wcs, field_ref, height_ref, width_ref):
        # Convert FITS data columns to native byte order if necessary
        field = data['FIELD'].byteswap().newbyteorder() if data['FIELD'].dtype.byteorder == '>' else data['FIELD']
        ra = data['RA'].byteswap().newbyteorder() if data['RA'].dtype.byteorder == '>' else data['RA']
        dec = data['DEC'].byteswap().newbyteorder() if data['DEC'].dtype.byteorder == '>' else data['DEC']
        modelflux = data['MODELFLUX'].byteswap().newbyteorder() if data['MODELFLUX'].dtype.byteorder == '>' else data['MODELFLUX']
        
        # Convert to pandas DataFrame with only the columns of interest
        df = pd.DataFrame({
            'FIELD': field,
            'RA': ra,
            'DEC': dec,
            'fluxu': modelflux[:, 0],
            'fluxg': modelflux[:, 1],
            'fluxr': modelflux[:, 2],
            'fluxi': modelflux[:, 3],
            'fluxz': modelflux[:, 4]
        })
        
        # Filter by FIELD and explicitly make a copy
        filtered_df = df[
            (df['FIELD'] == field_ref) |
            (df['FIELD'] == field_ref + 1) |
            (df['FIELD'] == field_ref - 1)
        ].copy()

        # Build SkyCoord object using RA and DEC values
        coords = SkyCoord(filtered_df['RA'].values, filtered_df['DEC'].values, unit='deg')

        # Convert world coordinates to pixel coordinates
        x, y = wcs.world_to_pixel(coords)

        # Save the pixel coordinates into new columns 'X' and 'Y' using .loc for clarity
        filtered_df.loc[:, 'X'] = x
        filtered_df.loc[:, 'Y'] = y

        # Filter by X and Y boundaries
        filtered_df = filtered_df[(filtered_df['X'] > -0.5) & (filtered_df['X'] < width_ref + 0.5)]
        filtered_df = filtered_df[(filtered_df['Y'] > -0.5) & (filtered_df['Y'] < height_ref + 0.5)]
        
        if self.filter_brightness:
            # Define the b parameters for each band
            b_values = {
                'fluxu': 1.4e-10,
                'fluxg': 0.9e-10,
                'fluxr': 1.2e-10,
                'fluxi': 1.8e-10,
                'fluxz': 7.4e-10
            }
            
            # Calculate magnitude for each band and store in new columns
            for band in ['fluxu', 'fluxg', 'fluxr', 'fluxi', 'fluxz']:
                filtered_df.loc[:, 'MAG_' + band[-1]] = self.Asinh_Magnitude(filtered_df[band].values, b_values[band])
            
            filtered_df = filtered_df[(filtered_df['MAG_g'] <23)]
            filtered_df = filtered_df[(filtered_df['MAG_r'] <22.5)]
            filtered_df = filtered_df[(filtered_df['MAG_i'] <23)]
            filtered_df = filtered_df[(filtered_df['MAG_z'] <23)]
            
        # Return only the (X, Y) pairs as a NumPy array (or modify as needed)
        return filtered_df[['X', 'Y']].to_numpy()
