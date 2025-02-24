import os
import bz2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.units import deg
from reproject import reproject_exact
from tqdm import tqdm
from itertools import cycle

# Asinh Magnitude reference: https://www.sdss3.org/dr8/algorithms/magnitudes.php

def load_data_from_filepath(rerun, run, camcol, star_or_galaxy, flux_type):
    """
    Helper function to load data from the specified filepath.
    Optionally decompresses and filters files by type.
    """
    if star_or_galaxy=='galaxy':
        filepath = f"./raw_data/rerun_{rerun}/run_{run}/camcol_{camcol}/gal.fits.gz"
    elif star_or_galaxy=='star':
        filepath = f"./raw_data/rerun_{rerun}/run_{run}/camcol_{camcol}/star.fits.gz"

    hdul = fits.open(filepath)
    fits.open(f"./raw_data/rerun_{rerun}/run_{run}/camcol_{camcol}/field_111_g.fits.bz2").info()
    print(fits.open(f"./raw_data/rerun_{rerun}/run_{run}/camcol_{camcol}/field_111_g.fits.bz2")[1].header)
    
    Fluxes = np.array(hdul[1].data[flux_type + "FLUX"]) # Unit: Nanomaggy
    Mag = Asinh_Magnitude(Fluxes)

    return Mag

def plot_mag_distribution(mags, labels, min_mag, max_mag, band_num, save_or_not = True):
    
    # Plot histogram
    plt.figure(figsize=(15, 12))  # Set the figure size
    # Create a color cycle iterator (will automatically cycle through a list of colors)
    color_cycle = cycle(['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray'])
    band = ['u', 'g', 'r', 'i','z']
    for i in range(len(mags)):
        plt.subplot(len(mags),1,i+1)
        color = next(color_cycle)  # Get the next color from the color cycle
        plt.hist(mags[i][:, band_num], bins=50, edgecolor='black', label=labels[i], color=color)  # 50 bins for the histogram
        plt.xlim(min_mag, max_mag)
        plt.legend()
        plt.title(band[band_num] + '-band Magnitude Distribution Profile')  # Title of the plot
        plt.xlabel('Magnitude')  # Label for the x-axis
        plt.ylabel('Number')  # Label for the y-axis
        plt.grid(True)  # Show grid lines
    
    if save_or_not==False:
        plt.show()
    else:
        
        plt.savefig('brightness_analysis/MagDistribution_'+band[band_num]+'band.png')

def save_mag_distribution(mags, labels, min_mag, max_mag):
    os.makedirs('brightness_analysis', exist_ok=True)
    for band_num in range(5):
        plot_mag_distribution(mags, labels, min_mag, max_mag, band_num, save_or_not = True)

def Asinh_Magnitude(flux):
    f_f0 = flux * 1e-9 # Convert from nanomaggy to maggy
    b = [1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10]
    logb = np.tile(np.log(b), (len(flux), 1))
    return -2.5 / np.log(10) * (np.arcsinh(f_f0/2/b) + logb)
