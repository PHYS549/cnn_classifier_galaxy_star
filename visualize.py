import sys
from plotting_functions.display_certain_patch import display_some_patches
from plotting_functions.brightness import load_data_from_filepath, save_mag_distribution

def plotting_data(rerun, run, camcol, patch_size, field, band_num, gal_id, star_id):
    display_some_patches(rerun, run, camcol, patch_size, field, gal_id, star_id)

def brightness_analysis(rerun, run, camcol):
    mag_gal = load_data_from_filepath(rerun, run, camcol, 'galaxy', 'MODEL')
    mag_star = load_data_from_filepath(rerun, run, camcol, 'star', 'MODEL')
    mags = mag_gal, mag_star
    labels = ["galaxy","star"]
    mag_min = 0
    mag_max = 35
    save_mag_distribution(mags, labels, mag_min,mag_max)

def main():
    # Check if command line arguments are provided (for rerun, run, camcol, and fields)
    try:
        # Example parameters for a specific observation (can be customized)
        rerun = 301
        run = 8162
        camcol = 6

        # Define patch size
        patch_size = 25

        # Demonstration
        field = 103
        band_num = 1 # g-band for example
        gal_id = 100
        star_id = 160

        filter_brightness = True
        if filter_brightness:
            identifier = f"patch_size{patch_size}_filter_brightness_frames_10_ref_test"
        else:
            identifier = f"patch_size{patch_size}_no_filter_frames_10_ref_test"

        try:
            plotting_data(rerun, run, camcol, patch_size, field, band_num, gal_id, star_id)
            brightness_analysis(rerun, run, camcol)
            
        except Exception as e:
            # Print any errors that occur during the download process
            print(f"Downloads Failed. Error: {e}. Please try again.")

    except (IndexError, ValueError):
        # If arguments are invalid or missing, print usage and exit
        print("Run pip install . first")
        print("Or run python -m pip install . first")
        sys.exit(1)

if __name__ == "__main__":
    main()
