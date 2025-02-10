import sys
from data_preprocessing_functions.download_sdss_data import download_catalog, download_frames
from data_preprocessing_functions.extract_coordinates import align_and_save_frames, save_target_coords
from data_preprocessing_functions.produce_patches import produce_patches_all_frames, produce_cnn_data
from plotting_functions.display_certain_patch import display_some_patches
from plotting_functions.plot_certain_gal_star import plot_a_star_and_a_galaxy
from plotting_functions.brightness import load_data_from_filepath, plot_mag_distribution, save_mag_distribution

def preprocesing_data(rerun, run, camcol, fields, patch_size, train_set, test_set, val_set):
    print(f"Downloading files for {len(fields)} fields:")
    # Download catalog files for stars and galaxies
    print(f"- Star and galaxy catalogs...", end=" ")
    download_catalog(rerun, run, camcol)
    print(f"done!")

    # Download the frames for each field in the specified list
    download_frames(rerun, run, camcol, fields)
    print(f"done!")

    print("File downloads successful.")

    # Aligning frames for different fields and bands
    print(f"Aligning frames for {len(fields)} fields:")
    align_and_save_frames(rerun, run, camcol, fields)
    print(f"All frames aligned!")

    # Extracting the coordinates of galaxies and coordinates from the catalogs (in ICRS) and then convert the ICRS into pixel coordinates in images
    print("Extracting galaxy and star coordinates...", end=" ")
    save_target_coords(rerun, run, camcol)
    print(f"done!")

    print("Field alignments and galaxy/star coordinate extraction successful.")

    identifier = f"patch_size{patch_size}_frames_10_ref_test"

    print(f"\nPreparing data for CNN model...\n")
    print(f"Training fields: {train_set}")
    print(f"Test fields: {test_set}\n")
    print(f"Validation fields: {val_set}")

    print(f"Extracting {patch_size}x{patch_size} patches from {len(fields)} fields:")
    produce_patches_all_frames(rerun, run, camcol, fields, patch_size)
    print("Patch extraction complete.\n")

    print("Generating CNN data...")
    produce_cnn_data(rerun, run, camcol, train_set, test_set, val_set, patch_size, identifier)
    print("CNN data preparation complete.\n")

def plotting_data(rerun, run, camcol, patch_size, field, band_num, gal_id, star_id):
    display_some_patches(rerun, run, camcol, patch_size, field, gal_id, star_id)

def brightness_analysis(rerun, run, camcol):
    mag_gal = load_data_from_filepath(rerun, run, camcol, 'galaxy', 'MODEL')
    mag_star = load_data_from_filepath(rerun, run, camcol, 'star', 'PSF')
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
        fields = [80, 103, 111, 120, 147, 174, 177, 214, 222, 228]

        # Define patch size
        patch_size = 25

        # Define field splits
        train_set = [80, 103, 111, 147, 177, 214, 222]
        test_set = [120, 228]
        val_set = [174]

        # Demonstration
        field = 103
        band_num = 1 # g-band for example
        gal_id = 100
        star_id = 160

        try:
            preprocesing_data(rerun, run, camcol, fields, patch_size, train_set, test_set, val_set)
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
