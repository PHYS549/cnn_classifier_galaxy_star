import sys
from data_preprocessing_functions.sdss_downloader import SDSSDownloader
from data_preprocessing_functions.frame_aligner import FrameAligner
from data_preprocessing_functions.coord_extracter import CoordExtracter
from data_preprocessing_functions.sdss_patch_generator import SDSSPatchGenerator
from model_functions.cnn_model import cnn_train_model, cnn_test_model, visualize_feature_maps, identifier


def download_data(rerun, run, camcol, fields):
    sdss_downloader = SDSSDownloader(rerun, run, camcol)
    print(f"Downloading files for {len(fields)} fields:")
    # Download catalog files for stars and galaxies
    print(f"- Star and galaxy catalogs...", end=" ")
    sdss_downloader.download_catalog()
    print(f"done!")

    # Download the frames for each field in the specified list
    sdss_downloader.download_frames(fields)
    print(f"done!")

    print("File downloads successful.")

def preprocesing_data(rerun, run, camcol, fields, patch_size, train_set, test_set, val_set, filter_brightness):
    frame_aligner = FrameAligner(rerun, run, camcol)
    # Aligning frames for different fields and bands
    print(f"Aligning frames for {len(fields)} fields:")
    frame_aligner.align_and_save_frames(fields)
    print(f"All frames aligned!")

    # Extracting the coordinates of galaxies and coordinates from the catalogs (in ICRS) and then convert the ICRS into pixel coordinates in images
    coord_extracter = CoordExtracter(rerun, run, camcol)
    print("Extracting galaxy and star coordinates...", end=" ")
    coord_extracter.filter_and_save_target_coords(filter_brightness)
    print(f"done!")

    print("Field alignments and galaxy/star coordinate extraction successful.")

    if filter_brightness:
        data_identifier = f"patch_size{patch_size}_filter_brightness_frames_10_ref_test"
    else:
        data_identifier = f"patch_size{patch_size}_no_filter_frames_10_ref_test"

    print(f"\nPreparing data for CNN model...\n")
    print(f"Training fields: {train_set}")
    print(f"Test fields: {test_set}\n")
    print(f"Validation fields: {val_set}")

    sdss_patch_generator = SDSSPatchGenerator(rerun, run, camcol, patch_size)
    print(f"Extracting {patch_size}x{patch_size} patches from {len(fields)} fields:")
    sdss_patch_generator.produce_patches_all_frames(fields)
    print("Patch extraction complete.\n")

    print("Generating CNN data...")
    sdss_patch_generator.produce_cnn_data(train_set, test_set, val_set, data_identifier)
    print("CNN data preparation complete.\n")

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

        filter_brightness = True

        data_identifier, model_identifier = identifier(filter_brightness, patch_size, epochs=20, batch_size=32, pooling_scheme='AveragePooling', dropout_rate=0.5)

        try:
            download_data(rerun, run, camcol, fields)
            preprocesing_data(rerun, run, camcol, fields, patch_size, train_set, test_set, val_set, filter_brightness)
            cnn_train_model(data_identifier, epochs=20, batch_size=32, pooling_scheme='AveragePooling', dropout_rate=0.5)
            cnn_test_model(data_identifier, model_identifier)
            
            visualize_feature_maps(model_identifier)

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
