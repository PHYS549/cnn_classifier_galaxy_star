import sys
from data_preprocessing_functions.sdss_downloader import SDSSDownloader
from data_preprocessing_functions.frame_aligner import FrameAligner
from data_preprocessing_functions.coord_extracter import CoordExtracter
from data_preprocessing_functions.sdss_patch_generator import SDSSPatchGenerator
from model_functions.cnn_model import cnn_train_model, cnn_test_model, visualize_feature_maps, identifier_model
import matplotlib.pyplot as plt

def identifier_data(rerun, run, camcol, fields, patch_size, filter_brightness, filter_mag):
    if filter_brightness:
        data_identifier = f"rerun{rerun}-run{run}-camcol{camcol}-fields{fields[0]}etc-patch_size{patch_size}-filter-mag-{filter_mag}"
    else:
        data_identifier = f"rerun{rerun}-run{run}-camcol{camcol}-fields{fields[0]}etc-patch_size{patch_size}-no_filter"
    return data_identifier

def identifier(rerun, run, camcol, fields, patch_size, filter_brightness, filter_mag, epochs=20, batch_size=32, pooling_scheme='AveragePooling', dropout_rate=0.5):
    if filter_brightness:
        data_identifier = f"rerun{rerun}-run{run}-camcol{camcol}-fields{fields[0]}etc-patch_size{patch_size}-filter-mag-{filter_mag}"
    else:
        data_identifier = f"rerun{rerun}-run{run}-camcol{camcol}-fields{fields[0]}etc-patch_size{patch_size}-no_filter"

    model_identifier = identifier_model(data_identifier, epochs, batch_size, pooling_scheme, dropout_rate)
    return data_identifier, model_identifier

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

def preprocesing_data(rerun, run, camcol, fields, patch_size, train_set, test_set, val_set, filter_brightness, filter_mag):
    frame_aligner = FrameAligner(rerun, run, camcol)
    # Aligning frames for different fields and bands
    print(f"Aligning frames for {len(fields)} fields:")
    frame_aligner.align_and_save_frames(fields)
    print(f"All frames aligned!")

    # Extracting the coordinates of galaxies and coordinates from the catalogs (in ICRS) and then convert the ICRS into pixel coordinates in images
    coord_extracter = CoordExtracter(rerun, run, camcol)
    print("Extracting galaxy and star coordinates...", end=" ")
    coord_extracter.filter_and_save_target_coords(filter_brightness, filter_mag)
    print(f"done!")

    print("Field alignments and galaxy/star coordinate extraction successful.")
    print(f"\nPreparing data for CNN model...\n")
    print(f"Training fields: {train_set}")
    print(f"Test fields: {test_set}\n")
    print(f"Validation fields: {val_set}")

    sdss_patch_generator = SDSSPatchGenerator(rerun, run, camcol, patch_size, filter_brightness, filter_mag)
    print(f"Extracting {patch_size}x{patch_size} patches from {len(fields)} fields:")
    sdss_patch_generator.produce_patches_all_frames(fields)
    print("Patch extraction complete.\n")

    print("Generating CNN data...")
    data_identifier = identifier_data(rerun, run, camcol, fields, patch_size, filter_brightness, filter_mag)
    sdss_patch_generator.produce_cnn_data(train_set, test_set, val_set, data_identifier)
    print("CNN data preparation complete.\n")

def process_dataset(
    rerun=301, 
    run=8162, 
    camcol=6, 
    fields=None, 
    patch_size=25, 
    field_splits=None, 
    filter_brightness=True, 
    bright_ones=True,
    filter_mag=23
):
    """
    Simplified function to download, preprocess, and identify data in one call.

    Args:
        rerun (int): Rerun number.
        run (int): Run number.
        camcol (int): Camera column.
        fields (list): List of field numbers.
        patch_size (int): Size of patches for preprocessing.
        field_splits (dict): Dictionary with keys 'train', 'test', 'val' and corresponding field lists.
        filter_brightness (bool): Whether to filter by brightness.
        bright_ones, filter_mag (bool): Whether to select bright objects.
    """
    if fields is None:
        fields = [80, 103, 111, 120, 147, 174, 177, 214, 222, 228]
    
    if field_splits is None:
        field_splits = {
            "train": [80, 103, 111, 147, 177, 214, 222],
            "test": [120, 228],
            "val": [174]
        }

    try:
        # Step 1: Download the data
        download_data(rerun, run, camcol, fields)
        
        # Step 2: Preprocess the data
        preprocesing_data(
            rerun, run, camcol, fields, patch_size, 
            field_splits["train"], field_splits["test"], field_splits["val"], 
            filter_brightness, filter_mag
        )
        
        # Step 3: Generate a data identifier
        data_identifier = identifier_data(rerun, run, camcol, fields, patch_size, filter_brightness, filter_mag)
        
        print(f"Data preparation complete. Identifier: {data_identifier}")
        return data_identifier

    except Exception as e:
        print(f"An error occurred: {e}")

def training_model(data_identifier, epochs=20, batch_size=32, pooling_scheme='AveragePooling', dropout_rate=0.5):
    model_identifier = identifier_model(data_identifier, epochs, batch_size, pooling_scheme, dropout_rate)
    cnn_train_model(data_identifier, epochs, batch_size, pooling_scheme, dropout_rate)
    return model_identifier

def main():
    # Check if command line arguments are provided (for rerun, run, camcol, and fields)
    try:
        Data8162_generic = process_dataset(
            rerun=301, 
            run=8162, 
            camcol=6, 
            fields=[80, 103, 111, 120, 147, 174, 177, 214, 222, 228, 116, 90], 
            patch_size=25, 
            field_splits = {
                "train": [80, 103, 111, 147, 177, 214, 222],
                "test": [120, 228],
                "val": [174, 116, 90]
            }, 
            filter_brightness=False, 
        )
        Magnitudes = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        Data8162_mags = []
        for i in range(len(Magnitudes)):
            Data8162_mags.append(process_dataset(
                rerun=301, 
                run=8162, 
                camcol=6, 
                fields=[80, 103, 111, 120, 147, 174, 177, 214, 222, 228, 116, 90], 
                patch_size=25, 
                field_splits = {
                    "train": [],
                    "test": [120, 228],
                    "val": []
                }, 
                filter_brightness=True, 
                bright_ones=True, 
                filter_mag=Magnitudes[i]
                )
            )
                
        Data7784_generic = process_dataset(
            rerun=301, 
            run=7784, 
            camcol=4, 
            fields=[25, 94, 146, 245, 351], 
            patch_size=25, 
            field_splits = {
                "train": [],
                "test": [25, 94, 146, 245, 351],
                "val": []
            }, 
            filter_brightness=False,
        )

        Model_generic = training_model(Data8162_generic, epochs=20, batch_size=32, pooling_scheme='AveragePooling', dropout_rate=0.5)
        Model_MaxPooling = training_model(Data8162_generic, epochs=20, batch_size=32, pooling_scheme='MaxPooling', dropout_rate=0.5)
        Model_highdropout = training_model(Data8162_generic, epochs=20, batch_size=32, pooling_scheme='AveragePooling', dropout_rate=0.75)
        Model_nodropout = training_model(Data8162_generic, epochs=20, batch_size=32, pooling_scheme='AveragePooling', dropout_rate=0.)

        # Define a helper function to save all accuracies in one file
        def save_all_accuracies_to_file(filename, accuracies):
            with open(filename, 'a') as file:  # Open in append mode ('a')
                for label, accuracy in accuracies.items():
                    file.write(f"{label} Accuracy: {accuracy}\n")
            print(f"All accuracies saved to {filename}")

        # Initialize a dictionary to hold all accuracies
        accuracies = {}

        # Run tests and store the results
        accuracies['generic_acc'] = cnn_test_model(Data8162_generic, Model_generic)

        for i in range(len(Magnitudes)):
            accuracies[f"mag-{Magnitudes[i]}_acc"] = cnn_test_model(Data8162_mags[i], Model_generic)
        
        accuracies['maxpooling_acc'] = cnn_test_model(Data8162_generic, Model_MaxPooling)
        accuracies['highdropout_acc'] = cnn_test_model(Data8162_generic, Model_highdropout)
        accuracies['nodropout_acc'] = cnn_test_model(Data8162_generic, Model_nodropout)
        accuracies['anotherset_acc'] = cnn_test_model(Data7784_generic, Model_generic)

        # Assuming Magnitudes and accuracies are already defined
        magnitudes = Magnitudes  # The list or array of magnitudes
        accuracy_values = [accuracies[f"mag-{mag}_acc"] for mag in magnitudes]  # Extract accuracies for each magnitude

        # Plotting the data
        plt.figure(figsize=(8, 6))
        plt.plot(magnitudes, accuracy_values, marker='o', linestyle='-', color='b')  # Change color or style as needed

        # Labeling the axes
        plt.xlabel('Magnitudes')
        plt.ylabel('Accuracies')
        plt.title('Magnitudes vs. Accuracies')

        # Optionally, add grid lines for better readability
        plt.grid(True)
        # Save the plot to the specified directory
        plt.savefig('result_plots/magnitudes_vs_accuracies.png')

        # Close the plot to free up memory
        plt.close()

        # Save all accuracies to the same file
        save_all_accuracies_to_file('result_plots/accuracies.txt', accuracies)
        visualize_feature_maps(Data8162_generic, Model_generic)

    except Exception as e:
        # Print any errors that occur during the download process
        print(f"Downloads Failed. Error: {e}. Please try again.")

if __name__ == "__main__":
    main()
