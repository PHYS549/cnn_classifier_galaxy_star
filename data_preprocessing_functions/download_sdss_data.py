import os
import requests
from tqdm import tqdm

# Base URLs for accessing data from the SDSS servers
BASE_URL = "https://data.sdss.org/sas/dr17/eboss/"
URL_FRAME = "photoObj/frames/"
URL_COORDS = "sweeps/dr13_final/"

def create_dir(path):
    """
    Helper function to create the directory if it doesn't exist.
    
    Args:
        path (str): The path of the directory to create.
    """
    os.makedirs(path, exist_ok=True)


def request_and_write(url, filename):
    """
    Function to download a file from a URL and write it to a local path.
    
    Args:
        url (str): The URL of the file to download.
        filename (str): The local path where the file should be saved.
    """
    create_dir("/".join(filename.split('/')[:-1]))  # Ensure directory exists

    with requests.get(url=url, stream=True) as request:
        request.raise_for_status()  # Raise an error if the request failed
        with open(file=filename, mode='wb') as file:
            for chunk in request.iter_content(chunk_size=4096):
                file.write(chunk)


def construct_url(base, rerun, run, camcol, file_type, extension, field=None, band=None):
    """
    Constructs the appropriate URL for the catalog or frame data.
    
    Args:
        base (str): The base URL for the data.
        rerun (int): The rerun number.
        run (int): The run number.
        camcol (int): The camera column number.
        file_type (str): The type of file ("coords" for coordinates, "frame" for frames).
        extension (str): The file extension to use.
        field (int, optional): The field number for frame data. Defaults to None.
        band (str, optional): The filter band for the frame data. Defaults to None.
    
    Returns:
        str: The constructed URL.
    """
    if file_type == "coords":
        return f"{base}{URL_COORDS}{rerun}/calibObj-{str(run).zfill(6)}-{camcol}{extension}"
    elif file_type == "frame":
        end_url = f"-{str(run).zfill(6)}-{camcol}-{str(field).zfill(4)}"
        start_url = f"{base}{URL_FRAME}{rerun}/{run}/{camcol}/frame-"
        if band == "irg":
            return start_url + band + end_url + ".jpg"
        return start_url + band + end_url + ".fits.bz2"


def download_catalog(rerun, run, camcol):
    """
    Download the galaxy and star catalogs (in FITS format) for a specific run, camcol, and rerun.
    
    Args:
        rerun (int): The rerun number.
        run (int): The run number.
        camcol (int): The camera column number.
    """
    # Directory to store the data
    data_dir = f"./raw_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    create_dir(data_dir)

    # Construct URLs for galaxy and star data
    gal_url = construct_url(BASE_URL, rerun, run, camcol, "coords", "-gal.fits.gz")
    star_url = construct_url(BASE_URL, rerun, run, camcol, "coords", "-star.fits.gz")

    # Download the galaxy and star catalogs
    request_and_write(gal_url, data_dir + "gal.fits.gz")
    request_and_write(star_url, data_dir + "star.fits.gz")


def download_frame(rerun, run, camcol, field):
    """
    Download frame data for a specific field in a particular run, camcol, and rerun.
    
    Args:
        rerun (int): The rerun number.
        run (int): The run number.
        camcol (int): The camera column number.
        field (int): The field number.
    """
    # Directory to store the frame data
    data_dir = f"./raw_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    create_dir(data_dir)

    # List of filter bands to download
    bands = ["i", "r", "g", "z", "u", "irg"]
    
    for band in bands:
        # Construct the URL for each band and download
        url = construct_url(BASE_URL, rerun, run, camcol, "frame", "", field, band)
        filename = f"field_{field}_{band}"
        
        if band == "irg":
            request_and_write(url, data_dir + filename + ".jpg")
        else:
            request_and_write(url, data_dir + filename + ".fits.bz2")

def download_frames(rerun, run, camcol, fields):
    for field in tqdm(fields, desc="Downloading frames", unit="frame"):
        download_frame(rerun, run, camcol, field)
