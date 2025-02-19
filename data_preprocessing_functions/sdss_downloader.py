import os
import requests
from tqdm import tqdm

class SDSSDownloader:
    BASE_URL = "https://data.sdss.org/sas/dr17/eboss/"
    URL_FRAME = "photoObj/frames/"
    URL_COORDS = "sweeps/dr13_final/"

    def __init__(self, rerun, run, camcol):
        """
        Initializes the SDSSDownloader with the specific rerun, run, and camcol.
        
        Args:
            rerun (int): The rerun number.
            run (int): The run number.
            camcol (int): The camera column number.
        """
        self.rerun = rerun
        self.run = run
        self.camcol = camcol
        self.data_dir = f"./raw_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
        self.create_dir(self.data_dir)

    def create_dir(self, path):
        """
        Helper function to create the directory if it doesn't exist.

        Args:
            path (str): The path of the directory to create.
        """
        os.makedirs(path, exist_ok=True)

    def request_and_write(self, url, filename):
        """
        Function to download a file from a URL and write it to a local path.

        Args:
            url (str): The URL of the file to download.
            filename (str): The local path where the file should be saved.
        """
        self.create_dir("/".join(filename.split('/')[:-1]))  # Ensure directory exists

        with requests.get(url=url, stream=True) as request:
            request.raise_for_status()  # Raise an error if the request failed
            with open(file=filename, mode='wb') as file:
                for chunk in request.iter_content(chunk_size=4096):
                    file.write(chunk)

    def construct_url(self, base, file_type, extension, field=None, band=None):
        """
        Constructs the appropriate URL for the catalog or frame data.

        Args:
            base (str): The base URL for the data.
            file_type (str): The type of file ("coords" for coordinates, "frame" for frames).
            extension (str): The file extension to use.
            field (int, optional): The field number for frame data. Defaults to None.
            band (str, optional): The filter band for the frame data. Defaults to None.

        Returns:
            str: The constructed URL.
        """
        if file_type == "coords":
            return f"{base}{self.URL_COORDS}{self.rerun}/calibObj-{str(self.run).zfill(6)}-{self.camcol}{extension}"
        elif file_type == "frame":
            end_url = f"-{str(self.run).zfill(6)}-{self.camcol}-{str(field).zfill(4)}"
            start_url = f"{base}{self.URL_FRAME}{self.rerun}/{self.run}/{self.camcol}/frame-"
            if band == "irg":
                return start_url + band + end_url + ".jpg"
            return start_url + band + end_url + ".fits.bz2"

    def download_catalog(self):
        """
        Download the galaxy and star catalogs (in FITS format) for a specific run, camcol, and rerun.
        """
        # Construct URLs for galaxy and star data
        gal_url = self.construct_url(self.BASE_URL, "coords", "-gal.fits.gz")
        star_url = self.construct_url(self.BASE_URL, "coords", "-star.fits.gz")

        # Download the galaxy and star catalogs
        self.request_and_write(gal_url, self.data_dir + "gal.fits.gz")
        self.request_and_write(star_url, self.data_dir + "star.fits.gz")

    def download_frame(self, field):
        """
        Download frame data for a specific field in a particular run, camcol, and rerun.

        Args:
            field (int): The field number.
        """
        # List of filter bands to download
        bands = ["i", "r", "g", "z", "u", "irg"]

        for band in bands:
            # Construct the URL for each band and download
            url = self.construct_url(self.BASE_URL, "frame", "", field, band)
            filename = f"field_{field}_{band}"

            if band == "irg":
                self.request_and_write(url, self.data_dir + filename + ".jpg")
            else:
                self.request_and_write(url, self.data_dir + filename + ".fits.bz2")

    def download_frames(self, fields):
        """
        Download frames for a list of fields.

        Args:
            fields (list): List of field numbers to download.
        """
        for field in tqdm(fields, desc="Downloading frames", unit="frame"):
            self.download_frame(field)
