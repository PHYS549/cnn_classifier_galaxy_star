from astropy.io import fits
import numpy as np

def write_fits(data, filename):
    # Step 1: Create a PrimaryHDU object with the data
    hdu = fits.PrimaryHDU(data)

    # Step 2: Add metadata to the header
    hdu.header['OBJECT'] = 'Random Data'
    hdu.header['AUTHOR'] = 'Your Name'
    hdu.header['COMMENT'] = 'This is an example FITS file'

    # Step 3: Create an HDUList and write to a FITS file
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename, overwrite=True)

    print("FITS file " +str(filename)+ " created successfully.")

    # Step 4: Verify the FITS file
    with fits.open(filename) as hdul:
        hdul.info()
        print("\nHeader information:")
        print(hdul[0].header)
        print("\nData preview (first 5 rows):")
        print(hdul[0].data[:5, :5])

if __name__ == '__main__':
    data = np.random.random((100, 100))
    write_fits(data, 'test.fits')

