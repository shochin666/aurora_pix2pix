from spacepy import pycdf
import astropy.io.fits as fits


def save_as_cdf(data, epoch_list, freq_list, save_path):
    data = data[::-1]
    cdf = pycdf.CDF(
        save_path,
        "",
    )
    cdf["data"] = data
    cdf["Epoch"] = epoch_list
    cdf["Frequency"] = freq_list
    cdf.close()


def save_as_fits(data, epoch_list, freq_list, save_path):
    data = data[::-1].T
    hdu = fits.PrimaryHDU(data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(save_path, overwrite=True)
