import cv2
import random
import numpy as np
import os
import astropy.io.fits as fits
from spacepy import pycdf


DATA_DIRECTORY = os.getenv(
    "DATA_DIRECTORY", "/Users/ogawa/Desktop/desktop_folders/data"
)


def integration_for_training(out_file_index, last_file_num):
    k, l = (random.randint(0, last_file_num - 1) for _ in range(2))
    random_cdf_path = f"{DATA_DIRECTORY}/out/random/cdf/{k}.cdf"
    random_fits_path = f"{DATA_DIRECTORY}/out/random/fits/{l}.fits"

    hdulist = fits.open(random_fits_path)

    cdf = np.array(pycdf.CDF(random_cdf_path)["data"])
    fits_ = np.array(hdulist[0].data)

    integrated_img = cdf + fits_

    cv2.imwrite(f"{DATA_DIRECTORY}/out/train/A/{out_file_index}.jpg", integrated_img)
    cv2.imwrite(f"{DATA_DIRECTORY}/out/train/B/{out_file_index}.jpg", cdf)


def integration_for_testing(out_file_index, last_file_num):
    k, l = (random.randint(0, last_file_num - 1) for _ in range(2))
    random_cdf_path = f"{DATA_DIRECTORY}/out/random/cdf/{k}.cdf"
    random_fits_path = f"{DATA_DIRECTORY}/out/random/fits/{l}.fits"

    hdulist = fits.open(random_fits_path)

    cdf = np.array(pycdf.CDF(random_cdf_path)["data"])
    fits_ = np.array(hdulist[0].data)

    integrated_img = cdf + fits_

    cv2.imwrite(f"{DATA_DIRECTORY}/out/test/A/{out_file_index}.jpg", integrated_img)
    cv2.imwrite(f"{DATA_DIRECTORY}/out/test/B/{out_file_index}.jpg", cdf)
