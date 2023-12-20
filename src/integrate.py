import cv2
import random
import numpy as np
import os
import astropy.io.fits as fits
from spacepy import pycdf


# /integrated作成

DATA_DIRECTORY = os.getenv(
    "DATA_DIRECTORY", "/Users/ogawa/Desktop/desktop_folders/data"
)


def integrate_cdf_and_fits(out_file_num, last_file_num):
    k, l = (random.randint(0, last_file_num - 1) for _ in range(2))  # 0~99に揃える
    random_cdf_path = f"{DATA_DIRECTORY}/out/random/cdf/{k}.cdf"
    random_fits_path = f"{DATA_DIRECTORY}/out/random/fits/{l}.fits"

    hdulist = fits.open(random_fits_path)

    cdf = np.array(pycdf.CDF(random_cdf_path)["data"])
    fits_ = np.array(hdulist[0].data)

    integrated_img = cdf + fits_

    cv2.imwrite(f"{DATA_DIRECTORY}/out/test/A/{out_file_num}.jpg", integrated_img)

    cv2.imwrite(f"{DATA_DIRECTORY}/out/test/B/{out_file_num}.jpg", cdf)
