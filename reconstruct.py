import os
from spacepy import pycdf
from src import (
    CdfHandler,
    FitsHandler,
    save_as_cdf,
    save_as_fits,
    show_fullimg,
    show_img,
    integrate_cdf_and_fits,
    Preprocess,
)

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import sys
import argparse
import glob


if __name__ == "__main__":
    META_DATA_DIRECTORY = os.getenv(
        "DATA_DIRECTORY", "/Users/ogawa/Desktop/desktop_folders/data/"
    )
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--date")  # example)19910103
    parser.add_argument("--extension")  # fits or cdf
    parser.add_argument("--filter_depth", default=150)  # 背景ノイズを除去するためのフィルタの高さ(0~255で指定)
    args = parser.parse_args()

    epoch_second_mag, freq_second_mag = (2, 2)
    filter_depth = int(args.filter_depth)
    data_directory_path = os.path.join(
        META_DATA_DIRECTORY,
        f"{args.extension}",
        args.date,
    )

    # FITS
    if parser.extension == "fits":
        fits_directory_path = os.path.join(
            META_DATA_DIRECTORY,
            "fits",
            args.fits_date,
        )
        files = glob.glob(os.path.join(fits_directory_path, "*"))
        fits_title = files[0].split("/")[-1]
        fits = FitsHandler(os.path.join(fits_directory_path, fits_title))

        # preprocess -> predict -> reconstruct -> save
        preprocessed = Preprocess(fits, epoch_second_mag, freq_second_mag)
        preprocessed.optimize_fits_size()
        preprocessed.separate_fits()
        preprocessed.predict_and_concatenate(filter_depth)
        preprocessed.save()

    # CDF
