import os
from spacepy import pycdf
from src import (
    CdfHandler,
    FitsHandler,
    save_as_cdf,
    save_as_fits,
    show_fullimg,
    show_img,
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
    parser.add_argument("--date", type=str)  # example)19910103
    parser.add_argument("--extension", type=str)  # fits or cdf
    parser.add_argument(
        "--filter_depth", type=int, default=150
    )  # 背景ノイズを除去するためのフィルタの高さ(0~255で指定)
    args = parser.parse_args()

    epoch_second_mag, freq_second_mag = (2, 2)
    filter_depth = args.filter_depth
    extension = args.extension
    data_directory_path = os.path.join(
        META_DATA_DIRECTORY,
        f"{args.extension}",
        args.date,
    )

    # FITS
    if extension == "fits":
        fits_directory_path = os.path.join(
            META_DATA_DIRECTORY,
            "fits",
            args.date,
        )
        files = glob.glob(os.path.join(fits_directory_path, "*"))
        fits_title = files[0].split("/")[-1]
        fits = FitsHandler(os.path.join(fits_directory_path, fits_title))

        # preprocess -> predict -> reconstruct -> save
        preprocessed = Preprocess(fits, epoch_second_mag, freq_second_mag, extension)
        preprocessed.optimize_data_size()
        preprocessed.separate_data()
        preprocessed.predict_and_concatenate(filter_depth)
        preprocessed.save()

    # CDF
    if extension == "cdf":
        cdf_directory_path = os.path.join(
            META_DATA_DIRECTORY,
            "cdf",
            args.date,
        )
        files = glob.glob(os.path.join(cdf_directory_path, "*"))
        cdf_title = files[0].split("/")[-1]
        cdf = CdfHandler(os.path.join(cdf_directory_path, cdf_title))

        preprocessed = Preprocess(cdf, epoch_second_mag, freq_second_mag, extension)
        preprocessed.optimize_data_size()
        preprocessed.separate_data()
        preprocessed.predict_and_concatenate(filter_depth)
        preprocessed.save()
