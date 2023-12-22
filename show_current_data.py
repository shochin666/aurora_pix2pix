import os
import argparse
import glob

from spacepy import pycdf
from src import (
    CdfHandler,
    FitsHandler,
    show_fullimg,
)


if __name__ == "__main__":
    # parser init
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--date", type=str)  # example)19910103
    parser.add_argument("--extension", type=str)  # fits or cdf
    parser.add_argument("--highlight", type=bool)  # true or false
    args = parser.parse_args()

    epoch_second_mag, freq_second_mag = (2, 2)
    META_DATA_DIRECTORY = os.getenv(
        "DATA_DIRECTORY", "/Users/ogawa/Desktop/desktop_folders/data/"
    )

    data_directory_path = os.path.join(
        META_DATA_DIRECTORY,
        f"{args.extension}",
        args.date,
    )  #  example)/Users/ogawa/Desktop/desktop_folders/data/cdf/19910103

    if args.extension == "cdf":
        highlight_target = (1200, 500)
        highlight_size = (256, 256)
        files = glob.glob(os.path.join(data_directory_path, "*"))
        cdf_title = files[0].split("/")[-1]
        cdf = CdfHandler(os.path.join(data_directory_path, cdf_title), "rr")
        cdf_changed_resolution = cdf.resolution(epoch_second_mag, freq_second_mag)

        if args.highlight == "true":
            show_fullimg(
                cdf_title,
                highlight_target,
                highlight_size,
                cdf_changed_resolution,
                cdf.epoch_new,
                cdf.freq_new,
                target_highlight=True,
            )
        else:
            show_fullimg(
                cdf_title,
                highlight_target,
                highlight_size,
                cdf_changed_resolution,
                cdf.epoch_new,
                cdf.freq_new,
            )

    elif args.extension == "fits":
        highlight_target = (1200, 500)
        highlight_size = (256, 256)
        files = glob.glob(os.path.join(data_directory_path, "*"))
        fits_title = files[0].split("/")[-1]
        fits = FitsHandler(os.path.join(data_directory_path, fits_title))
        fits_changed_resolution = fits.resolution(epoch_second_mag, freq_second_mag)

        if args.highlight == "true":
            show_fullimg(
                fits_title,
                highlight_target,
                highlight_size,
                fits_changed_resolution,
                fits.epoch_new,
                fits.freq_new,
                target_highlight=True,
            )
        else:
            show_fullimg(
                fits_title,
                highlight_target,
                highlight_size,
                fits_changed_resolution,
                fits.epoch_new,
                fits.freq_new,
            )
