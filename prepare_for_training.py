import os
import random
import argparse
import glob
import cv2

from src import (
    CdfHandler,
    FitsHandler,
    save_as_cdf,
    save_as_fits,
    integration_for_training,
    integration_for_testing,
)

if __name__ == "__main__":
    META_DATA_DIRECTORY = os.getenv(
        "DATA_DIRECTORY", "/Users/ogawa/Desktop/desktop_folders/data/"
    )

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cdf_date", type=str, default="")
    parser.add_argument("--cdf1_date", type=str, default="")
    parser.add_argument("--cdf2_date", type=str, default="")
    parser.add_argument("--fits_date", type=str, default="")
    parser.add_argument("--random_file_num", type=int)
    parser.add_argument("--integration_file_num", type=int)
    args = parser.parse_args()

    random_file_num = args.random_file_num
    integration_file_num = args.integration_file_num
    epoch_second_mag, freq_second_mag = (2, 2)
    cdf_combo = False

    # CDFとFITSのconcat
    if len(args.cdf_date) and len(args.fits_date):
        cdf_directory_path = os.path.join(
            META_DATA_DIRECTORY,
            "cdf",
            args.cdf_date,
        )
        files = glob.glob(os.path.join(cdf_directory_path, "*"))
        cdf_title = files[0].split("/")[-1]
        cdf = CdfHandler(os.path.join(cdf_directory_path, cdf_title), "rr")
        cdf.resolution(epoch_second_mag, freq_second_mag)

        for i in range(random_file_num):
            cdf_target = (1810, 1000)

            random_cdf_save_path = os.path.join(
                META_DATA_DIRECTORY,
                f"out/random/cdf/{i}.cdf",
            )
            data, epoch, freq = cdf.cut_cdf(cdf_target)

            m = random.randint(-100, 100)

            if m <= -50:
                save_as_cdf(data[::-1, :], epoch, freq, random_cdf_save_path)
            elif -50 < m <= 0:
                save_as_cdf(data[:, ::-1], epoch, freq, random_cdf_save_path)
            elif 0 < m <= 50:
                save_as_cdf(data, epoch, freq, random_cdf_save_path)
            elif 50 < m <= 100:
                save_as_cdf(
                    data[::-1, ::-1], epoch, freq, random_cdf_save_path
                )  # out/random/cdf以下に保存

        # FITS
        fits_directory_path = os.path.join(
            META_DATA_DIRECTORY,
            "fits",
            args.fits_date,
        )
        target = (600, 200)

        files = glob.glob(os.path.join(fits_directory_path, "*"))
        fits_title = files[0].split("/")[-1]
        fits = FitsHandler(os.path.join(fits_directory_path, fits_title))
        fits.resolution(epoch_second_mag, freq_second_mag)

        for i in range(random_file_num):
            random_save_path = os.path.join(
                META_DATA_DIRECTORY,
                f"out/random/fits/{i}.fits",
            )

            x_range, y_range = (1810, 1000)

            data, epoch, freq = fits.cut_fits(target, x_range, y_range)

            m = random.randint(-100, 100)

            if m <= -50:
                save_as_fits(data[::-1, :], epoch, freq, random_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise/{i}.jpg",
                    ),
                    data[::-1, :],
                )

            elif -50 < m <= 0:
                save_as_fits(data[:, ::-1], epoch, freq, random_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise/{i}.jpg",
                    ),
                    data[:, ::-1],
                )

            elif 0 < m <= 50:
                save_as_fits(data, epoch, freq, random_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise/{i}.jpg",
                    ),
                    data,
                )

            elif 50 < m <= 100:
                save_as_fits(data[::-1, ::-1], epoch, freq, random_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise/{i}.jpg",
                    ),
                    data,
                )

    # CDFを2つでconcat
    if len(args.cdf1_date) and len(args.cdf2_date):
        cdf1_target = (1810, 1000)
        cdf2_target = (2710, 1250)

        cdf_combo = True
        cdf1_directory_path = os.path.join(
            META_DATA_DIRECTORY,
            "cdf",
            args.cdf1_date,
        )
        cdf2_directory_path = os.path.join(
            META_DATA_DIRECTORY,
            "cdf",
            args.cdf2_date,
        )

        cdf1_files = glob.glob(os.path.join(cdf1_directory_path, "*"))
        cdf1_title = cdf1_files[0].split("/")[-1]
        cdf1 = CdfHandler(os.path.join(cdf1_directory_path, cdf1_title), "rr")
        cdf1.resolution(epoch_second_mag, freq_second_mag)

        cdf2_files = glob.glob(os.path.join(cdf2_directory_path, "*"))
        cdf2_title = cdf2_files[0].split("/")[-1]
        cdf2 = CdfHandler(os.path.join(cdf2_directory_path, cdf2_title), "rr")
        cdf2.resolution(epoch_second_mag, freq_second_mag)

        for i in range(random_file_num):
            random_cdf1_save_path = os.path.join(
                META_DATA_DIRECTORY,
                f"out/random/cdf1/{i}.cdf",
            )

            x_range, y_range = (180, 650)

            data, epoch, freq = cdf1.cut_cdf(cdf1_target, x_range, y_range)

            m = random.randint(-100, 100)

            if m <= -50:
                save_as_cdf(data[::-1, :], epoch, freq, random_cdf1_save_path)
            elif -50 < m <= 0:
                save_as_cdf(data[:, ::-1], epoch, freq, random_cdf1_save_path)
            elif 0 < m <= 50:
                save_as_cdf(data, epoch, freq, random_cdf1_save_path)
            elif 50 < m <= 100:
                save_as_cdf(data[::-1, ::-1], epoch, freq, random_cdf1_save_path)

        for i in range(random_file_num):
            random_cdf2_save_path = os.path.join(
                META_DATA_DIRECTORY,
                f"out/random/cdf2/{i}.cdf",
            )

            x_range, y_range = (180, 60)

            data, epoch, freq = cdf2.cut_cdf(cdf2_target, x_range, y_range)

            m = random.randint(-100, 100)

            if m <= -50:
                save_as_cdf(data[::-1, :], epoch, freq, random_cdf2_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise/{i}.jpg",
                    ),
                    data[::-1, :],
                )

            elif -50 < m <= 0:
                save_as_cdf(data[:, ::-1], epoch, freq, random_cdf2_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise/{i}.jpg",
                    ),
                    data[:, ::-1],
                )

            elif 0 < m <= 50:
                save_as_cdf(data, epoch, freq, random_cdf2_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise/{i}.jpg",
                    ),
                    data,
                )

            elif 50 < m <= 100:
                save_as_cdf(data[::-1, ::-1], epoch, freq, random_cdf2_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise/{i}.jpg",
                    ),
                    data[::-1, ::-1],
                )

    # integration
    for i in range(integration_file_num * 3 // 4):
        integration_for_training(i, integration_file_num, cdf_combo)

    for i in range(integration_file_num // 4):  # 25%の枚数をテスト用にintegration
        integration_for_testing(i, integration_file_num, cdf_combo)
