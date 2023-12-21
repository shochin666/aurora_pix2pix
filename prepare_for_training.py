import os
import random
import argparse
import glob

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
    parser.add_argument("--cdf_date")
    parser.add_argument("--fits_date")
    parser.add_argument("--random_file_num")
    parser.add_argument("--integration_file_num")
    args = parser.parse_args()

    # CDF
    cdf_directory_path = os.path.join(
        META_DATA_DIRECTORY,
        "cdf",
        args.cdf_date,
    )
    random_file_num = int(args.random_file_num)
    integration_file_num = int(args.integration_file_num)
    epoch_second_mag, freq_second_mag = (2, 2)

    files = glob.glob(os.path.join(cdf_directory_path, "*"))
    cdf_title = files[0].split("/")[-1]
    cdf = CdfHandler(os.path.join(cdf_directory_path, cdf_title), "rr")
    cdf.resolution(epoch_second_mag, freq_second_mag)

    for i in range(random_file_num):
        n = random.randint(-100, 100)

        # targetのパラメーターは学習用に持ちるデータによって変える必要がある
        if n <= -50:
            cdf_target = (1850, 1400)
        elif -50 < n <= 0:
            cdf_target = (1950, 1600)
        elif 0 < n <= 50:
            cdf_target = (1450, 1500)
        elif 50 < n <= 100:
            cdf_target = (1450, 1700)

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
        data, epoch, freq = fits.cut_fits(target)

        m = random.randint(-100, 100)

        if m <= -50:
            save_as_fits(data[::-1, :], epoch, freq, random_save_path)
        elif -50 < m <= 0:
            save_as_fits(data[:, ::-1], epoch, freq, random_save_path)
        elif 0 < m <= 50:
            save_as_fits(data, epoch, freq, random_save_path)
        elif 50 < m <= 100:
            save_as_fits(
                data[::-1, ::-1], epoch, freq, random_save_path
            )  # out/random/fits以下に保存

    # integration
    for i in range(integration_file_num):
        integration_for_training(i, integration_file_num)

    for j in range(integration_file_num // 4):  # 25%分の枚数をテスト用にintegration
        integration_for_testing(j, integration_file_num)
