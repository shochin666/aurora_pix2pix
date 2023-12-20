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
import astropy.io.fits as fits_
import random
import cv2
import sys


def main():
    outfile_num = 200
    last_file_num = 100
    integrated_file_num = 100
    target = (1200, 500)  # CDF
    # FITS
    size = (256, 256)
    epoch_second_mag, freq_second_mag = (2, 2)  # 要検討
    DATA_DIRECTORY = os.getenv(
        "DATA_DIRECTORY", "/Users/ogawa/Desktop/desktop_folders/data"
    )  # デフォルト値を設定しないとNoneになってしまう

    original_fits_path = os.path.join(
        DATA_DIRECTORY, "fits/JUPITER_TRACKING_20201216_100033_2.spectra.fits"
    )

    fits = FitsHandler(original_fits_path)
    preprocessed = Preprocess(fits, epoch_second_mag, freq_second_mag)
    preprocessed.optimize_fits_size()
    preprocessed.separate_fits()
    preprocessed.predict_and_concatenate()
    preprocessed.save()

    # if DATA_DIRECTORY:
    #     # resolution(基本的に実行)
    #     path = os.path.join(
    #         DATA_DIRECTORY,
    #         "cdf/19910103/srn_nda_routine_jup_edr_199101022159_199101030558_V12.cdf",
    #     )
    #     cdf_title = path.split("/")[7]
    #     save_directory = os.path.join(DATA_DIRECTORY, "out/aurora")
    #     cdf = CdfHandler(path, "rr")
    #     # cdf.resolution(epoch_second_mag, freq_second_mag)

    #     cdf_changed_resolution = cdf.resolution(epoch_second_mag, freq_second_mag)

    ####################################################################

    # CDFだとtargetの座標がxy逆転してしまう問題
    # save_as_cdf(
    #     cdf_changed_resolution,
    #     cdf.epoch_new,
    #     cdf.freq_new,
    #     cdf_title,
    #     os.path.join(save_directory, cdf_title),
    # )

    # データをランダムに切り取って保存
    # for i in range(outfile_num):
    #     n = random.randint(-100, 100)
    #     if n <= -50:
    #         target = (1850, 1400)
    #     elif -50 < n <= 0:
    #         target = (1950, 1600)
    #     elif 0 < n <= 50:
    #         target = (1450, 1500)
    #     elif 50 < n <= 100:
    #         target = (1450, 1700)

    #     random_save_path = os.path.join(
    #         DATA_DIRECTORY,
    #         f"out/random/cdf/{i}.cdf",
    #     )
    #     data, epoch, freq = cdf.cut_cdf(target)

    #     m = random.randint(-100, 100)

    #     if m <= -50:
    #         save_as_cdf(data[::-1, :], epoch, freq, random_save_path)
    #     elif -50 < m <= 0:
    #         save_as_cdf(data[:, ::-1], epoch, freq, random_save_path)
    #     elif 0 < m <= 50:
    #         save_as_cdf(data, epoch, freq, random_save_path)
    #     elif 50 < m <= 100:
    #         save_as_cdf(data[::-1, ::-1], epoch, freq, random_save_path)

    # ランダムに切り取った画像を描画
    # random_saved_directory = os.path.join(DATA_DIRECTORY, "out/random/cdf")
    # for i in range(outfile_num):
    #     random_saved_path = os.path.join(random_saved_directory, f"{i}.cdf")
    #     cdf = pycdf.CDF(random_saved_path)
    #     show_img(cdf["data"], cdf["Epoch"], cdf["Frequency"])

    # show_fullimg(
    #     cdf_title,
    #     target,
    #     size,
    #     cdf_changed_resolution,
    #     cdf.epoch_new,
    #     cdf.freq_new,
    #     target_highlight=True,
    # )

    # else:
    #     print(CDF_DIRECTORY)

    #####################################################################

    # FITS
    # if DATA_DIRECTORY:
    #     # resolution(基本的に実行)
    #     path = os.path.join(
    #         DATA_DIRECTORY, "fits/JUPITER_TRACKING_20201216_100033_2.spectra.fits"
    #     )
    #     fits = FitsHandler(path)
    #     fits_title = path.split("/")[-1]
    #     fits_changed_resolution = fits.resolution(epoch_second_mag, freq_second_mag)
    # fits.resolution(epoch_second_mag, freq_second_mag)

    # show_fullimg(
    #     fits_title,
    #     target,
    #     size,
    #     fits_changed_resolution,
    #     fits.epoch_new,
    #     fits.freq_new,
    # )

    # cv2.imwrite(
    #     f"{DATA_DIRECTORY}/out/real_srn_nda_routine_jup_edr_199101022159_199101030558_V12.jpg",
    #     fits_changed_resolution[::-1, :],
    # )

    # trainとtestを分けて書く
    # random作成
    # target = (600, 200)
    # for i in range(outfile_num):
    #     random_save_path = os.path.join(
    #         DATA_DIRECTORY,
    #         f"out/random/fits/{i}.fits",
    #     )
    #     data, epoch, freq = fits.cut_fits(target)

    #     # print(data.shape)

    #     m = random.randint(-100, 100)

    #     if m <= -50:
    #         save_as_fits(data[::-1, :], epoch, freq, random_save_path)
    #     elif -50 < m <= 0:
    #         save_as_fits(data[:, ::-1], epoch, freq, random_save_path)
    #     elif 0 < m <= 50:
    #         save_as_fits(data, epoch, freq, random_save_path)
    #     elif 50 < m <= 100:
    #         save_as_fits(data[::-1, ::-1], epoch, freq, random_save_path)

    #     save_as_fits(data, epoch, freq, random_save_path)

    # 表示
    # random_saved_directory = os.path.join(DATA_DIRECTORY, "out/random/fits")
    # for i in range(outfile_num):
    #     random_saved_path = os.path.join(random_saved_directory, f"{i}.fits")
    #     hdulist = fits_.open(random_saved_path)
    #     # print(hdulist[0].data.shape)
    #     show_img(hdulist[0].data)

    # integration
    # for i in range(200):
    #     integrate_cdf_and_fits(i, last_file_num)

    ########################################################

    # preprocess 1枚のfitsに対して行う

    # show_fullimg(title, target, cut_size, data, epoch, freq, target_highlight=False)
    # show_fullimg(path, target, size, fits_optimized, optimized_epoch, optimized_freq)


if __name__ == "__main__":
    main()
