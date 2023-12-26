import os
import argparse
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src import FitsHandler, min_max


if __name__ == "__main__":
    # parser init
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--date", type=str)  # example)19910103

    args = parser.parse_args()

    epoch_second_mag, freq_second_mag = (2, 2)
    META_DATA_DIRECTORY = os.getenv(
        "DATA_DIRECTORY", "/Users/ogawa/Desktop/desktop_folders/data/"
    )

    data_directory_path = os.path.join(
        META_DATA_DIRECTORY,
        f"fits",
        args.date,
    )  #  example)/Users/ogawa/Desktop/desktop_folders/data/cdf/19910103

    files = glob.glob(os.path.join(data_directory_path, "*"))
    fits_title = files[0].split("/")[-1]
    fits = FitsHandler(os.path.join(data_directory_path, fits_title))
    fits_changed_resolution = fits.resolution(epoch_second_mag, freq_second_mag)

    partial_fits = fits_changed_resolution[:512, 1024:1536][::-1, :]

    # # ML処理後
    path = os.path.join(
        META_DATA_DIRECTORY,
        "out/result/3rd_model/FILTER_0_RECONSTUCTED_JUPITER_TRACKING_20201216_100033_2.jpg",
    )

    ml_jpg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    partial_ml = ml_jpg[::-1, :][:512, 1024:1536][::-1, :]

    aurora_index_set = []
    noise_index_set = []
    ml_sum_aurora = 0
    ml_sum_noise = 0
    raw_sum_aurora = 0
    raw_sum_noise = 0

    # ML処理後のSN比の計算
    # オーロラ電波のインデックスを取得
    for i in range(512):
        for j in range(512):
            if 160 <= partial_ml[i, j] < 200:  # オーロラ箇所を取得
                aurora_index_set.append([i, j])
            else:
                noise_index_set.append([i, j])

    for k in range(np.array(aurora_index_set).shape[0]):
        ml_sum_aurora += partial_ml[aurora_index_set[k][0], aurora_index_set[k][1]]

    for l in range(np.array(noise_index_set).shape[0]):
        ml_sum_noise += partial_ml[noise_index_set[i][0], noise_index_set[i][1]]

    ml_sn_ratio = ml_sum_aurora / ml_sum_noise

    # ML処理前
    for m in range(np.array(aurora_index_set).shape[0]):
        raw_sum_aurora += partial_fits[aurora_index_set[k][0], aurora_index_set[k][1]]

    for n in range(np.array(noise_index_set).shape[0]):
        raw_sum_noise += partial_fits[noise_index_set[i][0], noise_index_set[i][1]]

    raw_sn_ratio = raw_sum_aurora / raw_sum_noise

    X = range(512 * 512)
    Y_1 = sorted(partial_ml.flatten())
    Y_2 = sorted(partial_fits.flatten())

    plt.plot(X, Y_1, "g", label="ML")
    plt.plot(X, Y_2, "b", label="RAW")

    plt.legend()

    plt.show()

    # print(ml_sum_aurora, ml_sum_noise)
    # print(raw_sum_aurora, raw_sum_noise)

    # ML処理前
    # S --- 1291個のデータの合計: 175514.42929844485
    # N --- 260853個のデータの合計: 17206203.030235615

    # ML処理後
    # aurora: 1.280243557691898倍
    # noise: 0.3749819887960181倍
    # S --- 1291個のデータの合計: 224701.21739130415
    # N --- 260853個のデータの合計: 6452016.231905824

# 度数の準備
# for i in range(256):
#     count = len(partial_ml[partial_ml == i])
#     result.append(count * i)
