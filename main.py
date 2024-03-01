import os
import argparse
import glob
import cv2
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from src import FitsHandler, reverse_min_max


# .envファイルの内容を読み込む
load_dotenv()
META_DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")

# シグナルとノイズの切り分けの基準となる(filter)を見つけるためのグラフを描画するファイル.
if __name__ == "__main__":
    # スクリプトファイルの引数の初期化
    parser = argparse.ArgumentParser(description="")

    # 必須
    parser.add_argument("--date", type=str)
    args = parser.parse_args()

    epoch_second_mag, freq_second_mag = (2, 2)

    data_directory_path = os.path.join(
        META_DATA_DIRECTORY,
        f"fits",
        args.date,
    )

    # ML処理後のデータを準備
    files = glob.glob(os.path.join(data_directory_path, "*"))
    fits_title = files[0].split("/")[-1]
    fits = FitsHandler(os.path.join(data_directory_path, fits_title))
    fits_changed_resolution = fits.resolution(epoch_second_mag, freq_second_mag)
    min, max = np.min(fits.tmp_data_for_rsn), np.max(fits.tmp_data_for_rsn)

    # 切り抜く画像の場所のスタート地点を指定
    x_beginning, y_beginning = (256 * 4, 256 * 2)

    # ベースとなる画像のパスを指定
    path = os.path.join(
        META_DATA_DIRECTORY,
        "out/result/3rd_model/RECONSTUCTED_JUPITER_TRACKING_20201216_100036_0.jpg",
    )
    ml_jpg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    tmp_partial_raw = fits.tmp_data_for_rsn[
        y_beginning : y_beginning + 512, x_beginning : x_beginning + 512
    ][::-1, :]

    partial_ml = ml_jpg[::-1, :][
        y_beginning : y_beginning + 512, x_beginning : x_beginning + 512
    ][::-1, ::]

    tmp_partial_ml = reverse_min_max(min, max, ml_jpg)[::-1, :][
        y_beginning : y_beginning + 512, x_beginning : x_beginning + 512
    ][::-1, ::]

    x1_list = []
    y1_list = []
    x2_list = []
    y2_list = []
    outputs = []

    # 0~255までfilter_heightを変化させて値を取得
    for filter_height in range(48, 256):
        # 計算のために必要なパラメータの初期化
        ml_sum_aurora = 0
        ml_sum_noise = 0
        raw_sum_aurora = 0
        raw_sum_noise = 0
        aurora_index_set = []
        noise_index_set = []

        # フォーカスを当てた場所(シグナルを含む512×512の画像)に絞って行う.
        for j in range(512):
            for k in range(512):
                # filter_heightと大小を比較してfilter_height以上のdBをシグナルとしてカウントし、未満のdBをノイズとしてカウントしてそのピクセルの配列にappend
                if filter_height <= partial_ml[j, k]:
                    aurora_index_set.append([j, k])
                else:
                    noise_index_set.append([j, k])

        # シグナルとノイズのカウント数を取得
        aurora_loop = np.array(aurora_index_set).shape[0]
        noise_loop = np.array(noise_index_set).shape[0]

        tmp_aurora_loop = aurora_loop
        tmp_noise_loop = noise_loop

        # もしオーロラとノイズのカウントが0だった場合tmp_loopに1を代入して直下のforループのエラーをエスケープしている.
        if aurora_loop == 0:
            ml_sum_aurora = 1
            raw_sum_aurora = 1
            tmp_aurora_loop = 1

        if noise_loop == 0:
            ml_sum_noise = 1
            raw_sum_noise = 1
            tmp_noise_loop = 1

        # ループごとに値を集積していく.

        for k in range(aurora_loop):
            if len(aurora_index_set) != 0:
                raw_sum_aurora += tmp_partial_raw[
                    aurora_index_set[k][0], aurora_index_set[k][1]
                ]

        for l in range(noise_loop):
            if len(noise_index_set) != 0:
                raw_sum_noise += tmp_partial_raw[
                    noise_index_set[l][0], noise_index_set[l][1]
                ]

        for m in range(tmp_aurora_loop):
            if len(aurora_index_set) != 0:
                ml_sum_aurora += tmp_partial_ml[
                    aurora_index_set[m][0], aurora_index_set[m][1]
                ]

        for n in range(tmp_noise_loop):
            if len(noise_index_set) != 0:
                ml_sum_noise += tmp_partial_ml[
                    noise_index_set[n][0], noise_index_set[n][1]
                ]

        rsn = ml_sum_aurora / ml_sum_noise

        x1_list.append(filter_height)
        y1_list.append(ml_sum_aurora)
        x2_list.append(filter_height)
        y2_list.append(ml_sum_noise)

        div_y1 = []
        div_y2 = []
        for i in range(len(y1_list) - 2):
            div_y1.append(y1_list[i + 1] - y1_list[i])
            div_y2.append(y2_list[i + 1] - y2_list[i])

        print(filter_height)

    if len(y1_list) > 1 and len(y2_list) > 1:
        plt.plot(x1_list, y1_list, label="SIGNAL")
        plt.plot(x2_list, y2_list, label="NOISE")
        plt.xlabel("filter_height")
        plt.ylabel("Differential_Rsn")
        plt.legend()
        plt.show()
