import os
import argparse
import glob
import cv2
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from src import FitsHandler, reverse_min_max

# calculate_filter.pyで算出したfilterを用いてSN比を強度に直して計算するファイル.
# for文を配列のwhereメソッドを使って劇的にコードを修正できる可能性がある.

# .envファイルの内容を読み込む
load_dotenv()
META_DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")

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
    min, max = np.min(fits.data), np.max(fits.data)

    # ML処理後
    # 切り抜く画像の場所のスタート地点を指定
    x_beginning, y_beginning = (256 * 4, 256 * 2)

    partial_raw = fits_changed_resolution[
        y_beginning : y_beginning + 512, x_beginning : x_beginning + 512
    ][::-1, :]

    tmp_partial_raw = fits.tmp_data_for_rsn[
        y_beginning : y_beginning + 512, x_beginning : x_beginning + 512
    ][::-1, :]

    # ベースとなる画像のパスを指定
    path = os.path.join(
        META_DATA_DIRECTORY,
        "out/result/3rd_model/RECONSTUCTED_JUPITER_TRACKING_20201216_100036_0.jpg",
    )
    ml_jpg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # jpgを配列として読み込むとjpgとして保存する前の配列とは異なるので配列の入れ替えが必要になる.
    partial_ml = ml_jpg[::-1, :][
        y_beginning : y_beginning + 512, x_beginning : x_beginning + 512
    ][::-1, ::]

    # SN比を出すためにdBから強度に戻す.
    tmp_partial_ml = reverse_min_max(min, max, ml_jpg)[::-1, :][
        y_beginning : y_beginning + 512, x_beginning : x_beginning + 512
    ][::-1, ::]

    aurora_index_set = []
    noise_index_set = []
    x_list = []
    y_list = []
    filter = 89  # calculate_filter.pyのグラフから得た値を設定

    ml_sum_aurora = 0
    ml_sum_noise = 0
    ml_filter_sum_aurora = 0
    ml_filter_sum_noise = 0
    raw_sum_aurora = 0
    raw_sum_noise = 0

    # フォーカスを当てた場所(シグナルを含む512×512の画像)に絞って行う.
    # 以下のほとんどの部分はcalculate_filter.pyと同じなので参照してください.
    for i in range(512):
        for j in range(512):
            if filter <= partial_ml[i, j]:
                if partial_ml[i, j] == filter:
                    noise_index_set.append([i, j])
                else:
                    aurora_index_set.append([i, j])

        aurora_loop = np.array(aurora_index_set).shape[0]
        noise_loop = np.array(noise_index_set).shape[0]

        # ML処理前
        if aurora_loop != 0:
            for k in range(aurora_loop):
                if len(aurora_index_set) != 0:
                    raw_sum_aurora += tmp_partial_raw[
                        aurora_index_set[k][0], aurora_index_set[k][1]
                    ]

        if noise_loop != 0:
            for l in range(noise_loop):
                if len(noise_index_set) != 0:
                    raw_sum_noise += tmp_partial_raw[
                        noise_index_set[l][0], noise_index_set[l][1]
                    ]

        # ML処理後
        if aurora_loop != 0:
            for k in range(aurora_loop):
                if len(aurora_index_set) != 0:
                    ml_sum_aurora += tmp_partial_ml[
                        aurora_index_set[k][0], aurora_index_set[k][1]
                    ]

        if noise_loop != 0:
            for l in range(noise_loop):
                if len(noise_index_set) != 0:
                    ml_sum_noise += tmp_partial_ml[
                        noise_index_set[l][0], noise_index_set[l][1]
                    ]

        raw_sn_ratio = raw_sum_aurora / raw_sum_noise
        ml_sn_ratio = ml_sum_aurora / ml_sum_noise

    #     x_list.append(filter)
    #     y_list.append(ml_sn_ratio / raw_sn_ratio)

    # plt.plot(x_list, y_list)
    # plt.show()

    print(f"SN比はRAW: {raw_sn_ratio}, SN: {ml_sn_ratio}でした")
    print(f"{ml_sn_ratio / raw_sn_ratio}倍")
