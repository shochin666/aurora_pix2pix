import os
import argparse
import glob
import cv2
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt


from src import FitsHandler

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

    # 切り抜く画像の場所のスタート地点を指定
    x_beginning, y_beginning = (256 * 4, 256 * 2)

    # ベースとなる画像のパスを指定
    path = os.path.join(
        META_DATA_DIRECTORY,
        "out/result/3rd_model/RECONSTUCTED_JUPITER_TRACKING_20201216_100036_0.jpg",
    )
    ml_jpg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 3次元プロットのために配列の入れ替えが必要
    partial_ml = ml_jpg[::-1, :][
        y_beginning : y_beginning + 512, x_beginning : x_beginning + 512
    ][::-1, ::]

    x_list = []
    y_list = []
    outputs = []

    # 0~255までfilter_depthを変化させて値を取得
    for filter_depth in range(256):
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
                # filter_depthと大小を比較してfilter_depth以上のdBをシグナルとしてカウントし、未満のdBをノイズとしてカウントしてそのピクセルの配列にappend
                if filter_depth <= partial_ml[j, k]:
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
            tmp_aurora_loop = 1

        if noise_loop == 0:
            ml_sum_noise = 1
            tmp_noise_loop = 1

        # ループごとに値を集積していく.
        for l in range(tmp_aurora_loop):
            if len(aurora_index_set) != 0:
                ml_sum_aurora += partial_ml[
                    aurora_index_set[l][0], aurora_index_set[l][1]
                ]

        for m in range(tmp_noise_loop):
            if len(noise_index_set) != 0:
                ml_sum_noise += partial_ml[noise_index_set[m][0], noise_index_set[m][1]]

        # シグナルの平均値を取得
        aurora_ave = ml_sum_aurora / tmp_aurora_loop

        x_list.append(filter_depth)
        y_list.append(aurora_ave)

    if len(y_list) > 1:
        plt.plot(x_list[1:], y_list[1:])
        plt.title("filter_depthを変えていった時のシグナルの平均値")
        plt.xlabel("filter_depth")
        plt.ylabel("aurora_average")
        plt.show()
