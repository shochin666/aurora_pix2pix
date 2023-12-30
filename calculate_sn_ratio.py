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
    )

    files = glob.glob(os.path.join(data_directory_path, "*"))
    fits_title = files[0].split("/")[-1]
    fits = FitsHandler(os.path.join(data_directory_path, fits_title))
    fits_changed_resolution = fits.resolution(epoch_second_mag, freq_second_mag)

    # highlight_target = (2900, 850)  # 20220112
    # x_beginning, y_beginning = 2900, 850
    x_beginning, y_beginning = (256 * 4, 256 * 2)
    partial_fits = fits_changed_resolution[
        y_beginning : y_beginning + 512, x_beginning : x_beginning + 512
    ][
        ::-1, :
    ]  # imshowのために必要
    # partial_fits = fits_changed_resolution[850:1362, 2900:3412]
    # partial_fits = fits_changed_resolution[:512, 1024:1536][::-1, :]

    # # ML処理後
    path = os.path.join(
        META_DATA_DIRECTORY,
        "out/result/3rd_model/RECONSTUCTED_JUPITER_TRACKING_20201216_100036_0.jpg",
    )

    ml_jpg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # partial_ml = ml_jpg[::-1, :][:512, 1024:1536][::-1, :]
    # partial_ml = ml_jpg[::-1, :][850:1362, 2900:3412][::-1, :]

    # 3次元プロット
    # partial_ml = ml_jpg[850:1362, 2900:3412][::-1, :]
    partial_ml = ml_jpg[::-1, :][
        y_beginning : y_beginning + 512, x_beginning : x_beginning + 512
    ][::-1, ::]

    # plt.imshow(partial_fits)
    # plt.imshow(partial_fits)
    aurora_index_set = []
    noise_index_set = []

    # ML処理後のSN比の計算
    # オーロラ電波のインデックスを取得

    # 平均で計算しないと個数が影響して意味がない
    # ml_sn_ratioを最大化するfilterをループを回して探す
    # シグナルが固まって存在していることを定量化してそこを求める -> 微分??

    x_list = []
    outputs = []
    y_list = []
    fig, ax = plt.subplots()

    for i in range(256):  # 上がったり下がったり問題
        for j in range(512):
            for k in range(512):
                if i <= partial_ml[j, k]:  # オーロラ箇所を取得
                    aurora_index_set.append([j, k])
                else:
                    noise_index_set.append([j, k])

        aurora_loop = np.array(aurora_index_set).shape[0]
        noise_loop = np.array(noise_index_set).shape[0]
        ml_sum_aurora = 0.00001
        ml_sum_noise = 0.00001
        raw_sum_aurora = 0
        raw_sum_noise = 0

        tmp_aurora_loop = aurora_loop
        tmp_noise_loop = noise_loop

        tmp_aurora_loop = 1 if tmp_aurora_loop == 0 else aurora_loop

        tmp_noise_loop = 1 if tmp_noise_loop == 0 else noise_loop

        for l in range(tmp_aurora_loop):
            if len(aurora_index_set) != 0:
                ml_sum_aurora += partial_ml[
                    aurora_index_set[l][0], aurora_index_set[l][1]
                ]

        for m in range(tmp_noise_loop):
            if len(noise_index_set) != 0:
                ml_sum_noise += partial_ml[noise_index_set[m][0], noise_index_set[m][1]]

        ml_ave_aurora = ml_sum_aurora / tmp_aurora_loop  # いいところで200とか
        ml_ave_noise = ml_sum_noise / tmp_noise_loop  # 大体50くらい

        ml_sn_ratio_average = ml_ave_aurora / ml_ave_noise

        outputs.append(ml_sn_ratio_average)
        x_list.append(i)

        ax.set_xlabel("Filter Depth")
        ax.set_ylabel("<ML> Average SNRatio")

        if 1 < outputs[i] - outputs[i - 1]:  # 本番環境ではいらない
            y_list.append(0.1)
        elif outputs[i] - outputs[i - 1] < -1:
            y_list.append(-0.1)
        else:
            y_list.append(outputs[i] - outputs[i - 1])
        print(i, y_list[-1])

    if 1 < len(outputs):
        ax.plot(x_list[1:], y_list[1:])

    ax.axhline(0, xmin=0, xmax=255, color="green", lw=2, ls="--", alpha=0.6)

    plt.show()

    # plt.pause(0.001)

    # # ML処理前
    # for m in range(np.array(aurora_index_set).shape[0]):
    #     raw_sum_aurora += partial_fits[aurora_index_set[k][0], aurora_index_set[k][1]]

    # for n in range(np.array(noise_index_set).shape[0]):
    #     raw_sum_noise += partial_fits[noise_index_set[i][0], noise_index_set[i][1]]

    # raw_sn_ratio = raw_sum_aurora / raw_sum_noise

    # print(ml_sn_ratio / raw_sn_ratio)

    # x = range(512 * 512)
    # Y_1 = np.array(sorted(partial_ml.flatten()))
    # # # Y_1 = sorted(partial_ml.flatten())
    # Y_2 = np.array(sorted(partial_fits.flatten()))

    # X, Y = np.meshgrid(range(512), range(512))

    # plt.hist(Y_1)

    # print(Y_1.shape, Y_2.shape)

    # Y = Y_2 - Y_1

    # max = np.max(Y)
    # for i, v in enumerate(Y):
    #     if v == max:
    #         print(f"{i}番目のindexでした")
    # 2426番目のindexでした
    # 2871番目のindexでした
    # 3273番目のindexでした
    # 4155番目のindexでした
    # 5719番目のindexでした
    # 5921番目のindexでした
    # 180614番目のindexでした
    # 262087番目のindexでした <- ±の入れ替わり

    # Y_2 = sorted(partial_fits.flatten())
    # print(X.shape, Y.shape, np.array(partial_fits).shape)

    # 3D描画
    # fig = plt.figure(figsize=(8, 8))  # 図の設定
    # ax = fig.add_subplot(projection="3d")  # 3D用の設定
    # # ax.plot_surface(X, Y, partial_fits[::-1, :], cmap="jet")
    # ax.plot_surface(X, Y, partial_ml[::-1, :], cmap="jet")
    # ax.set_xlabel("x")  # x軸ラベル
    # ax.set_ylabel("y")  # y軸ラベル
    # ax.set_zlabel("z")  # z軸ラベル

    # plt.show()

# plt.plot(X, Y_1, "g", label="ML")
# plt.plot(X, Y_2, "b", label="RAW")
# # # plt.plot(X, Y_2, "b", label="RAW")

# plt.legend()

# plt.show()

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
