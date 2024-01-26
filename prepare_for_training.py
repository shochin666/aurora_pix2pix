import os
import random
import argparse
import glob
import cv2
import shutil
import numpy as np
from dotenv import load_dotenv


from src import (
    CdfHandler,
    FitsHandler,
    save_as_cdf,
    save_as_fits,
    integration_for_training,
    integration_for_testing,
)

# pix2pixの学習のためのデータを準備するファイル.

load_dotenv()
META_DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # [Nançayのシグナル+NanuFARのノイズ]もしくは[Nançayのシグナル+Nançayのノイズ]で訓練データセットを作成できる.
    # [Nançayのシグナル+NanuFARのノイズ]の場合、以下の二つを引数を与えて実行
    parser.add_argument("--cdf_date", type=str, default="")
    parser.add_argument("--fits_date", type=str, default="")

    # [Nançayのシグナル+Nançayのノイズ]の場合、以下の二つを引数を与えて実行
    parser.add_argument("--cdf1_date", type=str, default="")
    parser.add_argument("--cdf2_date", type=str, default="")

    # 必須
    parser.add_argument("--random_file_num", type=int)  # データセットの生成に用いるランダムな画像の数を設定
    parser.add_argument("--integration_file_num", type=int)  # シグナルとノイズを重ねた画像の枚数を設定
    args = parser.parse_args()

    cdf_combo = False

    # 一時的な出力先ディレクトリのパスを指定
    out_dirs = [
        os.path.join(META_DATA_DIRECTORY, "out/random/cdf"),
        os.path.join(META_DATA_DIRECTORY, "out/random/fits"),
        os.path.join(META_DATA_DIRECTORY, "out/random/cdf1"),
        os.path.join(META_DATA_DIRECTORY, "out/random/cdf2"),
    ]

    # 以下のコード内について
    # out/train/Aなどは最終的に出力するディレクトリ.
    # out/random/cdfなどは訓練データ作成のために一時的に使用するディレクトリ.
    # 以上の構成はpix2pixのREAD_MEを読んでモデルを訓練する時に用いる訓練用のディレクトリ構成と同じ構成にしている.
    random_file_num = args.random_file_num
    integration_file_num = args.integration_file_num
    epoch_second_mag, freq_second_mag = (2, 2)
    existing_train_files = glob.glob(
        os.path.join(
            META_DATA_DIRECTORY,
            "out/train/A/*",
        )
    )
    existing_test_files = glob.glob(
        os.path.join(
            META_DATA_DIRECTORY,
            "out/test/A/*",
        )
    )

    train_file_num = len(existing_train_files)
    test_file_num = len(existing_test_files)

    # Nançayのシグナル+NanuFARのノイズ
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

        # cut_cdfに必要なパラメータを設定
        # データによって異なるのでその都度変更する.
        cdf_target = (465, 1450)
        x_range, y_range = (165, 0)

        for i in range(random_file_num):
            random_cdf_save_path = os.path.join(
                META_DATA_DIRECTORY,
                f"out/random/cdf/{i}.cdf",
            )
            data, epoch, freq = cdf.cut_cdf(cdf_target, x_range, y_range)
            m = random.randint(-100, 100)

            # 乱数mの値によって配列の軸を入れ替えてランダムにaugumentationする
            if m <= -50:
                save_as_cdf(data[::-1, :], epoch, freq, random_cdf_save_path)
            elif -50 < m <= 0:
                save_as_cdf(data[:, ::-1], epoch, freq, random_cdf_save_path)
            elif 0 < m <= 50:
                save_as_cdf(data, epoch, freq, random_cdf_save_path)
            elif 50 < m <= 100:
                save_as_cdf(data[::-1, ::-1], epoch, freq, random_cdf_save_path)

        # FITS
        fits_directory_path = os.path.join(
            META_DATA_DIRECTORY,
            "fits",
            args.fits_date,
        )

        # cut_fitsに必要なパラメータを設定
        # データによって異なるのでその都度変更する.
        fits_target = (1400, 0)
        x_range, y_range = (1100, 30)  # 変数を上書きして使い回している.
        files = glob.glob(os.path.join(fits_directory_path, "*"))
        fits_title = files[0].split("/")[-1]
        fits = FitsHandler(os.path.join(fits_directory_path, fits_title))
        fits.resolution(epoch_second_mag, freq_second_mag)

        for i in range(random_file_num):
            random_save_path = os.path.join(
                META_DATA_DIRECTORY,
                f"out/random/fits/{i}.fits",
            )

            data, epoch, freq = fits.cut_fits(fits_target, x_range, y_range)
            m = random.randint(-100, 100)

            # 上記と同じaugumentationを行うが、読み込み時にデータの型が変わってしまうので.Tで転置している.
            if m <= -50:
                save_as_fits(data[::-1, :], epoch, freq, random_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise_jpg/{i + train_file_num}.jpg",
                    ),
                    data.T[::-1, :],
                )

            elif -50 < m <= 0:
                save_as_fits(data[:, ::-1], epoch, freq, random_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise_jpg/{i + train_file_num}.jpg",
                    ),
                    data.T[:, ::-1],
                )

            elif 0 < m <= 50:
                save_as_fits(data, epoch, freq, random_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise_jpg/{i + train_file_num}.jpg",
                    ),
                    data.T,
                )

            elif 50 < m <= 100:
                save_as_fits(data[::-1, ::-1], epoch, freq, random_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise_jpg/{i + train_file_num}.jpg",
                    ),
                    data.T[::-1, ::-1],
                )

    # Nançayのシグナル+Nançayのノイズ(cdf1:シグナル, cdf2:ノイズ)
    if len(args.cdf1_date) and len(args.cdf2_date):
        cdf_combo = True

        # データによって異なるのでその都度変更する.
        cdf1_target = (1980, 680)
        cdf2_target = (2710, 1250)

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

            # データによって異なるのでその都度変更する.
            x_range, y_range = (0, 220)

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

            # データによって異なるのでその都度変更する.
            x_range, y_range = (180, 60)

            data, epoch, freq = cdf2.cut_cdf(cdf2_target, x_range, y_range)

            m = random.randint(-100, 100)

            if m <= -50:
                save_as_cdf(data[::-1, :], epoch, freq, random_cdf2_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise_jpg/{i + train_file_num}.jpg",
                    ),
                    data[::-1, :],
                )

            elif -50 < m <= 0:
                save_as_cdf(data[:, ::-1], epoch, freq, random_cdf2_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise_jpg/{i + train_file_num}.jpg",
                    ),
                    data[:, ::-1],
                )

            elif 0 < m <= 50:
                save_as_cdf(data, epoch, freq, random_cdf2_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise_jpg/{i + train_file_num}.jpg",
                    ),
                    data,
                )

            elif 50 < m <= 100:
                save_as_cdf(data[::-1, ::-1], epoch, freq, random_cdf2_save_path)
                cv2.imwrite(
                    os.path.join(
                        META_DATA_DIRECTORY,
                        f"out/random/noise_jpg/{i + train_file_num}.jpg",
                    ),
                    data[::-1, ::-1],
                )

    # integration
    for i in range(
        integration_file_num * 3 // 4
    ):  # 全体の75%を訓練用に画像の重ね合わせ -> よりランダムに画像をsplitするためにscikitlearnのモジュール使ってもいいかもしれないが容量を考慮してやめた.
        integration_for_training(i + train_file_num, integration_file_num, cdf_combo)

    for i in range(integration_file_num // 4):  # 全体の25%をテスト用に画像の重ね合わせ
        integration_for_testing(i + test_file_num, integration_file_num, cdf_combo)

    # 一時的に用いたディレクトリをsanitizeして不要なフォルダを削除
    for path in out_dirs:
        shutil.rmtree(path)
        os.mkdir(path)
        print(f"{path}を削除しました")
