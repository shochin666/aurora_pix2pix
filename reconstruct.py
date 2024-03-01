import os
import argparse
import glob
from dotenv import load_dotenv

from src import (
    CdfHandler,
    FitsHandler,
    Preprocess,
)


# filterをかけて画像のコントラストを調整するファイル.そのまま/data/out/に保存される.
# 内部的にはfilter_heightに満たないdBをゼロにすることで低dBを排除している.
# filter_heightの値を変えることによってコントラストを調整できる(0~255).
# Preprocessを見るとわかるが, 1枚の画像を256×256ピクセルに切り取って1枚ずつpredictして最後にそれらを繋ぎ合わせて1枚の画像を出力する

load_dotenv()
META_DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--date", type=str)
    parser.add_argument("--extension", type=str)  # fitsかcdfを引数にとる
    parser.add_argument(
        "--filter_height", type=int, default=0
    )  # 背景ノイズを除去するためのフィルタの高さ(0~255で指定)
    args = parser.parse_args()

    epoch_second_mag, freq_second_mag = (2, 2)
    filter_height = args.filter_height
    extension = args.extension
    data_directory_path = os.path.join(
        META_DATA_DIRECTORY,
        f"{args.extension}",
        args.date,
    )

    # FITS
    if extension == "fits":
        fits_directory_path = os.path.join(
            META_DATA_DIRECTORY,
            "fits",
            args.date,
        )
        files = glob.glob(os.path.join(fits_directory_path, "*"))
        fits_title = files[0].split("/")[-1]
        fits = FitsHandler(os.path.join(fits_directory_path, fits_title))

        # 一時的に/out/separate/以下にフォルダを作成して
        # preprocess -> predict -> reconstruct -> save を行う
        preprocessed = Preprocess(fits, epoch_second_mag, freq_second_mag, extension)
        preprocessed.optimize_data_size()
        preprocessed.separate_data()
        preprocessed.predict_and_concatenate(filter_height)
        preprocessed.save()

    # CDF
    if extension == "cdf":
        cdf_directory_path = os.path.join(
            META_DATA_DIRECTORY,
            "cdf",
            args.date,
        )
        files = glob.glob(os.path.join(cdf_directory_path, "*"))
        cdf_title = files[0].split("/")[-1]
        cdf = CdfHandler(os.path.join(cdf_directory_path, cdf_title))

        preprocessed = Preprocess(cdf, epoch_second_mag, freq_second_mag, extension)
        preprocessed.optimize_data_size()
        preprocessed.separate_data()
        preprocessed.predict_and_concatenate(filter_height)
        preprocessed.save()
