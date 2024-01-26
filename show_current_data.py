import os
import argparse
import glob
from dotenv import load_dotenv

from src import CdfHandler, FitsHandler, show_fullimg

# 生データを画像として表示したい時に実行するファイル.
# 切り取り箇所を確認するために任意の大きさ(highlight_size)と位置(highlight_target)で
# 画像をハイライトすることができる.
# 画像の切り取りおよびハイライト箇所はhighlight_targetをターゲットの左端の座標に設定してそこから
# highlight_sizeの大きさで切り取る.
# ex)python show_current_data.py --date 20201216 --extension fits --highlight
load_dotenv()
META_DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--date", type=str)
    parser.add_argument("--extension", type=str)  # fitsかcdfを引数にとる
    parser.add_argument(
        "--highlight", action="store_true"
    )  # --highlightをオプションにつければTrueが格納される
    args = parser.parse_args()

    epoch_second_mag, freq_second_mag = (2, 2)

    data_directory_path = os.path.join(
        META_DATA_DIRECTORY,
        f"{args.extension}",
        args.date,
    )

    if args.extension == "cdf":
        # データによって異なるのでその都度変更する.
        highlight_target = (2710, 1250)
        highlight_size = (256, 256)

        files = glob.glob(os.path.join(data_directory_path, "*"))
        cdf_title = files[0].split("/")[-1]
        cdf = CdfHandler(os.path.join(data_directory_path, cdf_title), "rr")
        cdf_changed_resolution = cdf.resolution(epoch_second_mag, freq_second_mag)

        show_fullimg(
            cdf_title,
            highlight_target,
            highlight_size,
            cdf_changed_resolution,
            cdf.epoch_new,
            cdf.freq_new,
            target_highlight=args.highlight,
        )

    elif args.extension == "fits":
        # データによって異なるのでその都度変更する.
        highlight_target = (1180, 800)
        highlight_size = (256, 256)

        files = glob.glob(os.path.join(data_directory_path, "*"))
        fits_title = files[0].split("/")[-1]
        fits = FitsHandler(os.path.join(data_directory_path, fits_title))
        fits_changed_resolution = fits.resolution(epoch_second_mag, freq_second_mag)

        show_fullimg(
            fits_title,
            highlight_target,
            highlight_size,
            fits_changed_resolution,
            fits.epoch_new,
            fits.freq_new,
            target_highlight=args.highlight,
        )
