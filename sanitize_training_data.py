import os
import shutil
from dotenv import load_dotenv


# prepare_for_training.pyによって生成された画像を削除するファイル.
# 新しく訓練データを作成する前に必ずこのファイルを実行する.

load_dotenv()
META_DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")

if __name__ == "__main__":
    sanitized_dirs = [
        os.path.join(META_DATA_DIRECTORY, "out/random/cdf"),
        os.path.join(META_DATA_DIRECTORY, "out/random/fits"),
        os.path.join(META_DATA_DIRECTORY, "out/random/cdf1"),
        os.path.join(META_DATA_DIRECTORY, "out/random/cdf2"),
        os.path.join(META_DATA_DIRECTORY, "out/random/noise_jpg"),
        os.path.join(META_DATA_DIRECTORY, "out/train/A"),
        os.path.join(META_DATA_DIRECTORY, "out/train/B"),
        os.path.join(META_DATA_DIRECTORY, "out/test/A"),
        os.path.join(META_DATA_DIRECTORY, "out/test/B"),
    ]

    for dir_path in sanitized_dirs:
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
