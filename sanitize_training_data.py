import os
import shutil


def main():
    META_DATA_DIRECTORY = os.getenv(
        "DATA_DIRECTORY", "/Users/ogawa/Desktop/desktop_folders/data"
    )
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

    for path in sanitized_dirs:
        shutil.rmtree(path)
        os.mkdir(path)


if __name__ == "__main__":
    main()
