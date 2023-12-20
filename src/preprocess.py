import numpy as np
import os
import subprocess
import cv2
import shutil
from aurora_pix2pix import Aurora_pix2pix
import torchvision
from torchvision import transforms
from .normalize import min_max


class Preprocess:
    def __init__(self, fits_original, epoch_second_mag, freq_second_mag):
        self.DATA_DIRECTORY = os.getenv(
            "DATA_DIRECTORY", "/Users/ogawa/Desktop/desktop_folders/data"
        )
        self.model_file_path = (
            "/Users/ogawa/Desktop/desktop_folders/data/latest_net_G.pth"
        )
        # input_img_path, output_img_path
        self.model = Aurora_pix2pix()
        self.netG = self.model.load_pix2pix_generator(self.model_file_path)

        self.fits_original = fits_original
        self.fits_changed_resolution = fits_original.resolution(
            epoch_second_mag, freq_second_mag
        )
        self.fits_title = fits_original.path.split("/")[-1].split(".")[
            0
        ]  # JUPITER_TRACKING_20201216_100033_2

    def optimize_fits_size(self):
        height, width = self.fits_changed_resolution.shape
        self.horizontal_loop = width // 256
        self.vertical_loop = height // 256

        self.optimized_height, self.optimized_width = (
            256 * self.vertical_loop,
            256 * self.horizontal_loop,
        )

        # 後で使う
        self.optimized_epoch = self.fits_original.epoch_new[: self.optimized_width]
        self.optimized_freq = self.fits_original.freq_new[: self.optimized_height]

        self.fits_optimized = self.fits_changed_resolution.copy()[
            : self.optimized_height, : self.optimized_width
        ]

    def separate_fits(self):
        fits = np.array(self.fits_optimized.copy())

        self.save_directory_path = os.path.join(
            self.DATA_DIRECTORY, f"out/separate/{self.fits_title}"
        )

        # separateファイル保存先ディレクトリ作成
        os.mkdir(self.save_directory_path)

        for i in range(self.vertical_loop):
            for j in range(self.horizontal_loop):
                save_file_path = os.path.join(self.save_directory_path, f"{i}_{j}.jpg")
                data = np.array(fits[i * 256 : 256 * (i + 1), j * 256 : 256 * (j + 1)])
                cv2.imwrite(save_file_path, data)

    def predict_and_concatenate(self):
        self.reconstructed_jpg = np.zeros((self.optimized_height, self.optimized_width))

        translated_jpg_directory_path = os.path.join(
            self.DATA_DIRECTORY, f"out/separate/tmp_{self.fits_title}"
        )

        # shutil.rmtree(saved_jpg_directory)
        os.mkdir(translated_jpg_directory_path)

        for i in range(self.vertical_loop):
            for j in range(self.horizontal_loop):
                saved_jpg_path = os.path.join(self.save_directory_path, f"{i}_{j}.jpg")
                transform = transforms.ToTensor()
                # print(cv2.imread(saved_jpg_path, cv2.IMREAD_GRAYSCALE).shape)
                data = transform(cv2.imread(saved_jpg_path, cv2.IMREAD_GRAYSCALE))

                translated_data = (
                    self.model.translate(np.squeeze(data), self.netG) * 256 + 256
                )

                translated_jpg_path = os.path.join(
                    translated_jpg_directory_path, f"{i}_{j}.jpg"
                )
                cv2.imwrite(translated_jpg_path, translated_data)
                self.reconstructed_jpg[
                    i * 256 : 256 * (i + 1), j * 256 : 256 * (j + 1)
                ] = translated_data
        self.reconstructed_jpg = min_max(self.reconstructed_jpg.copy())
        self.reconstructed_jpg = np.where(
            self.reconstructed_jpg < 150, 0, self.reconstructed_jpg
        )

        # print(
        #     np.min(self.reconstructed_jpg),
        #     np.max(self.reconstructed_jpg),
        # )

        shutil.rmtree(self.save_directory_path)
        shutil.rmtree(translated_jpg_directory_path)

    def save(self):
        cv2.imwrite(
            os.path.join(
                self.DATA_DIRECTORY,
                f"out/separate{self.fits_title}.jpg",
            ),
            self.reconstructed_jpg[::-1, :],
        )
        # hoge = cv2.imread(
        #     os.path.join(saved_jpg_directory, "reconstructed.jpg"),
        # )
