import numpy as np
import os
import cv2
import shutil
from dotenv import load_dotenv

from .aurora_pix2pix import Aurora_pix2pix
from torchvision import transforms
from .normalize import min_max


load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")
MODEL_DIRECTORY = os.getenv("MODEL_DIRECTORY")


class Preprocess:
    def __init__(self, data_original, epoch_second_mag, freq_second_mag, extension):
        self.model_file_path = os.path.join(MODEL_DIRECTORY, "latest_net_G.pth")
        self.model = Aurora_pix2pix()
        self.netG = self.model.load_pix2pix_generator(self.model_file_path)

        self.data_original = data_original
        self.data_changed_resolution = data_original.resolution(
            epoch_second_mag, freq_second_mag
        )
        self.data_title = data_original.path.split("/")[-1].split(".")[0]

    def optimize_data_size(self):
        height, width = self.data_changed_resolution.shape
        self.horizontal_loop = width // 256
        self.vertical_loop = height // 256

        self.optimized_height, self.optimized_width = (
            256 * self.vertical_loop,
            256 * self.horizontal_loop,
        )

        # 後で使う
        self.optimized_epoch = self.data_original.epoch_new[: self.optimized_width]
        self.optimized_freq = self.data_original.freq_new[: self.optimized_height]

        self.data_optimized = self.data_changed_resolution.copy()[
            : self.optimized_height, : self.optimized_width
        ]

    def separate_data(self):
        whole_data = np.array(self.data_optimized.copy())

        self.save_directory_path = os.path.join(
            DATA_DIRECTORY, f"out/separate/{self.data_title}"
        )

        # separateファイル保存先ディレクトリ作成
        os.mkdir(self.save_directory_path)

        for i in range(self.vertical_loop):
            for j in range(self.horizontal_loop):
                save_file_path = os.path.join(self.save_directory_path, f"{i}_{j}.jpg")
                data = np.array(
                    whole_data[i * 256 : 256 * (i + 1), j * 256 : 256 * (j + 1)]
                )
                cv2.imwrite(save_file_path, data)

    def predict_and_concatenate(self, filter_height):
        self.reconstructed_jpg = np.zeros((self.optimized_height, self.optimized_width))

        translated_jpg_directory_path = os.path.join(
            DATA_DIRECTORY, f"out/separate/tmp_{self.data_title}"
        )
        os.mkdir(translated_jpg_directory_path)

        for i in range(self.vertical_loop):
            for j in range(self.horizontal_loop):
                saved_jpg_path = os.path.join(self.save_directory_path, f"{i}_{j}.jpg")
                transform = transforms.ToTensor()
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
            self.reconstructed_jpg < filter_height, 0, self.reconstructed_jpg
        )

        shutil.rmtree(self.save_directory_path)
        shutil.rmtree(translated_jpg_directory_path)

    def save(self):
        cv2.imwrite(
            os.path.join(
                DATA_DIRECTORY,
                f"out/RECONSTUCTED_{self.data_title}.jpg",
            ),
            self.reconstructed_jpg[::-1, :],
        )
