from spacepy import pycdf
from scipy import interpolate
from matplotlib import dates as mdates
from .normalize import min_max
import os
import random
import numpy as np


class CdfHandler:
    def __init__(self, path, mode="rr"):
        cdfs = pycdf.CDF(path)
        self.path = path
        self.mode = mode
        self.title = path.split("/")[-2]
        self.epoch = cdfs["Epoch"]
        self.freq = cdfs["Frequency"]
        self.data = cdfs.copy()[f"{mode}".upper()]
        self.cut_size = (256, 256)

    def resolution(self, epoch_second_mag, freq_second_mag):
        self.epoch_first_mag = 4
        self.freq_first_mag = 3
        self.epoch_second_mag = epoch_second_mag
        self.freq_second_mag = freq_second_mag

        n_f = self.freq.shape[0]
        n_e = self.epoch.shape[0]

        interp_1d = interpolate.interp1d(
            np.arange(n_f), self.freq, fill_value="extrapolate"
        )

        self.epoch_new = self.epoch[:: self.epoch_first_mag * self.epoch_second_mag][
            :-1
        ]
        self.freq_new = interp_1d(
            np.linspace(0, n_f, self.freq_first_mag * self.freq_second_mag * n_f)
        )

        tmp_before_epoch_list = np.arange(len(self.epoch))
        tmp_after_epoch_list = np.linspace(
            1, n_e, n_e // (self.epoch_first_mag * self.epoch_second_mag)
        )

        interp_2d = interpolate.interp2d(tmp_before_epoch_list, self.freq, self.data.T)

        self.data_new = min_max(interp_2d(tmp_after_epoch_list, self.freq_new))

        return self.data_new

    def cut_cdf(self, target):
        target_x, target_y = target
        data = self.data_new.copy()

        # init(乱数を2つ発生させてターゲット画像の四隅の座標のベンチマークを作成)
        k = random.randint(-1000, 1000)
        l = random.randint(-200, 200)

        x_axis_beginning = target_x + k
        y_axis_beginning = target_y + l
        cut_size_x, cut_size_y = self.cut_size

        renewed_epoch = self.epoch_new[x_axis_beginning : x_axis_beginning + 256]
        renewed_freq = self.freq_new[y_axis_beginning : y_axis_beginning + 256]

        try:
            renewed_data = np.array(
                data[
                    y_axis_beginning : y_axis_beginning + cut_size_x,
                    x_axis_beginning : x_axis_beginning + cut_size_y,
                ]
            )
        except:
            print("ターゲットの範囲を再確認してください！")

        return renewed_data, renewed_epoch, renewed_freq
