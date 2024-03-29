from scipy import interpolate
from .normalize import min_max
import random
import numpy as np
import astropy.io.fits as fits
import datetime


class FitsHandler:
    def __init__(self, path):
        date = str(path.split("/")[-2])
        hdulist = fits.open(path)

        # dateによってデータの構造が変わる -> まじでクソ
        if date == "20201216":
            tmp_freq = np.array(hdulist[8].data)
        elif date == "20201221":
            tmp_freq = np.array(hdulist[6].data)
        elif date == "20201106":
            tmp_freq = np.array(hdulist[6].data)
        elif date == "20201107":
            tmp_freq = np.array(hdulist[6].data)
        elif date == "20220112":
            tmp_freq = np.array(hdulist[8].data)

        # 異常箇所のindexを取ってくる
        self.freq_turningpoints = [
            i + 1 for i in range(len(tmp_freq) - 1) if tmp_freq[i + 1] - tmp_freq[i] < 0
        ]

        # もし異常がなかったらそのまま最後のindexを代入する
        if len(self.freq_turningpoints):
            self.freq_turningpoint = self.freq_turningpoints[0]

        else:
            self.freq_turningpoint = len(tmp_freq) - 1

        self.path = path

        if date == "20201216":
            self.raw_data = hdulist[3].data[: self.freq_turningpoint, :]
        elif date == "20201221":
            self.raw_data = hdulist[3].data[0][: self.freq_turningpoint, :]
        elif date == "20201106":
            self.raw_data = hdulist[3].data[0][: self.freq_turningpoint, :]
        elif date == "20201107":
            # self.data = hdulist[4].data[: self.freq_turningpoint, :]
            self.raw_data = hdulist[3].data[0][: self.freq_turningpoint, :]
        elif date == "20220112":
            self.raw_data = hdulist[3].data[: self.freq_turningpoint, :]

        # maxが100くらい
        self.data = 10 * np.log10(self.raw_data)

        self.epoch = []

        if date == "20201216":
            tmp_epoch = np.array(hdulist[8].data)
        elif date == "20201221":
            tmp_epoch = np.array(hdulist[5].data)
        elif date == "20201106":
            tmp_epoch = np.array(hdulist[5].data)
        elif date == "20201107":
            tmp_epoch = np.array(hdulist[5].data)
        elif date == "20220112":
            tmp_epoch = np.array(hdulist[8].data)

        # データによって不要だったりするのでその都度変える必要がある
        for i in range(tmp_epoch.shape[0]):
            self.epoch = np.append(
                self.epoch,
                datetime.datetime.fromtimestamp(hdulist[2].data["timestamp"][0])
                + datetime.timedelta(seconds=float(tmp_epoch[i])),
            )

        if date == "20201216":
            self.freq = np.array(hdulist[9].data)[: self.freq_turningpoint]
        elif date == "20201221":
            self.freq = np.array(hdulist[6].data)[: self.freq_turningpoint]
        elif date == "20201106":
            self.freq = np.array(hdulist[6].data)[: self.freq_turningpoint]
        elif date == "20201107":
            self.freq = np.array(hdulist[6].data)[: self.freq_turningpoint]
        elif date == "20220112":
            self.freq = np.array(hdulist[9].data)[: self.freq_turningpoint]

        self.cut_size = (256, 256)
        self.epoch_new = []
        self.freq_new = []
        self.data_new = []

        # print(TimeDelta(hdulist[7].data, format="sec"))

        # 要調整
        self.title = ""

    def resolution(self, epoch_second_mag, freq_second_mag, train=False):
        self.epoch_first_mag = 16
        self.freq_first_mag = 1
        self.epoch_second_mag = epoch_second_mag
        self.freq_second_mag = freq_second_mag
        self.data = self.data[:: -1 * self.freq_first_mag, :: self.epoch_first_mag]
        self.freq = self.freq[:: self.freq_first_mag]

        self.epoch_new = self.epoch[:: self.epoch_first_mag]

        n_f = len(self.freq)
        n_e = len(self.epoch_new)

        if not len(self.epoch_new) % (self.freq_first_mag * self.freq_second_mag):
            self.epoch_new = self.epoch_new[:: self.epoch_second_mag]
        else:
            self.epoch_new = self.epoch_new[:: self.epoch_second_mag][:-1]

        self.interp_1d = interpolate.interp1d(
            np.arange(n_f), self.freq, fill_value="extrapolate"
        )

        self.freq_new = self.interp_1d(
            np.linspace(0, n_f - 1, self.freq_first_mag * self.freq_second_mag * n_f)
        )  # こいつの最後の値

        tmp_before_epoch_list = np.arange(n_e)
        tmp_after_epoch_list = np.linspace(0, n_e - 1, n_e // self.epoch_second_mag)

        self.interp_2d = interpolate.interp2d(
            tmp_before_epoch_list, self.freq, self.data
        )

        self.tmp_data_for_rsn = np.array(
            self.interp_2d(tmp_after_epoch_list, self.freq_new)[::-1, :]
        )

        self.data_new = min_max(self.tmp_data_for_rsn)

        # print((self.freq_new[-1] - self.freq_new[0]) / len(self.freq_new))

        # print((self.epoch_new[0] - self.epoch_new[-1]) / len(self.epoch_new))
        # print(self.data_new.shape)

        return self.data_new

    def cut_fits(self, target, x_range, y_range):
        target_x, target_y = target
        data = self.data_new.copy()

        k = random.randint(0, x_range)
        l = random.randint(0, y_range)

        x_axis_beginning = target_x + k
        y_axis_beginning = target_y + l
        cut_size_x, cut_size_y = self.cut_size

        renewed_epoch = self.epoch_new[x_axis_beginning : x_axis_beginning + 256]
        renewed_freq = self.freq_new[y_axis_beginning : y_axis_beginning + 256]

        try:
            renewed_data = data[
                y_axis_beginning : y_axis_beginning + cut_size_y,
                x_axis_beginning : x_axis_beginning + cut_size_x,
            ].T
        except:
            print("ターゲットの範囲を再確認してください！")

        return renewed_data, renewed_epoch, renewed_freq
