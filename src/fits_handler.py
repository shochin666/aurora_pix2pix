from scipy import interpolate
from .normalize import min_max
import random
import numpy as np
import astropy.io.fits as fits
import datetime


class FitsHandler:
    def __init__(self, path):
        hdulist = fits.open(path)
        tmp_freq = np.array(hdulist[8].data)

        # 異常箇所のindexを取ってくる
        self.freq_turningpoint = [
            i + 1 for i in range(len(tmp_freq) - 1) if tmp_freq[i + 1] - tmp_freq[i] < 0
        ][0]

        # もし異常がなかったらそのまま最後のindexを代入する
        if not self.freq_turningpoint:
            self.freq_turningpoint = len(tmp_freq) - 1

        self.path = path
        self.data = 10 * np.log10(hdulist[3].data)[: self.freq_turningpoint, :]
        self.epoch = []

        # データによって不要だったりするのでその都度変える必要がある
        for i in range(hdulist[7].data.shape[0]):
            self.epoch = np.append(
                self.epoch,
                datetime.datetime.fromtimestamp(hdulist[2].data["timestamp"][0])
                + datetime.timedelta(seconds=hdulist[7].data[i]),
            )

        self.freq = np.array(hdulist[8].data)[: self.freq_turningpoint]
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
        self.data = self.data[::-1, :: self.epoch_first_mag]
        self.freq = self.freq[:: self.freq_first_mag]
        self.epoch_new = self.epoch[:: self.epoch_first_mag]  # 30260

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

        self.data_new = min_max(
            np.array(self.interp_2d(tmp_after_epoch_list, self.freq_new)[::-1, :])
        )

        # print((self.freq_new[-1] - self.freq_new[0]) / len(self.freq_new))

        # print((self.epoch_new[0] - self.epoch_new[-1]) / len(self.epoch_new))
        # print(self.data_new.shape)

        return self.data_new

    def cut_fits(self, target):
        target_x, target_y = target
        data = self.data_new.copy()

        k = random.randint(-300, 300)
        l = random.randint(-50, 50)

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
