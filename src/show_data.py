from matplotlib import dates as mdates
import numpy as np
import matplotlib.pyplot as plt
import cv2


def show_fullimg(title, target, cut_size, data, epoch, freq, target_highlight=False):
    target_x, target_y = target
    cut_size_x, cut_size_y = cut_size
    _cdf = data.copy()

    X, Y = np.meshgrid(
        epoch,
        freq,
    )

    if target_highlight:
        try:
            _cdf[
                target_y : target_y + cut_size_x,
                target_x : target_x + cut_size_y,
            ] = 170
        except:
            print("ターゲットの範囲を再確認してください！")

    plt.figure(figsize=(12, 5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.pcolormesh(X, Y, np.array(_cdf), cmap="jet")
    plt.title(f"{title}{data.shape}")
    plt.colorbar()
    plt.show()


def show_img(data, epoch=None, freq=None, cdf=False):
    if cdf:
        X, Y = np.meshgrid(epoch, freq)
    else:
        X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

    plt.figure(figsize=(1, 1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.pcolormesh(X, Y, np.array(data), cmap="jet")
    plt.show()
