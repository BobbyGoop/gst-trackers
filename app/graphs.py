import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same').astype(np.uint16)
    return y_smooth


def show_stats(arr):
    for s in arr:
        return np.mean(s), np.max(s), np.min(s)


if __name__ == "__main__":
    # font = {'family': 'normal', 'weight': 'normal', 'size': 12}
    #

    fs = 12
    folder = ""
    cap = "frame_rate"
    test_1 = np.array(pd.read_csv(folder + 'fps_siam_480.csv')[cap], dtype=np.uint32)
    test_2 = np.array(pd.read_csv(folder + 'fps_siam_hd.csv')[cap], dtype=np.uint32)
    test_3 = np.array(pd.read_csv(folder + 'fps_siam_fhd.csv')[cap], dtype=np.uint32)
    # test_2 = np.array(pd.read_csv(folder + 'mosse_multiple_roi200.csv')[cap], dtype=np.uint32)
    # test_3 = np.array(pd.read_csv(folder + 'mosse_multiple_roi280.csv')[cap], dtype=np.uint32)
    # if cap == "PSR":
    #     test_1 = np.array([i if i < 600 else 600 for i in test_1])
    #     test_2 = np.array([i if i < 600 else 600 for i in test_2])
    #     test_3 = np.array([i if i < 600 else 600 for i in test_3])
    print("Stats 480p: ", show_stats([test_1]))
    print("Stats 720p: ", show_stats([test_2]))
    print("Stats 1080p: ", show_stats([test_3]))

    ticks = list(range(0, 1800, 1))
    window_length = 10
    poly_order = 3

    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = str(fs)
    fig.suptitle("Производительность SiamMask при наличии в кадре схожих объектов")
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(fs)

    print([show_stats([test_1])[0]] * 1800)
    ax.plot(ticks, signal.savgol_filter(test_1, window_length=window_length, polyorder=poly_order), label="480p", c='g')
    ax.plot(ticks, [show_stats([test_1])[0]] * 1800, label="480p MEAN", c='g', linestyle='--')
    ax.plot(ticks, signal.savgol_filter(test_2, window_length=window_length, polyorder=poly_order), label="HD", c='orange')
    ax.plot(ticks, [show_stats([test_2])[0]] * 1800, label="HD MEAN", c='orange', linestyle='--')
    ax.plot(ticks, signal.savgol_filter(test_3, window_length=window_length, polyorder=poly_order), label="FHD", c='purple')
    ax.plot(ticks, [show_stats([test_3])[0]] * 1800, label="FHD MEAN", c='purple', linestyle='--')
    ax.set_xticks(np.arange(0, 1900, 100))
    ax.set_yticks(np.arange(0, 60, 5))
    ax.set_xlabel("Номер кадра", fontsize=fs)
    ax.set_ylabel("FPS", fontsize=fs)

    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.98, top=0.9)
    plt.legend(loc="best")
    plt.grid()
    plt.show()