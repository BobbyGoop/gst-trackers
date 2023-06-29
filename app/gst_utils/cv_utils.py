import numpy as np
import cv2


def grayscale(img):   
    """
        Convert colored frame to Grayscale

        :param img: [height, width, channels >= 3]
        :type img: np.ndarray

        :rtype: np.ndarray
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size=3, sigma=(1, 1)):  
    """
        Blur frame

        :param img: [height, width, channels >= 3]
        :type img: np.ndarray

        :param kernel_size:
        :type kernel_size: int

        :param sigma: [height, width, channels >= 3]
        :type sigma: tuple -> (int, int)

        :rtype: np.ndarray
    """  
    sigmaX, sigmaY = sigma
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigmaX, sigmaY=sigmaY)


def linear_mapping(img):
    """Used for linear mapping..."""
    return (img - img.min()) / (img.max() - img.min())


def pre_process(img):
    """Pre-processing the frame..."""
    # get the size of the img...
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # use the hanning window...
    window = window_func_2d(height, width)
    img = img * window

    return img


def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)

    win = mask_col * mask_row

    return win


def random_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    # rotate the frame...
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot
