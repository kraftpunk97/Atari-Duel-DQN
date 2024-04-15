import numpy as np
from collections import abc


def preprocessing(framebuffer: abc.Iterable):
    """
    Accepts a list of game frames and preprocesses them, as described by the paper
    First it down-samples and crops the 210 x 160 game screen to 80 x 80
    then it converts the cropped+downsampled screen to grayscale.

    :param framebuffer:
    :return:
    """
    return [(np.dot(frame[34:194:2, ::2, :],  # Cropping the frame and down-sampling
                    [0.0008, 0.0028, 0.0002]))  # Applying grayscale
            for frame in framebuffer]
