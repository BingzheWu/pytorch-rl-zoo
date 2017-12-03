import logging
import numpy as numpy
#import cv2

from collections import namedtuple

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')
def loggerConfig(log_file, verbose=2):
   logger      = logging.getLogger()
   formatter   = logging.Formatter('[%(levelname)-8s] (%(processName)-11s) %(message)s')
   fileHandler = logging.FileHandler(log_file, 'w')
   fileHandler.setFormatter(formatter)
   logger.addHandler(fileHandler)
   if verbose >= 2:
       logger.setLevel(logging.DEBUG)
   elif verbose >= 1:
       logger.setLevel(logging.INFO)
   else:
       # NOTE: we currently use this level to log to get rid of visdom's info printouts
       logger.setLevel(logging.WARNING)
   return logger
def preprocessAtari(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame*= (1. / 255.)
    return frame

# TODO: check the order rgb to confirm
def rgb2gray(rgb):
    gray_image     = 0.2126 * rgb[..., 0]
    gray_image[:] += 0.0722 * rgb[..., 1]
    gray_image[:] += 0.7152 * rgb[..., 2]
    return gray_image

# TODO: check the order rgb to confirm
def rgb2y(rgb):
    y_image     = 0.299 * rgb[..., 0]
    y_image[:] += 0.587 * rgb[..., 1]
    y_image[:] += 0.114 * rgb[..., 2]
    return y_image

def scale(image, hei_image, wid_image):
    return cv2.resize(image, (wid_image, hei_image),
                      interpolation=cv2.INTER_LINEAR)

def one_hot(n_classes, labels):
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in range(n_classes):
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels
