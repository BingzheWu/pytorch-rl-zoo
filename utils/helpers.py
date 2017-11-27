import logging
import numpy as numpy
import cv2

from collections import namedtuple

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')
