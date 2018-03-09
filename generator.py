import numpy as np
import cv2
from cv2 import imread
import os


def getImage(i, source, main_dir, ext, size):
    name = str(i) + ext
    print(main_dir + source + '/' + name)

    path = os.path.join(main_dir, source , name)
    img = imread(path, 0)

    img = cv2.resize(img, size)
    img = img.reshape((img.shape[0], img.shape[1], 1))

    img = img.astype('float32')
    img /= 255

    return img

def generate(size):
    dir = os.path.join('..', 'data', 'data', '0')
    audio_ext = '.png'
    audio_size = (953, 273)
    frame_ext = '.jpg'
    frame_size = (953, 273)

    i = 1

    while i < size:
        a = getImage(i, '', dir, audio_ext, audio_size)
        x2 = getImage(i, '', dir, frame_ext, frame_size)
        x3 = getImage(i+1, '', dir, frame_ext, frame_size)

        a = a.reshape((1, a.shape[0], a.shape[1], a.shape[2]))
        x2 = x2.reshape((1, x2.shape[0], x2.shape[1], x2.shape[2]))
        x3 = x3.reshape((1, x3.shape[0], x3.shape[1], x3.shape[2]))

        print(a.shape)

        i += 1

        yield([a, x2], x3)
