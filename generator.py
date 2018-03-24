import numpy as np
import cv2
from cv2 import imread
import cv2
import os


def getImage(i, source, main_dir, ext, prefix):
    name = prefix + str(i) + ext
    path = os.path.join(main_dir, source , name)
    img = imread(path, 0)

    img = imread(main_dir + source + '/' + name, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = cv2.resize(img, (224, 224))
    img = img.reshape(224, 224, 1)
    img = img.astype('float32')
    img /= 255

    return img


def generate(size, max=6):
    folder_num = 1
    while(folder_num < max):
        dir = os.path.join('..', 'data', 'data', str(folder_num))
        frame_ext = '.jpg'
        audio_ext = '.png'
        audio_pre = ''
        video_pre = ''
        frame_size = (224, 224)

        i = 1

        while i < size:
            x = getImage(i  , '' , dir, frame_ext, video_pre)
            a = getImage(i  , '' , dir, audio_ext, audio_pre)
            g = getImage(i+1, '' , dir, frame_ext, video_pre)

            x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
            a = a.reshape((1, a.shape[0], a.shape[1], a.shape[2]))
            g = g.reshape((1, g.shape[0], g.shape[1], g.shape[2]))

            i += 1

            yield([x, a], g)

        folder_num += 1
        if(folder_num == max):
            folder_num = 1
