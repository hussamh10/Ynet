import numpy as np
from cv2 import imread
import os
import cv2


def getImage(i, source, main_dir, ext, size):
    name = str(i) + ext
    print(main_dir + source + '\\' + name)

    path = os.path.join(main_dir, source , name)
    img = imread(path, 0)

    img = cv2.resize(img, size)
    img = img.reshape((img.shape[0], img.shape[1], 1))

    img = img.astype('float32')
    img /= 255

    return img

def generate(size, max=6):
    folder_num = 1
    while(folder_num < max):
        dir = os.path.join('..', 'data', 'data', str(folder_num))
        frame_ext = '.jpg'
        frame_size = (224, 224)

        i = 1

        while i < size:
            x1 = getImage(i, '', dir, frame_ext, frame_size)
            x2 = getImage(i+1, '', dir, frame_ext, frame_size)
            x3 = getImage(i+2, '', dir, frame_ext, frame_size)

            x1 = x1.reshape((1, x1.shape[0], x1.shape[1], x1.shape[2]))
            x2 = x2.reshape((1, x2.shape[0], x2.shape[1], x2.shape[2]))
            x3 = x3.reshape((1, x3.shape[0], x3.shape[1], x3.shape[2]))

            x1_2 = np.concatenate([x1, x2], axis=3)

            i += 1

            yield(x1_2, x3)

        folder_num += 1
        if(folder_num == max):
            folder_num = 1
