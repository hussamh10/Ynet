import numpy as np
from cv2 import imread


def getImage(i, source, main_dir):
    name = str(i) + '.jpg'

    print(main_dir + source + '\\' + name)

    img = imread(main_dir + source + '\\' + name, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.reshape(224, 224, 1)
    img = img.astype('float32')
    img /= 255
    return img

def generate(size):
    dir = 'data\\'
    src_audio = 'audio\\'
    src_x2 = '2\\'
    src_x3 = '3\\'

    i = 0

    while i < size:
        a = getImage(i, src_audio, dir)
        x2 = getImage(i, src_x2, dir)
        x3 = getImage(i, src_x3, dir)

        a = a.reshape((1, a.shape[0], a.shape[1], a.shape[2]))
        x2 = x2.reshape((1, x2.shape[0], x2.shape[1], x2.shape[2]))
        x3 = x3.reshape((1, x3.shape[0], x3.shape[1], x3.shape[2]))

        i += 1

        yield([a, x2], x3)


