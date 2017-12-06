from cv2 import imread
import numpy as np

import numpy as np
import cv2

def getImage(i, source, main_dir):
    #i = (i%10) + 1
    name = str(i) + '.jpg'
    print(main_dir + source + '\\' + name)
    img = imread(main_dir + source + '\\' + name, 0)
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype('float32')
    img /= 255
    return img

def getData(end, start=0, main_dir='data\\'):
    img_src = 'img'
    label_src = 'label'
    audio_src = 'audio'

    imgs = []
    labels = []
    audios = []

    for i in range(start, end):
        i+=1 #off by one
        imgs.append(getImage(i, img_src, main_dir))
        labels.append(getImage(i, label_src, main_dir))
        audios.append(getImage(i, audio_src, main_dir))
        
    imgs = np.array(imgs)
    audios = np.array(audios)
    labels = np.array(labels)

    pairs = [imgs, audios]

    return pairs, labels
