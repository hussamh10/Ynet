from cv2 import imread
import os
import numpy as np
import cv2

def getImage(i, source, main_dir, ext):
    name = str(i) + ext

    print(main_dir + source + '/' + name, 0)

    img = imread(main_dir + source + '/' + name, 0)
    img = cv2.resize(img, (224, 224))
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype('float32')
    img /= 255
    return img

def getData(end, start=0, main_dir='../data/data', video=1):

    main_dir = os.path.join(main_dir, str(video))

    imgs = []
    labels = []
    audios = []

    for i in range(start, end):
        imgs.append(getImage(i, '', main_dir, '.jpg'))
        audios.append(getImage(i, '', main_dir, '.png'))
        labels.append(getImage(i+1, '', main_dir, '.jpg'))
        i+=1
        
    imgs = np.array(imgs)
    audios = np.array(audios)
    labels = np.array(labels)

    pairs = [imgs, audios]

    return pairs, labels
