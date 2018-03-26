from keras import backend as K
from keras import losses as L
import tensorflow as tf
import os
import numpy as np
from cv2 import imread
import cv2

def bce(y_true, y_pred):
    return K.mean(K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1))

def mse(y_true, y_pred):
    return K.mean(K.mean(L.mean_squared_error(y_true, y_pred), axis=-1))

def getError(a, b, error):
    sess = tf.Session()

    ta = tf.constant(a)
    tb = tf.constant(b)

    return(sess.run(error(ta, tb)))

def getRandom(size):
    return np.random.rand(size[0], size[1])

def getImage(i, source, main_dir, ext):
    name = str(i) + ext

    img = imread(os.path.join(main_dir, source,  name), 0)
    img = cv2.resize(img, (224, 224))
    img = img.reshape((img.shape[0], img.shape[1]))
    img = img.astype('float32')
    img /= 255

    return img

def testImages(error):
    #same
    i1 = getImage(1, 'same', 'test', '.jpg')
    i2 = getImage(1, 'same', 'test', '.jpg')
    
    print('Error for same images 1.jpg ', str(getError(i1, i2, error)))

    i1 = getImage(1, 'same', 'test', '.jpg')
    i2 = getImage(1, 'same', 'test', '.jpg')
    
    print('Error for same images 1.jpg but different order', str(getError(i2, i1, error)))

    i3 = getImage(2, 'same','test', '.jpg')
    i4 = getImage(2, 'same','test', '.jpg')

    print('Error for same images 2.jpg ', str(getError(i3, i4, error)))

    #black and black
    i1 = getImage(1, 'black', 'test', '.jpg')
    i2 = getImage(1, 'black', 'test', '.jpg')
    
    print('Error for black images 1.jpg ', str(getError(i1, i2, error)))

    i3 = getImage(2, 'black', 'test', '.jpg')
    i4 = getImage(2, 'black', 'test', '.jpg')

    print('Error for white images 2.jpg ', str(getError(i3, i4, error)))

    #black and image
    i1 = getImage(1, 'same', 'test', '.jpg')
    i2 = getImage(1, 'black', 'test', '.jpg')
    
    print('Error for regular image and black image 1.jpg ', str(getError(i1, i2, error)))

    i1 = getImage(1, 'black', 'test', '.jpg')
    i2 = getImage(1, 'same', 'test', '.jpg')
    
    print('Error for black image and regular image 1.jpg ', str(getError(i1, i2, error)))

    #random images

    i1 = getRandom((224, 224))
    i2 = np.copy(i1)
    
    print('Error for 2 same random arrays', str(getError(i1, i2, error)))

    #different images
    i1 = getImage(1, 'same', 'test', '.jpg')
    i2 = getImage(2, 'same', 'test', '.jpg')
    
    print('Error for vastly different regular image and regular image 1.jpg ', str(getError(i1, i2, error)))
    
    #frame_1 and frame_2
    i1 = getImage(1, 'frame', 'test', '.jpg')
    i2 = getImage(2, 'frame', 'test', '.jpg')
    
    print('Error for frames 1 and 2', str(getError(i1, i2, error)))

    #out and gt
    out = getImage(1, 'out', 'test', '.png')
    gt = getImage(2, 'out', 'test', '.png')
    
    print('Error for output of ENET and GT 1, 2', str(getError(gt, out, error)))

    out = getImage(3, 'out', 'test', '.png')
    gt = getImage(4, 'out', 'test', '.png')
    
    print('Error for output of ENET and GT 3, 4', str(getError(gt, out, error)))

print("BCE")
print("~")
testImages(bce)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("")
print("MSE")
testImages(mse)
