import os
from unet import get_unet
from hussam_data import getData as gd
from matplotlib import pyplot as plt
import keras

def save(data, i, path):
    data = data.astype('float32')
    img = plt.imshow(data, interpolation='nearest')
    img.set_cmap('gray')
    plt.axis('off')
    plt.savefig(path + str(i) + ".png", bbox_inches='tight')

def test(start, video, tag):
    md = get_unet()
    md.load_weights('ynet_single_video.hdf5')

    h, hy = gd(start = start, end = start + 10, main_dir='../data/data', video=video)
    
    hp = md.predict(h)

    if not os.path.exists(tag):
        os.makedirs(tag)

    i = start
    for p, g in zip(hp, hy):
        save(p.reshape((224, 224)), i, tag)
        save(g.reshape((224, 224)), i, tag+'gt')
        i += 1
    print("Done")

parent = 'out_multi_video'

test(1, 1, parent + '/training_data/')
test(100, 1,  parent +'/similar_training_data/')
test(1, 2,  parent +'/traingin_data2/')
test(1, 4,  parent +'/traingin_data3/')
test(1, 6,  parent +'/traingin_data4/')
test(1, 10,  parent +'/test2/')
test(1, 22,  parent +'/test3/')
test(1, 42,  parent +'/test4/')
