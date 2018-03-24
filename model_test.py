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

def test(start):
    md = get_unet()
    md.load_weights('ynet_15_3.hdf5')

    h, hy = gd(start = start, end = start + 10)
    
    hp = md.predict(h)

    i = start
    for p, g in zip(hp, hy):
        save(p.reshape((224, 224)), i, 'out/')
        save(g.reshape((224, 224)), i, 'out/gt_')
        i += 1
    print("Done")

test(1)
