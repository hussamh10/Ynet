from unet import get_unet
from hussam_data import getData as gd
from matplotlib import pyplot as plt
import keras


def test(start):
    md = get_unet()
    md.load_weights('ynet.hdf5')

    h, hy = gd(start = start, end = start + 10)

    del hy
    
    print("predicting")

    hp = md.predict(h)

    i = start+1
    for p in hp:
        plt.imshow(p.reshape((224, 224)))
        plt.savefig('' + 'out\\' + str(i) + '.png')
        print(''  + 'out\\' + str(i) + '.png')
        i += 1

    print("Done")

for i in range(310, 400, 10):
    test(i)
