from unet import get_unet
from hussam_data import getData as gd
from matplotlib import pyplot as plt

def test():
    md = get_unet()
    md.load_weights('unet.hdf5')

    main_dir = 'test\\'

    h, hy = gd(start = 0, end = 1, main_dir='test\\')

    print("predicting")

    hp = md.predict(h)

    i = 0
    for p in hp:
        i += 1
        plt.imshow(p.reshape((224, 224)))
        plt.savefig(main_dir + 'out\\' + str(i) + '.jpg')
        print(main_dir + 'out\\' + str(i) + '.jpg')

    print("Done")

test()
