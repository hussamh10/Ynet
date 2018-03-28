from matplotlib import pyplot as plt
from generator_random import generate as g

gen = g(100, 2)

while(1):
    [x, a], y = next(gen)

    x = x.reshape(224, 224)
    plt.imshow(x, cmap="gray")
    plt.show()
