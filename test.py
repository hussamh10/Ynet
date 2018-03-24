import hussam_data as data
from matplotlib import pyplot as plt

[x, a], l = data.getData(start=1, end=2)

plt.imshow(x.reshape(224, 224), cmap='gray')
plt.show()
plt.imshow(a.reshape(224, 224), cmap='gray')
plt.show()
plt.imshow(l.reshape(224, 224), cmap='gray')
plt.show()

