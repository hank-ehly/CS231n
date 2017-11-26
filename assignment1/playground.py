import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('./test.jpeg')
print(img[0][0])
img_tinted = np.uint8(img * [0.45, 0.45, 0.45])
print(img_tinted[0][0])

plt.subplot(2, 1, 1)
plt.imshow(img)

plt.subplot(2, 1, 2)
plt.imshow(img_tinted)
plt.show()
