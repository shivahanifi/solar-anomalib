import matplotlib.pyplot as plt
from PIL import Image

bottle_broken = Image.open("/home/shiva/Documents/code/anomalib/datasets/MVTec/bottle/test/broken_large/000.png")
plt.imshow(bottle_broken)
plt.show()
