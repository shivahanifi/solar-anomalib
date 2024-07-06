# transforms.py
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
import torchvision.transforms.functional as F

IMAGE_SIZE = (256, 256)
CONTRAST_FACTOR = 1.2
GAMMA = 0.5
SATURATION_FACTOR = 1.2

transform_pv = Compose([
    ToTensor(),
    Resize(IMAGE_SIZE, antialias=True),
    Lambda(lambda img: F.adjust_contrast(img, CONTRAST_FACTOR)),
    Lambda(lambda img: F.adjust_saturation(img, SATURATION_FACTOR)),
    Lambda(lambda img: F.adjust_gamma(img, GAMMA)),
])
