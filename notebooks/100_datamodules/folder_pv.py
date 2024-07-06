""" PV dataset 
Use `Folder` for PV dataset Datasets

Use Folder Dataset (for Custom Datasets) via API
Here we show how one can utilize custom datasets to train anomalib models. 
A custom dataset in this model can be of the following types:
- A dataset with good and bad images.
- A dataset with good and bad images as well as mask ground-truths for pixel-wise evaluation.
- A dataset with good and bad images that is already split into training and testing sets.
"""
from pathlib import Path
from PIL import Image
from torchvision.transforms.v2 import Resize, Compose, Lambda, ToTensor, ToPILImage
import torchvision.transforms.v2.functional as F
import matplotlib.pyplot as plt
import numpy as np

from anomalib.data.image.folder import Folder, FolderDataset
from anomalib import TaskType

# path to the dataset root directory
dataset_root = Path.cwd() / "datasets" / "conform_3"

IMAGE_SIZE = (256, 256)
CONTRAST_FACTOR = 1.2
#BRIGHTNESS_FACTOR = 0.5
GAMMA = 0.5
SATURATION_FACTOR = 1.2
transform = Compose([
    ToTensor(),
    Resize(IMAGE_SIZE, antialias=True),
    Lambda(lambda img: F.adjust_contrast_image(img, CONTRAST_FACTOR)),
    #Lambda(lambda img: F.equalize(img)),
    #Lambda(lambda img: F.adjust_brightness(img, brightness_factor)),
    Lambda(lambda img: F.adjust_saturation(img,SATURATION_FACTOR)),
    Lambda(lambda img: F.adjust_gamma(img, GAMMA)),
    #ConvertImageDtype(torch.uint8),

])

input_image = Image.open(dataset_root/"good"/"Cam 0_1798.jpg")
plt.imshow(input_image)
plt.title("input sample")
plt.show()

folder_datamodule = Folder(
    name="conform_3",
    root=dataset_root,
    normal_dir="good",
    abnormal_dir="bad",
    transform=transform,
    task=TaskType.CLASSIFICATION,
    image_size=(256, 256),
)
folder_datamodule.setup()

# Train images
i, data = next(enumerate(folder_datamodule.train_dataloader()))

image_to_show = data['image'][0].numpy().transpose((1,2,0))*255
# Create a PIL Image from the numpy array
image_pil = Image.fromarray(np.uint8(image_to_show))
plt.imshow(image_pil)
plt.title("datamodule_image sample")
plt.show()


# Test images
i, data = next(enumerate(folder_datamodule.test_dataloader()))


folder_dataset_classification_train = FolderDataset(
    name="conform_3",
    normal_dir=dataset_root / "good",
    abnormal_dir=dataset_root / "bad",
    split="train",
    transform=transform,
    task=TaskType.CLASSIFICATION,
)

# Visualize train samples
img_path = folder_dataset_classification_train.samples.iloc[0,0]
image = Image.open(img_path)
plt.imshow(image)
plt.title("train sample")
plt.show()

data = folder_dataset_classification_train[0]

#img = Image.open("/home/shiva/Documents/code/pvdetects/data/Archives_Inspection/10M91204730_CONFORME/panel_0018/Cam 0_10587.jpg")
img = F.to_pil_image(data["image"][0].clone())
transformed_img = transform(img)
transformed_pil_img = ToPILImage()(transformed_img)
print(transformed_pil_img.size)

plt.imshow(transformed_pil_img)
plt.title("double transform") 
plt.show()

# Folder Classification Test Set
folder_dataset_classification_test = FolderDataset(
    name="conform_3",
    normal_dir=dataset_root / "good",
    abnormal_dir=dataset_root / "bad",
    split="test",
    transform=transform,
    task=TaskType.CLASSIFICATION,
)
data = folder_dataset_classification_test[0]
