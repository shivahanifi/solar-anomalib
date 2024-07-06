import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import to_pil_image

from anomalib.data.image.folder import Folder, FolderDataset
from anomalib import TaskType

from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine
from pathlib import Path

# Path to the dataset root directory
dataset_root = Path.cwd()/"datasets"/"panels"

# Anomalib datamodule for 'panels' dataset
folder_datamodule = Folder(
    name="panels",
    root=dataset_root,
    normal_dir="/home/shiva/Documents/code/anomalib/datasets/panels/train/good",
    abnormal_dir="/home/shiva/Documents/code/anomalib/datasets/panels/test/bad",
    task=TaskType.CLASSIFICATION,
    image_size=(256, 256),
    train_batch_size=5,
    eval_batch_size=5,
)
folder_datamodule.setup()

# Train images
i, data = next(enumerate(folder_datamodule.train_dataloader()))
print(data.keys(), data["image"].shape)

# Test images
i, data = next(enumerate(folder_datamodule.test_dataloader()))
print(data.keys(), data["image"].shape, data["mask"].shape)

img = to_pil_image(data["image"][0].clone())
plt.imshow(img)
plt.show()

# Initialize the datamodule, model and engine
datamodule = MVTec()
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)

predictions = engine.predict(
    datamodule=datamodule(),
    model=model,

)