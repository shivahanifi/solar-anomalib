from pathlib import Path
dataset_root = Path.cwd().parents[1] / "datasets" / "pv"
print(dataset_root)

import wandb
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, Compose, Lambda, ToTensor, ToPILImage
import torchvision.transforms.v2.functional as F

from anomalib.data import PredictDataset, Folder
from anomalib.engine import Engine
from anomalib.models import Fastflow
from anomalib.utils.post_processing import superimpose_anomaly_map
from anomalib import TaskType

wandb.login()

run = wandb.init(
    project="fastflow-anomaly",
    config={}
)

task = TaskType.SEGMENTATION
wandb.log({"task": task})

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
wandb.log({"transform": transform})


datamodule = Folder(
    name="pv",
    root=dataset_root,
    normal_dir="train/good",
    abnormal_dir="test/bad",
    transform=transform,
    task=TaskType.CLASSIFICATION,
    image_size=(256, 256),
)

Fastflow

model = Fastflow(backbone="resnet18", pre_trained=True, flow_steps=8)
engine = Engine()
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    ckpt_path="/home/shiva/Documents/code/anomalib/notebooks/200_models/results/Fastflow/MVTec/bottle/v0/checkpoints/epoch=13-step=98.ckpt",
)     
wandb.log({"predictions":predictions})             
wandb.log({"roc":wandb.plot.roc_curve(predictions[0]["label"], predictions[0]["box_labels"], labels=None, classes_to_plot=None)})             

test_result = engine.test(datamodule=datamodule, model=model)
wandb.log({"image_AUROC": test_result[0]["image_AUROC"], "image_F1Score": test_result[0]["image_F1Score"]})

print(dataset_root)

inference_dataset = PredictDataset(path=dataset_root / "test/bad/110M90500098_DEFECTUEUX__panel_0011__Cam-5_3540.jpg")

inference_dataloader = DataLoader(dataset=inference_dataset)

for batch in inference_dataloader:
    print(batch.keys())  # Inspect the keys of the dictionary
    break

for batch in inference_dataloader:
    inputs = batch["image"]
    print(inputs.shape)


predictions = engine.predict(model=model, dataloaders=inference_dataloader)[0]

print(
    f'Image Shape: {predictions["image"].shape},\n'
    'Anomaly Map Shape: {predictions["anomaly_maps"].shape}, \n'
    'Predicted Mask Shape: {predictions["pred_masks"].shape}',
)