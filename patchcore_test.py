from pathlib import Path

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


from transforms import transform_pv
from anomalib import TaskType


dataset_root = Path.cwd()/ "datasets" / "pv"
print("Dataset root is: ", dataset_root)

# Initialize the datamodule, model and engine

datamodule = Folder(
    name="pv",
    root=dataset_root,
    normal_dir="train/good",
    abnormal_dir="test/bad",
    transform=transform_pv,
    task=TaskType.CLASSIFICATION,
    image_size=(256, 256),
)

datamodule.setup()

model = Patchcore()
engine = Engine()

