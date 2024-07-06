import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.callbacks.visualizer import VisualizationCallback
from transforms import transform_pv

def load_config(config_path):
    """Reads and loads the YAML configuration from a given path.
"""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def evaluate_model(model_config_path, data_config):
    # Load configuration
    model_config = load_config(model_config_path)
    
    # Merge model configuration with data configuration
    config = {**model_config, **data_config}

    # Prepare data module
    data_module = get_datamodule(config)

    # Initialize model
    model = get_model(config)

    # Callbacks
    visualizer_callback = VisualizerCallback(
        task=config['model']['name'],
        image_save_path=config['evaluation']['output_path']
    )

    # Initialize trainer
    trainer = Trainer(
        max_epochs=1,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[visualizer_callback]
    )

    # Run evaluation
    trainer.test(model=model, datamodule=data_module)

# Configuration file paths
dataset_config_path = "/home/shiva/Documents/code/anomalib/configs/data/pv.yaml"
data_config = load_config(dataset_config_path)

model_config_files = ["/home/shiva/Documents/code/anomalib/configs/model/cflow.yaml",
                      "/home/shiva/Documents/code/anomalib/configs/model/padim.yaml"]

# Evaluate each model
for model_config_file in model_config_files:
    evaluate_model(model_config_file, data_config)
