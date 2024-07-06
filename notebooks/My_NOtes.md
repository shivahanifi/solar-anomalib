These are my notes during the implementations.

- To find the thorugh documentation of the anomalib you can use: [anomalib.readthedoc.io](https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html)

- If the visualization in notebooks does not work add the `%matplotlib inline`

## Task types
- CLASSIFICATION
- DETECTION
- SEGMENTATION

## Models:

- [CFA](https://arxiv.org/abs/2206.04325)
- [CS-Flow](https://arxiv.org/abs/2110.02855v1)
- [CFlow](https://arxiv.org/pdf/2107.12571v1.pdf)
- [DFKDE](https://github.com/openvinotoolkit/anomalib/tree/main/anomalib/models/dfkde)
- [DFM](https://arxiv.org/pdf/1909.11786.pdf)
- [DRAEM](https://arxiv.org/abs/2108.07610)
- [FastFlow](https://arxiv.org/abs/2111.07677)
- [Ganomaly](https://arxiv.org/abs/1805.06725)
- [Padim](https://arxiv.org/pdf/2011.08785.pdf)
- [Patchcore](https://arxiv.org/pdf/2106.08265.pdf)
- [Reverse Distillation](https://arxiv.org/abs/2201.10703)
- [R-KDE](https://ieeexplore.ieee.org/document/8999287)
- [STFPM](https://arxiv.org/pdf/2103.04257.pdf)

## Geeting started
- To have other categories tested in the [001_getting_started.ipynb](/home/shiva/Documents/code/anomalib/notebooks/000_getting_started/001_getting_started.ipynb) include the desired category name with the `datamodule = MVTec(num_workers=0, category="carpet")`

## Dataset.py file for the pv panels dataset
The dataset file should be created in the `anomalib/data/image` path and called when required.
Each `DATASET.py` file contains a Dataset class and a DataModule class. 
- Dataset:
    It defines how to load and preprocess individual data samples. It's typically used to encapsulate data loading logic, including reading from files, applying transformations, and providing the data in a format suitable for training or evaluation.
- [Lightening DataModule](https://lightning.ai/docs/pytorch/latest/data/datamodule.html#prepare-data): 
    encapsulating the entire data loading and processing pipeline, including training, validation, and test data loaders. It abstracts and standardizes the process of data preparation, which helps in organizing the code and making it more reusable and easier to manage.
    1. [prepare_data](https://lightning.ai/docs/pytorch/latest/data/datamodule.html#prepare-data)
    2. [setup](https://lightning.ai/docs/pytorch/latest/data/datamodule.html#setup)
    3. [train_dataloader](https://lightning.ai/docs/pytorch/latest/data/datamodule.html#setup)
    4. [test_dataloader](https://lightning.ai/docs/pytorch/latest/data/datamodule.html#test-dataloader)
    5. [predict_dataloader](https://lightning.ai/docs/pytorch/latest/data/datamodule.html#predict-dataloader)

## SOTA
based on the performance of the [sota models on MVTeC dataset](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad) the following models from anomalib are the best ones to be tried.

1. EfficientAD #2
2. PatchCore large #11
3. reverse distillation ++ #19
4. DSR #44
5. Dream #47
6. PaDim #49
