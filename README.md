# dlcv-ss20

DLCV Practical Course - SS20

# Setup
## Using conda 
```
conda create -n dlcv python=3.7
conda activate dlcv
conda install --file environment.yaml
pip install nonechucks
pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@master --upgrade
```
Install [kaolin](https://github.com/NVIDIAGameWorks/kaolin/) according to the instructions
on the official page.

# Run tests
Small parts of functionality are tested with unit tests. Test data is in
`tests/data`
```
python -m pytest
```
To print something during the tests, add the print statements and run
```
python -m pytest -s
```

# Basic Examples
## Pointnet classification + Tensorboard
First configure the experiment. Set the `data_path` and other values in `pointnet2_classification.yaml`. Then run 

```
python pointnet2_classification.py
```

Look at the tensorboard plot using 
```
tensorboard --logdir=PN2classification/v2
```

# PolyGen 
## Augmented meshes
First generate 50 scaled and warped meshes for each original mesh

```
python scripts/augment_ply.py /shared/data/shapenet_aug/ply_table_1 <output_dir>
```

Then apply decimation on each mesh. `input_dir` is the same as `output_dir` in the first step

```
blender -b -P transforms/decimation.py -- <input_dir> <output_dir>
```
The double dash is required to separate blender args from the Python script's args

## Get dataset stats and configure
Run this to get statistics of the dataset. Set parameters in the yaml accordingly.

```
python scripts/dataset_stats.py tests/configs/polygen.yaml
```

## Training
Train the PolyGen model. Configure the experiment in 
`tests/configs/polygen.yaml`.
Then run 
```
python scripts/train_polygen.py tests/configs/polygen.yaml
```
## Inference
Generate new meshes 
```
python scripts/test_polygen.py tests/configs/polygen.yaml
```

## Training with Pytorch-Lightning
Quick run on a subset of the data, use to set the batch size 
```
python scripts/train_vertex_lt.py tests/configs/augmented.yaml  --subset
```

Then remove the `--subset` flag to train on the full dataset.
Use `train_face_lt.py` to train the face model

## Inference with Pytorch-Lightning
Set the checkpoint path in the config `yaml` file, then run

```
python scripts/test_both.py tests/configs/augmented.yaml
```
