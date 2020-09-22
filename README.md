# IONet
This repo is the implementation of paper **Weakly-Supervised Action Localization via Embedding-Modeling Iterative Optimization.

## Recommended Environment
* Python 3.6
* Cuda 9.0
* PyTorch 1.1.0

## Prerequisites
* Install dependencies: `pip install -r requirements.txt`.
* [Install Matlab API for Python](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) (matlab.engine).
* Prepare THUMOS14 and ActivityNet datasets.

### Feature Extraction
We employ I3D features in the paper. 

**We recommend to extract the features using the followingf repo:**
* [I3D Features](https://github.com/Finspire13/pytorch-i3d-feature-extraction)

## Run

1. For training the model:
```
python train.py
```

2. For testing the model:
```
python test.py
```

The final results are saved in **.npz** format.

## Citation

## License
MIT

