# HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images

A multiple branch network that performs nuclear instance segmentation and classification within a single network. The network leverages the horizontal and vertical distances of nuclear pixels to their centres of mass to separate clustered cells. A dedicated up-sampling branch is used to classify the nuclear type for each segmented instance. <br />

```

## Dataset

Download the CoNSeP dataset as used in our paper from [this link](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/). <br />

## Repository Structure

- `src/` contains executable files used to run the model. Further information on running the code can be found in the corresponding directory.
- `loader/`contains scripts for data loading and self implemented augmentation functions.
- `metrics/`contains evaluation code. 
- `misc/`contains util scripts. 
- `model/` contains scripts that define the architecture of the segmentation models. 
- `opt/` contains scripts that define the model hyperparameters. 
- `postproc/` contains post processing utils. 
- `config.py` is the configuration file. Paths need to be changed accordingly.
- `train.py` and `infer.py` are the training and inference scripts respectively.
- `process.py` is the post processing script for obtaining the final instances. 
- `extract_patches.py` is the patch extraction script. 

## HoVer-Net

![](diagram.png)

## Overlaid Segmentation and Classification Prediction

<p float="left">
  <img src="/seg.gif" alt="Segmentation" width="870" />
</p>

The colour of the nuclear boundary denotes the type of nucleus. <br />
Blue: epithelial<br />
Red: inflammatory <br />
Green: spindle-shaped <br />
Cyan: miscellaneous

## Additional Implementations available 
 
* [FCN8](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
* [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
* [SegNet](https://arxiv.org/pdf/1511.00561.pdf)
* [DCAN](https://www.sciencedirect.com/science/article/abs/pii/S1361841516302043) 
* [Micro-Net](https://www.sciencedirect.com/science/article/abs/pii/S1361841518300628)
* [DIST](https://ieeexplore.ieee.org/document/8438559)

## Getting Started

Install the required libraries before using this code. Please refer to `requirements.txt`

## Results

All comparative results on the CoNSeP, Kumar and CPM-17 datasets can be found [here](https://drive.google.com/drive/folders/1WTkleeaE6ne8qxuYzptv2bKwMdZVBpzr?usp=sharing). 

## Extra

The cell profiler pipeline that we used in our comparative experiments can be found [here](https://drive.google.com/file/d/1E5UII9fsYT2N2KBUNLS89OV9AstYDLlZ/view?usp=sharing).



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


