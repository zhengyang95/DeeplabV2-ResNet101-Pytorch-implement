# Segmentation network training and inference
This repository contains allows you to train a segmentation network (DeepLabV2) which is based on ResNet-101.

After the network is trained, you can infering pixel-wise masks for given new images and evaluate the accuracy of these masks by using mean Intersection-over-Union (mIoU) metric.

Remember to change all files' paths to your own paths.

## Prerequisite
* Python 3.7, PyTorch 1.1.0, and more in requirements.txt
* PASCAL VOC 2012 devkit
* NVIDIA GPU with more than 1024MB of memory

## Usage

#### Install python dependencies
```
pip install -r requirements.txt
```
#### Download PASCAL VOC 2012 devkit
* Follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

#### train the segmentation network 
```
python train_seg.py
```
* ALL 3 executing scripts exist in the script directory.
* The segmentation network can be trained by the ground truth masks or the pseudo-masks generated in a weakly supervised case.

#### Infering masks (on validation set) by the trained segmentation network 
```
python infer.py
```
* An available pretrained weights (in fully supervised case, i.e., trained by ground truth masks) is online: https://drive.google.com/file/d/12oLbmwwEbD8vpGH7QwogSggHNS4eTSiT/view?usp=sharing (mIoU on validation set is 77.78).

#### Evaluation the infered masks using mIoU metric
```
python results_evaluation.py
```
