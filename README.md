# MRI segmentation
This project aids researchers to get better segmentation results of MRI brain images.   
We improve the segmentation results towards two criteria:   
	1. the mask should be smooth and without noise  
	2. there should be minimum misclassification  
(There might be minor bugs since the version is not the most up-to-date)

This project features two aspects:

	1. It is able to take 3d patches
	2. The user can choose between Resnet(deeper) and Unet(require less memory consumption)

Training data used in this project:

	1. 2 high-resolution (0.5*0.5*0.5mm) data acquired at the BIC center at Beckman Institute in UIUC, and HCP data (0.7*0.7*0.7mm with different fov)
	2. Use HR images as prior information, warp the HCP data to a HR space for artificial data
	3. The crude segmentation is done using SPM toolbox



This script is revised from:
https://github.com/brade31919/SRGAN-tensorflow
