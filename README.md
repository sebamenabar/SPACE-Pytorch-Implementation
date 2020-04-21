# SPACE Pytorch
Note: If you don't see any progress (commits) don't despair, I'm still working on this (10/04/20)

This is an attempt to replicate SPACE: Unsupervised Object-Oriented Scene Representation via Spatial Attention and Decomposition https://arxiv.org/abs/2001.02407 on Pytorch. Most of the code is a clone from https://github.com/NVlabs/SSV.

A model that combines the Self-Supervised Viewpoint Learning from Image Collections (SSV) (http://arxiv.org/abs/2004.01793) encoder and the SPACE decoder is currently working on the CLEVR dataset with a very simple train look with few constraints.

Current used files are:
	- space-only.ipynb
	- utils/network_blocks.py
	- utils/space_modules.py
	- ssv.py
	- extern/network_blocks.py
	- clevr_dataset.py

For training and results please see the space-only notebook.

![space-clevr](images/space-clevr.png])
Took about 3 epochs to train, ~20 minutes per epoch on a GTX1080Ti. Weights used to generate this image:
https://drive.google.com/file/d/1F4KIOP2sxXrNFr33w5p1lHvo3daYWBna/view?usp=sharing
