# Generative sourceseparation with GANs.
* This repository contains code used in the paper [Generative Adversarial Source Separation](https://arxiv.org/pdf/1710.10779.pdf).
* We have three main files: 
	* main.py - this file is for source separation. The cases it handles are: 2d- gmm toy example, mnist digits, and synthetic audio source separation.
	* main_timit.py - Single source separation experiment on the TIMIT dataset.
	* main_timit_multiplefiles.py - This file to recreate our experiments in the paper, which implements separation with multiple speaker pairs.
* In all cases if the argument --tr_method adversarial is used, the training is done adversarially, if --tr_method ML is used, maximum likelihood training is used. 
