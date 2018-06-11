# Generative sourceseparation with GANs.
* This repository contains code used in the paper [Generative Adversarial Source Separation](https://arxiv.org/pdf/1710.10779.pdf).
* We have several main files: 
	* main.py - this file is for source separation. The cases it handles are: 2d- gmm toy example, mnist digits, and synthetic audio source separation.
	* main_timit.py - Single source separation experiment on the TIMIT dataset.
	* main_timit_multiplefiles.py - This file to recreate our experiments in the paper, which implements separation with multiple speaker pairs.
* In all cases if the argument --tr_method adversarial is used, the training is done adversarially, if --tr_method ML is used, maximum likelihood training is used. 
	* main_toy_examples.py  - This main file is used to generate generate data from mixture of K spherical gaussian distributions. Example usage is: 
	* records/read_records_timit_cleaned.py - You can use this script to plot your results obtained with 'main_timit_multiplefiles.py', in order to generate a figure similar to results figure in the paper. 

```
python main_toy_examples.py --task toy_data --tr_method adversarial --EP_train 3000 --num_means 4 --optimizer RMSprop
```
These options would use the standard GAN training (tr_method) to train for 3000 iterations (EP_train), on a mixture of 4 gaussian (num_means), with RMSprop optimizer. 
