# sourceseparation_misc
* We have three main files: 
	* main.py - this file is for source separation. The cases it handles are: 2d- gmm toy example, mnist digits, and synthetic audio source separation.
	* main_timit.py - source separation experiment on the TIMIT dataset.
	* main_singlegenerativemodel.py - this file only trains a generative model and doesn't do the source separation experiment. 
* In all cases if the argument --tr_method adversarial is used, the training is done adversarially, if --tr_method ML is used, maximum likelihood training is used. 
