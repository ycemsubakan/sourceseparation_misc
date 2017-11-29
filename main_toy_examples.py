import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import argparse
import pdb
import matplotlib.pyplot as plt
from drawnow import drawnow, figure
import torch.utils.data as data_utils
from itertools import islice
from gan_things import *
import utils as ut
import pickle

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=2000, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--task', type=str, default='toy_data', metavar='task',
                    help='Seperation task')
parser.add_argument('--optimizer', type=str, default='RMSprop', metavar='optim',
                    help='Optimizer')
parser.add_argument('--tr_method', type=str, default='adversarial')
parser.add_argument('--test_method', type=str, default='optimize')

parser.add_argument('--input_type', type=str, default='noise')
parser.add_argument('--save_files', type=int, default=1)
parser.add_argument('--EP_train', type=int, default=400)
parser.add_argument('--EP_test', type=int, default=2000)
parser.add_argument('--save_records', type=int, default=1)
parser.add_argument('--data', type=str, default='spoken_digits', help='spoken_digits or synthetic_sounds')
parser.add_argument('--L1', type=int, default=200)
parser.add_argument('--feat_match', type=int, default=0) 
parser.add_argument('--load_models', type=int, default=0)
parser.add_argument('--adjust_tradeoff', type=int, default=0)
parser.add_argument('--smooth_output', type=int, default=0)
parser.add_argument('--plot_training', type=int, default=1)
parser.add_argument('--num_means', type=int, default=1)

parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--noise_type', type=str, default='gaussian')
parser.add_argument('--pack_num', type=int, default=2)

arguments = parser.parse_args()

arguments.cuda = not arguments.no_cuda and torch.cuda.is_available()

torch.manual_seed(arguments.seed)
if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)

exp_info = '_'.join([arguments.tr_method, 
                     arguments.data, 
                     arguments.input_type, arguments.optimizer, 
                     'feat_match', str(arguments.feat_match)])

tr_method = arguments.tr_method
loss = 'Poisson'

if arguments.task == 'images':
    if arguments.data == 'mnist':
        train_loader, test_loader = ut.get_loaders(1000, arguments=arguments)
        loader1, loader2, loader_mix = ut.form_mixtures(0, 1, train_loader, arguments)

        arguments.smooth_output = True
        arguments.L2 = 28*28
        arguments.nfts = 28
        arguments.T = 28
        arguments.K = 50

elif arguments.task == 'spoken_digits':
    loader1, loader2, loader_mix = ut.form_spoken_digit_mixtures(digit1=0, digit2=1, arguments=arguments)
    arguments.smooth_output = False
elif arguments.task == 'atomic_sourcesep':
    loader1, loader2, loader_mix = ut.preprocess_audio_files(arguments=arguments)
    arguments.smooth_output = False
elif arguments.task == 'toy_data':
    loader1, loader_mix = ut.prepare_mixture_gm_data(arguments=arguments)
else:
    raise ValueError('I do not know which task is that')


exp_info = '_'.join([arguments.tr_method,
                     arguments.task,
                     arguments.input_type, 
                     arguments.optimizer, 
                     'num_means', str(arguments.num_means),
                     'K', str(arguments.K),
                     'L1', str(arguments.L1)])
arguments.exp_info = exp_info

ngpu = 1
L1 = arguments.L1
L2 = arguments.L2
K = arguments.K
smooth_output = arguments.smooth_output

generator1 = netG_images(ngpu, K=K, L1=L1, L2=L2, arguments=arguments)
discriminator1 = netD(ngpu, K=K, L=L2, arguments=arguments)

if arguments.cuda:
    generator1.cuda()
    discriminator1.cuda()

# Train the generative models for the sources
if arguments.load_models:
    exp_info = '_'.join([arguments.tr_method, 
                         arguments.data, 
                         arguments.input_type, arguments.optimizer, 
                         'feat_match', str(arguments.feat_match)])
    modelfldr = 'model_parameters'
    generator1.load_state_dict(torch.load(os.path.join(modelfldr, 'generator1_' + exp_info + '.trc')))

    discriminator1.load_state_dict(torch.load(os.path.join(modelfldr, 'discriminator1_' + exp_info + '.trc')))
else:
    EP = arguments.EP_train
    if tr_method == 'adversarial':
        criterion = nn.BCELoss()
        adversarial_trainer(loader_mix=loader_mix,
                            train_loader=loader1,
                            generator=generator1, 
                            discriminator=discriminator1, 
                            EP=EP,
                            arguments=arguments,
                            criterion=criterion,
                            conditional_gen=False,
                            source_num=1)

    elif tr_method == 'adversarial_wasserstein':
        criterion = nn.BCELoss()
        adversarial_wasserstein_trainer(loader_mix=loader_mix,
                            train_loader=loader1,
                            generator=generator1, 
                            discriminator=discriminator1, 
                            EP=EP,
                            arguments=arguments,
                            criterion=criterion,
                            conditional_gen=False,
                            source_num=1)


    elif tr_method == 'moment_match':
        criterion = nn.MSELoss()
        moment_trainer(loader_mix=loader_mix,
                            train_loader=loader1,
                            generator=generator1, 
                            discriminator=discriminator1, 
                            EP=EP,
                            arguments=arguments,
                            criterion=criterion,
                            conditional_gen=False,
                            source_num=1)



    elif tr_method == 'ML':
        if loss == 'Euclidean': 
            criterion = nn.MSELoss()
        elif loss == 'Poisson':
            eps = 1e-20
            criterion = lambda lam, tar: torch.mean(-tar*torch.log(lam+eps) + lam)
        generative_trainer(loader_mix=loader_mix,
                           train_loader=loader1,
                           generator=generator1, 
                           EP=EP,
                           arguments=arguments,
                           criterion=criterion,
                           conditional_gen=False)
        
    # save models
    savepath = os.path.join(os.getcwd(), 'model_parameters')
    if not os.path.exists(savepath):
        os.mkdir(savepath) 

    ut.save_models([generator1], [discriminator1], exp_info,
                    savepath, arguments)

if arguments.task == 'images':
    Nsamples = 16
    samples = sample_outputs(generator1, Nsamples, arguments) 
    savepath = os.path.join(os.getcwd(), 'generated_samples')
    if not os.path.exists(savepath):
        os.mkdir(savepath) 

    ut.save_image_samples(samples, savepath, exp_info, 'generated', arguments)

    _, real_samples, _ = loader1.__iter__().next()
    if arguments.tr_method == 'adversarial':
        real_samples = real_samples[:Nsamples].contiguous().view(-1, arguments.L2)
    else:
        real_samples = real_samples[Nsamples:2*Nsamples].contiguous().view(-1, arguments.L2)
    ut.save_image_samples(real_samples, savepath, exp_info, 'real', arguments)



