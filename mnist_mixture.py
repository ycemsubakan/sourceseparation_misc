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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

arguments = parser.parse_args()
arguments.cuda = not arguments.no_cuda and torch.cuda.is_available()

torch.manual_seed(arguments.seed)
if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)

batch_size = 1000
data = 'mnist'
tr_method = 'adversarial'
loss = 'Poisson'

train_loader, test_loader = get_loaders(data, batch_size, arguments=arguments)

loader1, loader2, loader_mix = form_mixtures(0, 1, train_loader, arguments)

K = 100
ngpu = 1
L = 28*28

generator1 = netG(ngpu, K=K, L=L)
discriminator1 = netD(ngpu, K=K, L=L)

generator2 = netG(ngpu, K=K, L=L)
discriminator2 = netD(ngpu, K=K, L=L)

if arguments.cuda:
    generator1.cuda()
    discriminator1.cuda()

    generator2.cuda()
    discriminator2.cuda()

fixed_noise = torch.FloatTensor(arguments.batch_size, L).normal_(0, 1)
if arguments.cuda:
    fixed_noise = fixed_noise.cuda()

# Train the generative models for the sources
EP = 20
if tr_method == 'adversarial':
    criterion = nn.BCELoss()
    adverserial_trainer(loader_mix=loader_mix,
                        train_loader=loader1,
                        generator=generator1, 
                        discriminator=discriminator1, 
                        EP=EP,
                        arguments=arguments,
                        criterion=criterion,
                        fixed_noise=fixed_noise,
                        conditional_gen=False)

    adverserial_trainer(loader_mix=loader_mix,
                        train_loader=loader2,
                        generator=generator2, 
                        discriminator=discriminator2, 
                        EP=EP,
                        arguments=arguments,
                        criterion=criterion,
                        fixed_noise=fixed_noise,
                        conditional_gen=False)
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
                       fixed_noise=fixed_noise,
                       conditional_gen=False)

    generative_trainer(loader_mix=loader_mix,
                       train_loader=loader2,
                       generator=generator2, 
                       EP=EP,
                       arguments=arguments,
                       criterion=criterion,
                       fixed_noise=fixed_noise,
                       conditional_gen=False)

 

###

# Separate out the sources 
maxlikelihood_separatesources(generators=[generator1, generator2],
                              loader_mix=loader_mix,
                              EP=2000,
                              arguments=arguments,
                              conditional=False,
                              data='mnist',
                              tr_method=tr_method,
                              loss=loss)

####
