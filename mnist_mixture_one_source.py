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

# experiment settings
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
train_loader, test_loader = get_loaders(data, batch_size, arguments=arguments)

loader1, loader2, loader_mix = form_mixtures(0, 1, train_loader, arguments)

K = 100
ngpu = 1
L = 28*28
tr_method = 'adversarial'
loss = 'Poisson'

generator1 = netG(ngpu, K=K, L=L)
discriminator1 = netD(ngpu, K=K, L=L)

if arguments.cuda:
    generator1.cuda()
    discriminator1.cuda()

criterion = nn.BCELoss()

EP = 20 
if tr_method == 'adversarial':
    adverserial_trainer(loader_mix=loader_mix,
                        train_loader=loader1,
                        generator=generator1, 
                        discriminator=discriminator1, 
                        EP=EP,
                        arguments=arguments,
                        criterion=criterion,
                        fixed_noise=None,
                        conditional_gen=True)
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
                       fixed_noise=None,
                       conditional_gen=True)


###
EP = 2000
x1hat, x2hat = [], []
mixes = []
for i, (mix, _ ) in enumerate(islice(loader_mix,0,1,1)): 
    if arguments.cuda:
        mix = mix.cuda()

    mix = mix[:20]

    print('Processing source ',i)
    Nmix = mix.size(0)
    mix_rshape = mix.view(Nmix, L)
        
    if arguments.cuda:
        x1 = Variable(torch.rand(Nmix, L).cuda(), requires_grad=True)
    else:
        x1 = Variable(torch.rand(Nmix, L), requires_grad=True)

    optimizer_sourcesep = optim.Adam([x1], lr=1e-3, betas=(0.5, 0.999))
    for ep in range(EP):
       
        mix_sum = generator1.forward(x1) 
        if loss == 'Euclidean': 
            err = torch.mean((Variable(mix) - mix_sum)**2)
        elif loss == 'Poisson':
            eps = 1e-20
            err = torch.mean(-Variable(mix)*torch.log(mix_sum+eps) + mix_sum)

        err.backward()

        optimizer_sourcesep.step()

        x1.grad.data.zero_()

        print('Step in batch [{:d}\{:d}]'.format(ep+1, EP))
        print('The error is ', err)
    x1hat.append(generator1.forward(Variable(mix_rshape)).data.cpu().numpy())
    mixes.append(mix.cpu().numpy())

num_ims, c = 10, 2
figure(num=None, figsize=(3*c, num_ims*c), dpi=80, facecolor='w', edgecolor='k')
sqrtL = int(np.sqrt(L))
for i in range(num_ims):
    plt.subplot(num_ims, 2, 2*i + 1)
    plt.imshow(x1hat[0][i].reshape(sqrtL, sqrtL))
    plt.title('Estimated Source ')


    plt.subplot(num_ims, 2, 2*i + 2)
    plt.imshow(mixes[0][i].reshape(sqrtL, sqrtL))
    plt.title('Mixture')

figname = '_'.join([data, tr_method, 
                    'conditional', str(True), 'sourceseparation'])
plt.savefig(figname)

####
