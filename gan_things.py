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
import os
import torch.nn.init as torchinit

def initializationhelper(param, nltype):
    torchinit.xavier_uniform(param.weight, gain=torchinit.calculate_gain(nltype))
    c = 0.01
    torchinit.uniform(param.bias, a=-c, b=c)

class netG(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(netG, self).__init__()
        self.ngpu = ngpu
        pl = 40
        self.L1 = kwargs['L1']
        self.L2 = kwargs['L2']
        self.K = kwargs['K']
        self.l1 = nn.Linear(self.L1, self.K+pl, bias=True)
        initializationhelper(self.l1, 'tanh')

        self.l2 = nn.Linear(self.K+pl, self.L2, bias=True) 
        initializationhelper(self.l1, 'relu')

        self.smooth_output = kwargs['smooth_output']
        if kwargs['smooth_output']:
            self.sml = nn.Conv2d(1, 1, 5, padding=2) 
            initializationhelper(self.sml, 'relu')

    def forward(self, inp):
        inp = inp.view(-1, self.L1)

        h = F.tanh(self.l1(inp))
        output = F.softplus(self.l2(h))

        if self.smooth_output:
            output = output.view(-1, 1, int(np.sqrt(self.L2)), int(np.sqrt(self.L2)))
            output = F.softplus(self.sml(output))
            output = output.view(-1, self.L2)
        return output

class netD(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(netD, self).__init__()
        self.ngpu = ngpu
        self.L = kwargs['L']
        self.K = kwargs['K']
        self.l1 = nn.Linear(self.L, self.K, bias=True)
        initializationhelper(self.l1, 'tanh')

        self.l2 = nn.Linear(self.K, 1, bias=True) 
        initializationhelper(self.l1, 'relu')


    def forward(self, inp):
        inp = inp.view(-1, self.L) 

        h = F.tanh(self.l1(inp))
        output = F.sigmoid(self.l2(h))
        return output

class net_separation(nn.Module):
    def __init__(self, net1, net2, inpsize, L):
        super(net_separation, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.inp1 = Variable(torch.FloatTensor(inpsize, L))

    def forward(self):
        out1 = self.net1.forward(self.inp1)
        out2 = self.net2.forward(self.inp2)
        return out1+out2, out1, out2

def adverserial_trainer(loader_mix, train_loader, 
                        generator, discriminator, EP = 5,
                        **kwargs):
    arguments = kwargs['arguments']
    criterion = kwargs['criterion']
    conditional_gen = kwargs['conditional_gen']
 
    L1, L2 = generator.L1, generator.L2
    K = generator.K
    def drawgendata():
        I, N = 4, 3

        for i in range(I):
            fg1 = plt.subplot(N, I, i+1)
            plt.imshow(out_g[i].view(int(np.sqrt(L2)), int(np.sqrt(L2))).data.numpy())

            plt.subplot(N, I, I+i+1) 
            plt.imshow(ft[i].view(int(np.sqrt(L1)), int(np.sqrt(L1))).numpy())

            plt.subplot(N, I, 2*I+i+1) 
            plt.imshow(tar[i].view(int(np.sqrt(L2)), int(np.sqrt(L2))).numpy())

    # end of drawnow function

    optimizerD = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))

    if not arguments.cuda:
        figure(figsize=(4,4))
    true, false = 1, 0
    for ep in range(EP):
        for (ft, tar), (mix, _) in zip(train_loader, loader_mix):
            if arguments.cuda:
                tar = tar.cuda()
                ft = ft.cuda()
                mix = mix.cuda()

            # discriminator gradient with real data
            discriminator.zero_grad()
            out_d = discriminator.forward(Variable(tar))
            labels = Variable(torch.ones(tar.size(0))*true).squeeze().float()
            if arguments.cuda:
                labels = labels.cuda()
            err_D_real = criterion(out_d, labels)
            err_D_real.backward()

            # discriminator gradient with generated data
            #if conditional_gen: 
            #    inp = mix.contiguous().view(-1, L1)
            #else:
            #    inp = ft_rshape # fixed_noise.contiguous().view(-1, L)

            out_g = generator.forward(Variable(ft))
            out_d_g = discriminator.forward(out_g)
            labels = Variable(torch.ones(ft.size(0))*false).squeeze().float()
            if arguments.cuda:
                labels = labels.cuda()
            err_D_fake = criterion(out_d_g, labels) 
            err_D_fake.backward(retain_variables=True)

            err_D = err_D_real + err_D_fake
            optimizerD.step()

            # show the current generated output
            if not arguments.cuda:
                drawnow(drawgendata)

            # generator gradient
            generator.zero_grad()
            out_d_g = discriminator.forward(out_g)
            labels = Variable(torch.ones(ft.size(0))*true).squeeze().float()
            if arguments.cuda:
                labels = labels.cuda()
            err_G = criterion(out_d_g, labels)
            err_G.backward()

            optimizerG.step()

            print(out_d.mean())
            print(out_d_g.mean())
            print(err_G.mean())
            print(ep)

def generative_trainer(loader_mix, train_loader, 
                        generator, EP = 5,
                        **kwargs):
    arguments = kwargs['arguments']
    criterion = kwargs['criterion']
    conditional_gen = kwargs['conditional_gen']

    L1 = generator.L1
    L2 = generator.L2
    K = generator.K

    def drawgendata():
        I, N = 4, 3

        for i in range(I):
            fg1 = plt.subplot(N, I, i+1)
            plt.imshow(out_g[i].view(int(np.sqrt(L2)), int(np.sqrt(L2))).data.numpy())

            plt.subplot(N, I, I+i+1) 
            plt.imshow(ft[i].view(int(np.sqrt(L1)), int(np.sqrt(L1))).numpy())

            plt.subplot(N, I, 2*I+i+1) 
            plt.imshow(tar[i].view(int(np.sqrt(L2)), int(np.sqrt(L2))).numpy())

    optimizerG = optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))

    if not arguments.cuda:
        figure(figsize=(4,4))
    true, false = 1, 0
    for ep in range(EP):
        for (ft, tar), (mix, _) in zip(train_loader, loader_mix):
            if arguments.cuda:
                tar = tar.cuda()
                ft = ft.cuda()
                mix = mix.cuda()

            #ft_rshape = ft.view(-1, L)
            tar_rshape = tar.view(-1, L2) 

            #if conditional_gen: 
            #    inp = mix.contiguous().view(-1, L)
            #else:
            #    inp = ft_rshape   # fixed_noise.contiguous().view(-1, L)

            # generator gradient
            generator.zero_grad()
            out_g = generator.forward(Variable(ft))
            err_G = criterion(out_g, Variable(tar_rshape))
            err_G.backward()

            # show the current generated output
            if not arguments.cuda:
                drawnow(drawgendata)

            optimizerG.step()

            print(err_G)
            print(ep)


def form_mixtures(digit1, digit2, loader, arguments): 
    dataset1, dataset2 = [], []
    for i, (ft, tar) in enumerate(loader):   
        # digit 1
        mask = torch.eq(tar, digit1)
        inds = torch.nonzero(mask).squeeze()
        ft1 = torch.index_select(ft, dim=0, index=inds)
        dataset1.append(ft1)

        # digit 2
        mask = torch.eq(tar, digit2)
        inds = torch.nonzero(mask).squeeze()
        ft2 = torch.index_select(ft, dim=0, index=inds)
        dataset2.append(ft2)
        print(i)
        
    dataset1 = torch.cat(dataset1, dim=0)
    dataset2 = torch.cat(dataset2, dim=0)

    if arguments.input_type == 'noise':
        inp1 = torch.randn(dataset1.size(0), arguments.L1) 
        inp2 = torch.randn(dataset2.size(0), arguments.L1) 
    elif arguments.input_type == 'autoenc':
        inp1 = dataset1
        inp2 = dataset2
    else:
        raise ValueError('Whaaaaaat input_type?')

    N1, N2 = dataset1.size(0), dataset2.size(0)
    Nmix = min([N1, N2])

    dataset_mix = dataset1[:Nmix] + dataset2[:Nmix]
        
    dataset1 = data_utils.TensorDataset(data_tensor=inp1,
                                        target_tensor=dataset1)
    dataset2 = data_utils.TensorDataset(data_tensor=inp2,
                                        target_tensor=dataset2)
    dataset_mix = data_utils.TensorDataset(data_tensor=dataset_mix,
                                        target_tensor=torch.ones(Nmix))

    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader1 = data_utils.DataLoader(dataset1, batch_size=arguments.batch_size, shuffle=False, **kwargs)
    loader2 = data_utils.DataLoader(dataset2, batch_size=arguments.batch_size, shuffle=False, **kwargs)
    loader_mix = data_utils.DataLoader(dataset_mix, batch_size=arguments.batch_size, shuffle=False, **kwargs)

    return loader1, loader2, loader_mix

def get_loaders(data, batch_size, **kwargs):
    arguments=kwargs['arguments']

    if data == 'mnist':
        kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               #transforms.Normalize((0,), (1,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               #transforms.Normalize((7,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

def maxlikelihood_separatesources(generators, loader_mix, EP, **kwargs):
    generator1, generator2 = generators
    arguments = kwargs['arguments']
    loss = kwargs['loss']
    L1 = generator1.L1
    L2 = generator1.L2

    x1hat, x2hat = [], []
    mixes = []
    for i, (mix, _ ) in enumerate(islice(loader_mix, 0, 1, 1)): 
        if arguments.cuda:
            mix = mix.cuda()

        print('Processing source ',i)
        Nmix = mix.size(0)

        if arguments.cuda:
            x1 = Variable(torch.rand(Nmix, L1).cuda(), requires_grad=True)
            x2 = Variable(torch.rand(Nmix, L1).cuda(), requires_grad=True)
        else:
            x1 = Variable(torch.rand(Nmix, L1), requires_grad=True)
            x2 = Variable(torch.rand(Nmix, L1), requires_grad=True)

        optimizer_sourcesep = optim.Adam([x1, x2], lr=1e-3, betas=(0.5, 0.999))
        for ep in range(EP):
           
            mix_sum = generator1.forward(x1) + generator2.forward(x2) 
            if loss == 'Euclidean': 
                err = torch.mean((Variable(mix) - mix_sum)**2)
            elif loss == 'Poisson':
                eps = 1e-20
                err = torch.mean(-Variable(mix)*torch.log(mix_sum+eps) + mix_sum)

            err.backward()

            optimizer_sourcesep.step()

            x1.grad.data.zero_()
            x2.grad.data.zero_()

            print('Step in batch [{:d}\{:d}]'.format(ep+1, EP))
            print('The error is ', err)
        x1hat.append(generator1.forward(x1).data.cpu().numpy())
        x2hat.append(generator2.forward(x2).data.cpu().numpy())
        mixes.append(mix.cpu().numpy())

    print('sum is:', x1hat[0].sum())
    num_ims, c = 10, 2
    figure(num=None, figsize=(3*c, num_ims*c), dpi=80, facecolor='w', edgecolor='k')
    sqrtL2 = int(np.sqrt(L2))
    for i in range(num_ims):
        plt.subplot(num_ims, 3, 3*i + 1)
        plt.imshow(x1hat[0][i].reshape(sqrtL2, sqrtL2))
        plt.title('Estimated Source 1')

        plt.subplot(num_ims, 3, 3*i + 2)
        plt.imshow(x2hat[0][i].reshape(sqrtL2, sqrtL2))
        plt.title('Estimated Source 2')

        plt.subplot(num_ims, 3, 3*i + 3)
        plt.imshow(mixes[0][i].reshape(sqrtL2, sqrtL2))
        plt.title('Mixture')

    curdir = os.getcwd()
    figdir = os.path.join(curdir, 'figures')
    if not os.path.exists(figdir):       
        os.mkdir(figdir)
    figname = '_'.join([kwargs['data'], kwargs['tr_method'], 
                        'conditional',
                        str(kwargs['conditional']), 
                        'smooth_output',
                        str(kwargs['arguments'].smooth_output),
                        'sourceseparation'])
    plt.savefig(os.path.join(figdir, figname))

