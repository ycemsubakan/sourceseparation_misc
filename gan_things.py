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


class netG(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(netG, self).__init__()
        self.ngpu = ngpu
        pl = 40
        self.L = kwargs['L']
        self.K = kwargs['K']
        self.l1 = nn.Linear(self.L, self.K+pl, bias=True)
        self.l2 = nn.Linear(self.K+pl, self.L, bias=True) 

    def forward(self, inp):
        h = F.tanh(self.l1(inp))
        output = F.softplus(self.l2(h))
        return output

class netD(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(netD, self).__init__()
        self.ngpu = ngpu
        self.L = kwargs['L']
        self.K = kwargs['K']
        self.l1 = nn.Linear(self.L, self.K, bias=True)
        self.l2 = nn.Linear(self.K,1, bias=True) 

    def forward(self, inp):
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
    fixed_noise = kwargs['fixed_noise' ]
    conditional_gen = kwargs['conditional_gen']

    L = generator.L
    K = generator.K
    def drawgendata():
        I = 4
        plt.subplot(2,2,1)
        plt.imshow(out_g[0].view(int(np.sqrt(L)), int(np.sqrt(L))).data.numpy())

        plt.subplot(2,2,2)
        plt.imshow(out_g[1].view(int(np.sqrt(L)), int(np.sqrt(L))).data.numpy())

        plt.subplot(2,2,3)
        plt.imshow(out_g[2].view(int(np.sqrt(L)), int(np.sqrt(L))).data.numpy())

        plt.subplot(2,2,4)
        plt.imshow(out_g[3].view(int(np.sqrt(L)), int(np.sqrt(L))).data.numpy())

    optimizerD = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))

    if not arguments.cuda:
        figure(figsize=(4,4))
    true, false = 1, 0
    for ep in range(EP):
        for (ft, _), (mix, _) in zip(train_loader, loader_mix):
            if arguments.cuda:
                ft = ft.cuda()
                mix = mix.cuda()

            ft_rshape = ft.view(-1, L)
            # discriminator gradient with real data
            discriminator.zero_grad()
            out_d = discriminator.forward(Variable(ft_rshape))
            labels = Variable(torch.ones(ft.size(0))*true).squeeze().float()
            if arguments.cuda:
                labels = labels.cuda()
            err_D_real = criterion(out_d, labels)
            err_D_real.backward()

            # discriminator gradient with generated data
            if conditional_gen: 
                inp = mix.contiguous().view(-1, L)
            else:
                inp = fixed_noise.contiguous().view(-1, L)

            out_g = generator.forward(Variable(inp[:ft.size(0)]))
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
    fixed_noise = kwargs['fixed_noise' ]
    conditional_gen = kwargs['conditional_gen']

    L = generator.L
    K = generator.K
    def drawgendata():
        I = 4
        plt.subplot(2,2,1)
        plt.imshow(out_g[0].view(int(np.sqrt(L)), int(np.sqrt(L))).data.numpy())

        plt.subplot(2,2,2)
        plt.imshow(out_g[1].view(int(np.sqrt(L)), int(np.sqrt(L))).data.numpy())

        plt.subplot(2,2,3)
        plt.imshow(out_g[2].view(int(np.sqrt(L)), int(np.sqrt(L))).data.numpy())

        plt.subplot(2,2,4)
        plt.imshow(out_g[3].view(int(np.sqrt(L)), int(np.sqrt(L))).data.numpy())

    optimizerG = optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))

    if not arguments.cuda:
        figure(figsize=(4,4))
    true, false = 1, 0
    for ep in range(EP):
        for (ft, _), (mix, _) in zip(train_loader, loader_mix):
            if arguments.cuda:
                ft = ft.cuda()
                mix = mix.cuda()

            ft_rshape = ft.view(-1, L)

            if conditional_gen: 
                inp = mix.contiguous().view(-1, L)
            else:
                inp = fixed_noise.contiguous().view(-1, L)

            # generator gradient
            generator.zero_grad()
            out_g = generator.forward(Variable(inp[:ft.size(0)]))
            err_G = criterion(out_g, Variable(ft_rshape))
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
        try:
            ft1 = torch.index_select(ft, dim=0, index=inds)
        except:
            pdb.set_trace()
        dataset1.append(ft1)

        # digit 2
        mask = torch.eq(tar, digit2)
        inds = torch.nonzero(mask).squeeze()
        ft2 = torch.index_select(ft, dim=0, index=inds)
        dataset2.append(ft2)
        print(i)

    dataset1 = torch.cat(dataset1, dim=0)
    dataset2 = torch.cat(dataset2, dim=0)

    N1, N2 = dataset1.size(0), dataset2.size(0)
    Nmix = min([N1, N2])

    dataset_mix = dataset1[:Nmix] + dataset2[:Nmix]
        
    dataset1 = data_utils.TensorDataset(data_tensor=dataset1,
                                        target_tensor=torch.ones(N1)*digit1)
    dataset2 = data_utils.TensorDataset(data_tensor=dataset2,
                                        target_tensor=torch.ones(N2)*digit2)
    dataset_mix = data_utils.TensorDataset(data_tensor=dataset_mix,
                                        target_tensor=torch.ones(Nmix))

    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader1 = data_utils.DataLoader(dataset1, batch_size=arguments.batch_size, shuffle=True, **kwargs)
    loader2 = data_utils.DataLoader(dataset2, batch_size=arguments.batch_size, shuffle=True, **kwargs)
    loader_mix = data_utils.DataLoader(dataset_mix, batch_size=arguments.batch_size, shuffle=True, **kwargs)

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
    L = generator1.L

    x1hat, x2hat = [], []
    mixes = []
    for i, (mix, _ ) in enumerate(islice(loader_mix, 0, 1, 1)): 
        if arguments.cuda:
            mix = mix.cuda()

        print('Processing source ',i)
        Nmix = mix.size(0)

        if arguments.cuda:
            x1 = Variable(torch.rand(Nmix, L).cuda(), requires_grad=True)
            x2 = Variable(torch.rand(Nmix, L).cuda(), requires_grad=True)
        else:
            x1 = Variable(torch.rand(Nmix, L), requires_grad=True)
            x2 = Variable(torch.rand(Nmix, L), requires_grad=True)

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


        num_ims, c = 10, 2
        figure(num=None, figsize=(3*c, num_ims*c), dpi=80, facecolor='w', edgecolor='k')
        sqrtL = int(np.sqrt(L))
        for i in range(num_ims):
            plt.subplot(num_ims, 3, 3*i + 1)
            plt.imshow(x1hat[0][i].reshape(sqrtL, sqrtL))
            plt.title('Estimated Source 1')

            plt.subplot(num_ims, 3, 3*i + 2)
            plt.imshow(x2hat[0][i].reshape(sqrtL, sqrtL))
            plt.title('Estimated Source 2')

            plt.subplot(num_ims, 3, 3*i + 3)
            plt.imshow(mixes[0][i].reshape(sqrtL, sqrtL))
            plt.title('Mixture')

    figname = '_'.join([kwargs['data'], kwargs['tr_method'], 
                        'conditional',
                        str(kwargs['conditional']), 
                        'sourceseparation'])
    plt.savefig(figname)

