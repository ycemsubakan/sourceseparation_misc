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
    def __init__(self, ngpu):
        super(netG, self).__init__()
        self.ngpu = ngpu
        pl = 40
        self.l1 = nn.Linear(L,K+pl, bias=True)
        self.l2 = nn.Linear(K+pl,L, bias=True) 

    def forward(self, inp):
        h = F.tanh(self.l1(inp))
        output = F.softplus(self.l2(h))
        return output


class netD(nn.Module):
    def __init__(self, ngpu):
        super(netD, self).__init__()
        self.ngpu = ngpu
        self.l1 = nn.Linear(L,K)
        self.l2 = nn.Linear(K,1) 

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


def adverserial_trainer(loader_mix, train_loader, generator, discriminator, EP = 5):

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

    figure(figsize=(4,4))
    true, false = 1, 0
    for ep in range(EP):
        for (ft, _), (mix, _) in zip(train_loader, loader_mix):

            ft_rshape = ft.view(-1, L)
            # discriminator gradient with real data
            discriminator.zero_grad()
            out_d = discriminator.forward(Variable(ft_rshape))
            labels = Variable(torch.ones(ft.size(0))*true).squeeze().float()
            err_D_real = criterion(out_d, labels)
            err_D_real.backward()

            # discriminator gradient with generated data
            mix_rshape = mix.view(-1, L)
            out_g = generator.forward(Variable(mix_rshape[:ft.size(0)]))
            out_d_g = discriminator.forward(out_g)
            labels = Variable(torch.ones(ft.size(0))*false).squeeze().float()
            err_D_fake = criterion(out_d_g, labels) 
            err_D_fake.backward(retain_variables=True)

            err_D = err_D_real + err_D_fake
            optimizerD.step()

            # show the current generated output
            drawnow(drawgendata)

            # generator gradient
            generator.zero_grad()
            out_d_g = discriminator.forward(out_g)
            labels = Variable(torch.ones(ft.size(0))*true).squeeze().float()
            err_G = criterion(out_d_g, labels)
            err_G.backward()

            optimizerG.step()

            print(out_d.mean())
            print(out_d_g.mean())
            print(err_G.mean())
            print(ep)


def form_mixtures(digit1, digit2, loader, args): 
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

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    loader1 = data_utils.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, **kwargs)
    loader2 = data_utils.DataLoader(dataset2, batch_size=args.batch_size, shuffle=True, **kwargs)
    loader_mix = data_utils.DataLoader(dataset_mix, batch_size=args.batch_size, shuffle=True, **kwargs)

    return loader1, loader2, loader_mix
        

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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

loader_bs = 1000
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0,), (1,))
                   ])),
    batch_size=loader_bs, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((7,), (0.3081,))
                   ])),
    batch_size=loader_bs, shuffle=True, **kwargs)

loader1, loader2, loader_mix = form_mixtures(0, 1, train_loader, args)


K = 100
ngpu = 1
L = 28*28


generator1 = netG(ngpu)
discriminator1 = netD(ngpu)

criterion = nn.BCELoss()
fixed_noise = torch.FloatTensor(args.batch_size, L).normal_(0, 1)

EP = 5
adverserial_trainer(loader_mix=loader_mix,
                    train_loader=loader1,
                    generator=generator1, 
                    discriminator=discriminator1, 
                    EP=EP)

###
EP = 500
x1hat, x2hat = [], []
mixes = []
for i, (mix, _ ) in enumerate(islice(loader_mix,0,1,1)): 
    mix = mix[:20]

    print('Processing source ',i)
    Nmix = mix.size(0)
    mix_rshape = mix.view(Nmix, L)

    x1 = Variable(torch.rand(Nmix, L), requires_grad=True)

    optimizer_sourcesep = optim.Adam([x1], lr=1e-3, betas=(0.5, 0.999))
    for ep in range(EP):
       
        mix_sum = generator1.forward(x1) 
        err = torch.mean((Variable(mix_rshape) - mix_sum)**2)

        err.backward()

        optimizer_sourcesep.step()

        x1.grad.data.zero_()

        print('Step in batch [{:d}\{:d}]'.format(ep+1, EP))
        print('The error is ', err)
    x1hat.append(generator1.forward(Variable(mix_rshape)).data.numpy())
    mixes.append(mix.numpy())

num_ims = 6
sqrtL = int(np.sqrt(L))
for i in range(num_ims):
    plt.subplot(num_ims, 2, 2*i + 1)
    plt.imshow(x1hat[0][i].reshape(sqrtL, sqrtL))
    plt.title('Estimated Source ')


    plt.subplot(num_ims, 2, 2*i + 2)
    plt.imshow(mixes[0][i].reshape(sqrtL, sqrtL))
    plt.title('Mixture')


#plt.savefig('mnist_gan_sourceseperation.png')

####
