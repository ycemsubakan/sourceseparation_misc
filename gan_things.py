import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from drawnow import drawnow, figure
import torch.utils.data as data_utils
import torch.nn.utils.rnn as rnn_utils
from itertools import islice
import os
import torch.nn.init as torchinit
import librosa as lr
import librosa.display as lrd
import utils as ut
import itertools as it
import copy

def initializationhelper(param, nltype):
    
    c = 1
    #torchinit.uniform(param.weight, a=-c, b=c)
    #torchinit.eye(param.weight)

    #torchinit.xavier_uniform(param.weight, gain=c*torchinit.calculate_gain(nltype))
    #torchinit.uniform(param.bias, a=-c, b=c)
    #torchinit.eye(param.bias)

class Normalizing_Flow(nn.Module):
    def __init__(self, ngpu, c=0.01, out='relu', **kwargs):
        super(Normalizing_Flow, self).__init__()

        self.L2 = kwargs['L2']
        self.L1 = kwargs['L1']
        self.nlayers = kwargs['nlayers']
        self.arguments = kwargs['arguments']
        self.out = out

        ws = [nn.Parameter(c*torch.randn(self.L2)) if nl > 0 else \
              nn.Parameter(c*torch.randn(self.L1)) for nl in range(self.nlayers)]
        self.ws = nn.ParameterList(ws) 

        us = [nn.Parameter(c*torch.randn(self.L2)) for nl in range(self.nlayers)]
        self.us = nn.ParameterList(us)

        bs = [nn.Parameter(c*torch.randn(1)) for nl in range(self.nlayers)]
        self.bs = nn.ParameterList(bs)

    def forward_layer1(self, h):
                
        for nl in range(self.nlayers):
           mat = Variable(torch.eye(h.size(1))) + (self.us[nl].unsqueeze(1)*self.ws[nl].unsqueeze(0)) 
           h = torch.matmul(mat, h.permute(1,0)) + self.us[nl].unsqueeze(1)*self.bs[nl]
           h = h.permute(1,0) 
        return h 

    def forward_layer2(self, h):
        
        for nl in range(self.nlayers):
            temp = (torch.matmul(self.ws[nl].unsqueeze(0), h.permute(1,0))) 
            h = h + self.us[nl].unsqueeze(0)*temp.permute(1,0) + self.us[nl].unsqueeze(0)*self.bs[nl]
        return h


    def forward(self, h):
        
        #out1 = self.forward_layer1(h)
        out2 = self.forward_layer2(h)
        if self.out == 'relu':
            return F.relu(out2)
        else:
            return out2
       
    def inv_forward(self, x):
        eps = 1e-10
        if self.out == 'relu':
            mask = torch.gt(x, 0).float()
            h = mask*x + (1-mask)*(-100) #(100*x)
            #h = torch.log(x.exp() - 1 + eps)
        else:
            h = x

        det = 0
        for nl in range(self.nlayers-1, -1, -1):
            h_mbias = h - self.bs[nl]*self.us[nl].unsqueeze(0)
            temp = h_mbias.mm(self.ws[nl].unsqueeze(1)) 
            inner_prod = torch.sum(self.us[nl]*self.ws[nl])
            #print('layer {} value = {}'.format(nl, inner_prod.data[0]))
            temp = (self.us[nl].unsqueeze(0)*temp) / (1 + inner_prod) 
            h = h_mbias - temp
            
            det = det + torch.log(torch.abs(1 + inner_prod))
      
        mismatch = (x.data  - self.forward(h).data).abs().sum()
        print(mismatch)
        h_prev = h

        #h = x
        #det = 0 
        #for nl in range(self.nlayers-1, -1, -1):
        #    h_mbias = (h - self.bs[nl]*self.us[nl].unsqueeze(0)).permute(1,0)
        #    inner_prod = torch.sum(self.us[nl]*self.ws[nl])

        #    mat = Variable(torch.eye(h.size(1))) + (self.us[nl].unsqueeze(1)*self.ws[nl].unsqueeze(0)) 
        #    inv_mat = Variable(torch.eye(h.size(1))) - (self.us[nl].unsqueeze(1)*self.ws[nl].unsqueeze(0))/(1+inner_prod)

        #    h = torch.matmul(inv_mat, h_mbias).permute(1,0)
        #    det = det + torch.log(torch.abs(1 + inner_prod))
        #h_prev2 = h

        #mismatch = (x.data  - self.forward(h_prev2).data).abs().sum()
        #print(mismatch)


        return h, det, mismatch

    def compute_density_normal(self, x, mu=0, sigma=1):
        h, detterm, mismatch = self.inv_forward(x)


        log_density = -detterm  - ((h - mu)**2/(2*sigma**2)).sum(1) 

        return log_density, torch.exp(log_density), mismatch
    
class VAE_nf(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(VAE_nf, self).__init__()

        self.L1 = kwargs['L1']
        self.L2 = kwargs['L2']
        self.K = kwargs['K']
        self.arguments = kwargs['arguments']

        self.fc1 = nn.Linear(self.L1, self.K)
        initializationhelper(self.fc1, 'relu')

        self.fc21 = nn.Linear(self.K, self.arguments.Kdisc)
        initializationhelper(self.fc21, 'relu')

        self.fc22 = nn.Linear(self.K, self.arguments.Kdisc)
        initializationhelper(self.fc22, 'relu')

        self.fc3 = Normalizing_Flow(1, L1=self.arguments.Kdisc,
                                    L2=self.arguments.L2, 
                                    nlayers=self.arguments.flow_layers,
                                    arguments=self.arguments)
        #initializationhelper(self.fc3, 'relu')

        #self.fc4 = nn.Linear(self.arguments.K, self.L2)
        #initializationhelper(self.fc4, 'relu')



    def encode(self, x):

        h1 = F.tanh(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return mu + eps.mul(std).add_(mu) # Variable(torch.randn(mu.size()))  #
        else:
          return mu

    def decode(self, z):
        z1 = (self.fc3(z))
        return z1

    def forward(self, inp):
        if not (type(inp) == Variable):
            inp = Variable(inp[0])

        mu, logvar = self.encode(inp)
        h = self.reparameterize(mu, logvar)

        #print('mean of mu {} variance of mu {}'.format(torch.mean(h).data[0], torch.var(h).data[0]))
        if self.arguments.out_type == 'noise':
            return self.decode(h), mu, logvar, h
        elif self.arguments.out_type == 'implicit':
            hhat, logdet, _ = self.fc3.inv_forward(inp)
            return hhat, mu, logvar, h, logdet, self.decode(h)


class VAE(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(VAE, self).__init__()

        self.L1 = kwargs['L1']
        self.L2 = kwargs['L2']
        self.K = kwargs['K']
        self.arguments = kwargs['arguments']

        self.fc1 = nn.Linear(self.L1, self.K)
        #initializationhelper(self.fc1, 'relu')

        self.fc21 = nn.Linear(self.K, self.arguments.Kdisc)
        #initializationhelper(self.fc21, 'relu')

        self.fc22 = nn.Linear(self.K, self.arguments.Kdisc)
        #initializationhelper(self.fc22, 'relu')

        self.fc3 = nn.Linear(self.arguments.Kdisc, self.arguments.K)
        #initializationhelper(self.fc3, 'relu')

        self.fc4 = nn.Linear(self.arguments.K, self.L2)
        #initializationhelper(self.fc4, 'relu')



    def encode(self, x):

        h1 = F.tanh(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        z1 = F.tanh(self.fc3(z))
        return (self.fc4(z1))

    def forward(self, inp):
        
        if not (type(inp) == Variable):
            inp = Variable(inp[0])

        mu, logvar = self.encode(inp)
        h = self.reparameterize(mu, logvar)

        print('mean of mu {} variance of mu {}'.format(torch.mean(h).data[0], torch.var(h).data[0]))
        return self.decode(h), mu, logvar, h

class netG(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(netG, self).__init__()
        self.ngpu = ngpu
        pl = 0
        self.L1 = kwargs['L1']
        self.L2 = kwargs['L2']
        self.K = kwargs['K']
        self.arguments = kwargs['arguments']
        self.l1 = nn.Linear(self.L1, self.K+pl, bias=True)
        initializationhelper(self.l1, 'relu')

        self.l2 = nn.Linear(self.K+pl, self.L2, bias=True) 
        initializationhelper(self.l2, 'relu')

        self.smooth_output = self.arguments.smooth_output
        if self.smooth_output:
            self.sml = nn.Conv2d(1, 1, 5, padding=2) 
            initializationhelper(self.sml, 'relu')

    def forward(self, inp):
        #if inp.dim() > 2:
        #    inp = inp.permute(0, 2, 1)
        #inp = inp.contiguous().view(-1, self.L1)
        #if not (type(inp) == Variable):
        #    inp = Variable(inp[0])

        if self.arguments.tr_method in ['adversarial', 'adversarial_wasserstein']:
            h = F.softplus((self.l1(inp)))
        elif self.arguments.tr_method == 'ML':
            h = F.softplus((self.l1(inp)))
        else:
            raise ValueError('Whaat method?')
        output = F.softplus(self.l2(h))

        if self.smooth_output:
            output = output.view(-1, 1, int(np.sqrt(self.L2)), int(np.sqrt(self.L2)))
            output = F.softplus(self.sml(output))
            output = output.view(-1, self.L2)
        return output

class netG_onelayer(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(netG_onelayer, self).__init__()
        self.ngpu = ngpu
        pl = 0
        self.L1 = kwargs['L1']
        self.L2 = kwargs['L2']
        self.arguments = kwargs['arguments']
        self.l1 = nn.Linear(self.L1, self.L2, bias=True)
        initializationhelper(self.l1, 'relu')


    def forward(self, inp):
        if not (type(inp) == Variable):
            inp = Variable(inp[0])

        output = F.softplus((self.l1(inp)))
     
        return output

class netG_onelayer_sp(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(netG_onelayer_sp, self).__init__()
        self.ngpu = ngpu
        pl = 0
        self.L1 = kwargs['L1']
        self.L2 = kwargs['L2']
        self.arguments = kwargs['arguments']

        c = 0.1
        self.l1 = torch.nn.Parameter((c*torch.randn(self.L1, self.L2)))
        #self.b1 = torch.nn.Parameter((c*torch.randn(1,self.L2)))

    def forward(self, inp):
        if not (type(inp) == Variable):
            inp = Variable(inp[0])

        output = torch.mm(inp, F.softplus(self.l1))
        #output = output + self.b1
        return output


class netD(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(netD, self).__init__()
        self.ngpu = ngpu
        self.L = kwargs['L']
        self.K = kwargs['K'] 
        self.arguments = kwargs['arguments']
        if hasattr(self.arguments, 'pack_num'):
            self.L = self.arguments.pack_num*self.L

        self.l1 = nn.Linear(self.L, self.K, bias=True)
        #initializationhelper(self.l1, 'tanh')
        #self.l1_bn = nn.BatchNorm1d(self.K)

        self.l3 = nn.Linear(self.K, 1, bias=True)
        #initializationhelper(self.l3, 'relu') 

    def forward(self, inp):
        #if inp.dim() > 2:
        #    inp = inp.permute(0, 2, 1)
        #inp = inp.contiguous().view(-1, self.L) 

        #if not (type(inp) == Variable):
        #    inp = Variable(inp[0])

        #if hasattr(self.arguments, 'pack_num'):
        #    N = inp.size(0)
        #    Ncut = int(N/self.arguments.pack_num)
        #    split = torch.split(inp, Ncut, dim=0)
        #    inp = torch.cat(split, dim=1)

        h1 = F.tanh((self.l1(inp)))
        
        if self.arguments.tr_method == 'adversarial_wasserstein':
            output = (self.l3(h1))
        else:
            output = F.sigmoid(self.l3(h1))

        return output, h1

class netD_images(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(netD, self).__init__()
        self.ngpu = ngpu
        self.L = kwargs['L']
        self.K = kwargs['K'] 
        self.arguments = kwargs['arguments']
        if hasattr(self.arguments, 'pack_num'):
            self.L = self.arguments.pack_num*self.L

        self.l1 = nn.Linear(self.L, 1, bias=True)
        
    def forward(self, inp):
        
        if not (type(inp) == Variable):
            inp = Variable(inp[0])

        if self.arguments.tr_method == 'adversarial_wasserstein':
            output = (self.l1(inp))
        else:
            output = F.sigmoid(self.l1(inp))

        return output, None



class netG_images(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(netG_images, self).__init__()
        self.ngpu = ngpu
        pl = 0
        self.L1 = kwargs['L1']
        self.L2 = kwargs['L2']
        self.K = kwargs['K']
        self.arguments = kwargs['arguments']
        self.l1 = nn.Linear(self.L1, self.K+pl, bias=True)
        initializationhelper(self.l1, 'tanh')

        self.l2 = nn.Linear(self.K+pl, self.L2, bias=True) 
        initializationhelper(self.l2, 'relu')

        self.smooth_output = self.arguments.smooth_output
        if self.smooth_output:
            self.sml = nn.Conv2d(1, 1, 5, padding=2) 
            initializationhelper(self.sml, 'relu')

    def forward(self, inp):
        #if inp.dim() > 2:
        #    inp = inp.permute(0, 2, 1)
        #inp = inp.contiguous().view(-1, self.L1)
        if not (type(inp) == Variable):
            inp = Variable(inp[0])

        h = F.tanh(self.l1(inp))
        output = (self.l2(h))

        if self.smooth_output:
            output = output.view(-1, 1, int(np.sqrt(self.L2)), int(np.sqrt(self.L2)))
            output = F.softplus(self.sml(output))
            output = output.view(-1, self.L2)
        return output


class netD_images(nn.Module):
    def __init__(self, ngpu, **kwargs):
        super(netD_images, self).__init__()
        self.ngpu = ngpu
        self.L = kwargs['L']
        self.K = kwargs['K']
        self.arguments = kwargs['arguments']

        self.l1 = nn.Linear(self.L, self.K, bias=True)
        initializationhelper(self.l1, 'tanh')
        self.l1_bn = nn.BatchNorm1d(self.K)

        self.l2 = nn.Linear(self.K, self.K, bias=True) 
        initializationhelper(self.l2, 'relu')
        #self.l2_bn = nn.BatchNorm1d(self.K)

        self.l3 = nn.Linear(self.K, 1, bias=True)
        initializationhelper(self.l3, 'relu') 

    def forward(self, inp):
        #if inp.dim() > 2:
        #    inp = inp.permute(0, 2, 1)
        #inp = inp.contiguous().view(-1, self.L) 

        if not (type(inp) == Variable):
            inp = Variable(inp[0])

        h1 = F.tanh((self.l1(inp)))
        
        h2 = F.tanh((self.l2(h1)))

        output = (self.l3(h2))
        return output, h1

def initialize_v(M, J, model, arguments):
    vs = []
    for m in range(M):
        v = []
        for j in range(J):
            temp = []
            for p in model[m][j].parameters():    
                if arguments.cuda:
                    temp.append(Variable(torch.randn(p.size()).cuda()))
                else:
                    temp.append(Variable(torch.randn(p.size())))
            v.append(temp)
        vs.append(v)
    return vs

def initialize_v_gibbs(M, model, arguments):
    vs = []
    for m in range(M):
        temp = []
        for p in model[m].parameters():    
            if arguments.cuda:
                temp.append(Variable(torch.randn(p.size()).cuda()))
            else:
                temp.append(Variable(torch.randn(p.size())))
        vs.append(temp)
    return vs


def gibbs_gan_trainer(loader_mix, train_loader,
                      generator, discriminator, EP=5,
                      **kwargs):
    arguments = kwargs['arguments']
    criterion = kwargs['criterion']
    conditional_gen = kwargs['conditional_gen']
    source_num = kwargs['source_num']

    M = 1 
    al = 0.9
    eta = 0.001
    nf = 0

    # initialize generators and discriminators
    discs = []
    gens = [] 

    for m in range(M):
        if arguments.cuda:
            discs.append(netD(1, K=arguments.K, L1=arguments.L1, L=arguments.L2, arguments=arguments).cuda())
            gens.append(netG_images(1, K=arguments.K, L1=arguments.L1, L2=arguments.L2, arguments=arguments).cuda()) 
        else:
            discs.append(netD(1, K=arguments.K, L1=arguments.L1, L=arguments.L2, arguments=arguments))
            gens.append(netG_images(1, K=arguments.K, L1=arguments.L1, L2=arguments.L2, arguments=arguments))

    # allocate v's
    vgens = initialize_v_gibbs(M, gens, arguments)
    vdiscs = initialize_v_gibbs(M, discs, arguments) 
   
    gen_EP = 1
    disc_EP = 10

    true, false = 1, 0
    for ep in range(EP):
        for (ft, tar, lens), mix in zip(train_loader, loader_mix):
            if arguments.cuda:
                ft = ft.cuda()
                tar = tar.cuda()
                lens = lens.cuda()
                #mix = mix.cuda()

            # sort the tensors within batch
            if arguments.task == 'images' or arguments.task == 'toy_data':
                tar = tar.contiguous().view(-1, arguments.L2)
                tar, ft = Variable(tar), Variable(ft)
            else:
                ft, tar = ut.sort_pack_tensors(ft, tar, lens)

            for m in range(M):
               gens[m].zero_grad() 

               labels = Variable(torch.ones(tar.size(0)))
               if arguments.cuda:
                   labels = labels.cuda()
               for e in range(gen_EP):
                    gens[m].zero_grad()
                    out_g = gens[m].forward(ft)
                    out_d, _ = discs[m].forward(out_g)  

                    errd_fake = criterion(out_d.squeeze(), labels)
                    errd_fake.backward()

                    for k, p in enumerate(gens[m].parameters()): 
                         vgens[m][k] = vgens[m][k]*(1-al) - eta*p.grad 
                         vr = float(np.sqrt(2*al*eta))*nf
                         if arguments.cuda:
                             vgens[m][k] = vgens[m][k] + Variable(vr*torch.randn(p.size()).cuda())   
                         else:
                             vgens[m][k] = vgens[m][k] + Variable(vr*torch.randn(p.size()))   
                         
                         # some debugging here 
                         old_data = copy.deepcopy(p.data)

                         p.data.copy_((vgens[m][k]+p).data)

                         assert 1-torch.equal(list(gens[m].parameters())[k].data, old_data), 'not assigning things'

               # update discriminator 
               labels_true = Variable(torch.ones(tar.size(0)))
               labels_false = Variable(torch.zeros(tar.size(0)))
               if arguments.cuda:
                   labels_true = labels_true.cuda()
                   labels_false = labels_false.cuda()

               for e in range(disc_EP):
                    discs[m].zero_grad()
                    out_g = gens[m].forward(ft)
                    out_d, _ = discs[m].forward(out_g)  
                    errd_fake = criterion(out_d.squeeze(), labels_false)

                    out_d_true, _ = discs[m].forward(tar)
                    errd_real = criterion(out_d_true.squeeze(), labels_true)
                   
                    err_d = errd_fake + errd_real
                    err_d.backward()

                    for k, p in enumerate(discs[m].parameters()): 
                         vdiscs[m][k] = vdiscs[m][k]*(1-al) - eta*p.grad 
                         vr = float(np.sqrt(2*al*eta))*nf
                         if arguments.cuda:
                             vdiscs[m][k] = vdiscs[m][k] + Variable(vr*torch.randn(p.size()).cuda())   
                         else:
                             vdiscs[m][k] = vdiscs[m][k] + Variable(vr*torch.randn(p.size()))   
                         
                         # some debugging here 
                         old_data2 = copy.deepcopy(p.data)

                         p.data.copy_((vdiscs[m][k]+p).data)

                         assert 1-torch.equal(list(discs[m].parameters())[k].data, old_data2), 'not assigning things'


                   
        
        if ep % 10 == 0:

            npts = 30
            tol = 30
            xmin = np.min(arguments.means[:,0]) - tol 
            xmax = np.max(arguments.means[:,0]) + tol
            xs = np.linspace(xmin, xmax, npts) 

            ymin = np.min(arguments.means[:,1]) - tol
            ymax = np.max(arguments.means[:,1]) + tol
            ys = np.linspace(ymin, ymax, npts)
            
            X, Y = np.meshgrid(xs, ys) 

            xy_pairs = torch.from_numpy(np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)).float()
            if arguments.cuda:
                xy_pairs = xy_pairs.cuda()
            sigmoids, _ = discs[0].forward(Variable(xy_pairs))
            sigmoids = sigmoids.data.cpu().numpy().reshape(npts, npts)

            cs = plt.contour(X, Y, sigmoids) 
            plt.clabel(cs, inline=1, fontsize=9) 

            tardat = tar.data.cpu().numpy()
            plt.plot(tardat[:, 0], tardat[:, 1], 'o')

            all_samples = out_g.data.cpu().numpy()
            all_sigs = out_d.data.cpu().numpy().squeeze()
            plt.scatter(all_samples[:, 0], all_samples[:, 1], c=all_sigs)
            
            plt.legend()
            plt.rc('legend',**{'fontsize':6})
            plt.title('Situation at iteration {}'.format(ep))

            folder_name = 'toy_example_figures_' + arguments.exp_info
            if not os.path.exists(folder_name):
                os.mkdir(folder_name) 

            if 1:
                plt.savefig(os.path.join(folder_name, 'iter_{}_experiment_{}'.format(ep, arguments.exp_info))) 
                plt.close()
            
            print('EP [{}/{}] Errg {}, out_d {}'.format(ep, EP, 
                                                        errd_fake.data[0], 
                                                        out_d.mean().data[0]))


def hmc_gan_trainer(loader_mix, train_loader,
                    generator, discriminator, EP=5,
                    **kwargs):
    arguments = kwargs['arguments']
    criterion = kwargs['criterion']
    conditional_gen = kwargs['conditional_gen']
    source_num = kwargs['source_num']

    Jg = 5
    Jd = 1
    M = 2 

    # initialize generators and discriminators
    discs = []
    gens = [] 

    if arguments.cuda:
        for m in range(M): 
            discs.append([netD(1, K=arguments.K, L1=arguments.L1, L=arguments.L2, arguments=arguments).cuda() for j in range(Jd)])
            gens.append([netG_images(1, K=arguments.K, L1=arguments.L1, L2=arguments.L2, arguments=arguments).cuda() for j in range(Jg)]) 
    else:
        for m in range(M): 
            discs.append([netD(1, K=arguments.K, L1=arguments.L1, L=arguments.L2, arguments=arguments) for j in range(Jd)])
            gens.append([netG_images(1, K=arguments.K, L1=arguments.L1, L2=arguments.L2, arguments=arguments) for j in range(Jg)]) 

    # allocate v's
    vgens = initialize_v(M, Jg, gens, arguments)
    vdiscs = initialize_v(M, Jd, discs, arguments) 

    al = 0.9 
    eta = 0.001
    nf = 0.1

    true, false = 1, 0
    for ep in range(EP):
        for (ft, tar, lens), mix in zip(train_loader, loader_mix):
            if arguments.cuda:
                ft = ft.cuda()
                tar = tar.cuda()
                lens = lens.cuda()
                #mix = mix.cuda()

            # sort the tensors within batch
            if arguments.task == 'images' or arguments.task == 'toy_data':
                tar = tar.contiguous().view(-1, arguments.L2)
                tar, ft = Variable(tar), Variable(ft)
            else:
                ft, tar = ut.sort_pack_tensors(ft, tar, lens)

            
            for m in range(M):
                for i in range(Jg):
                    gens[m][i].zero_grad() 

                    errg = 0 
                    for j in range(Jd):
                        sz = ft.size(0)*Jg

                        labels = Variable(torch.ones(sz)*true).squeeze().float()
                        if arguments.cuda:
                            labels = labels.cuda()
                            h = torch.randn(sz, ft.size(1)).cuda()
                        else:
                            h = torch.randn(sz, ft.size(1))

                        out_g = gens[m][i].forward(Variable(h))
                        out_d, _ = discs[m][j].forward(out_g)  

                        errg = errg + criterion(out_d, labels)
                    errg.backward()

                    for k, p in enumerate(gens[m][i].parameters()): 
                        vgens[m][i][k] = -vgens[m][i][k]*(1-al) - eta*p.grad 
                        vr = float(np.sqrt(2*al*eta))*nf
                        if arguments.cuda:
                            vgens[m][i][k] = vgens[m][i][k] + Variable(vr*torch.randn(p.size()).cuda())   
                        else:
                            vgens[m][i][k] = vgens[m][i][k] + Variable(vr*torch.randn(p.size()))   

                        p.data.copy_((vgens[m][i][k]+p).data)

                        assert torch.equal(list(gens[m][i].parameters())[k].data, p.data), 'not assigning things'

            # discriminator updates
            for m in range(M): 
                for i in range(Jd):
                    discs[m][i].zero_grad()  
                   
                    sz = tar.size(0)
                    labels = Variable(torch.ones(sz)*true).squeeze().float()
                    if arguments.cuda:
                        labels = labels.cuda()

                    out_d_real, _ = discs[m][i].forward(tar)
                    errd_real = criterion(out_d_real, labels)

                    errd_fake = 0 
                    for j in range(Jg):
                        sz = ft.size(0)*Jd

                        labels = Variable(torch.ones(sz)*false).squeeze().float()
                        if arguments.cuda:
                            labels = labels.cuda()
                            h = torch.randn(sz, ft.size(1)).cuda()
                        else:
                            h = torch.randn(sz, ft.size(1))

                        out_g = gens[m][j].forward(Variable(h))
                        out_d_fake, _ = discs[m][i].forward(out_g)  

                        errd_fake = errd_fake + criterion(out_d_fake, labels)
                    errd_fake = errd_fake/Jg
                    errd = errd_real + errd_fake

                    errd.backward()

                    for k, p in enumerate(discs[m][i].parameters()): 
                        vdiscs[m][i][k] = -vdiscs[m][i][k]*(1-al) - eta*p.grad 
                        vr = float(np.sqrt(2*al*eta))*nf
                        if arguments.cuda:
                            vdiscs[m][i][k] = vdiscs[m][i][k] + Variable(vr*torch.randn(p.size()).cuda())   
                        else:
                            vdiscs[m][i][k] = vdiscs[m][i][k] + Variable(vr*torch.randn(p.size()))   

                        p.data.copy_((vdiscs[m][i][k]+p).data)

                        assert torch.equal(list(discs[m][i].parameters())[k].data, p.data), 'not assigning things'

            print('EP[{}/{}], errD = {}, errG = {} '.format(ep, EP, errd.data[0], errg.data[0]))

        if (ep % 100) == 0:
            sz = 1000
            if arguments.cuda:
                h = torch.randn(sz, ft.size(1)).cuda()
            else:
                h = torch.randn(sz, ft.size(1))

            samples = []
            probs = []
            for m in range(M): 
                for i in range(Jg):
                    sample = gens[m][i].forward(Variable(h))
                    samples.append(sample)    
                    prob, _ = discs[m][0].forward(sample)
                    probs.append(prob)

            all_samples = torch.cat(samples, dim=0).data.cpu().numpy()
            all_probs = torch.cat(probs, dim=1) / torch.cat(probs, dim=1).sum(1, keepdim=True) 
            all_pr_cat = torch.cat(probs, dim=0).data.cpu().numpy()
            gannum = torch.multinomial(all_probs)

            for i, samp in enumerate(samples):
                samples[i] = samp.unsqueeze(2)
            sample_tensor = torch.cat(samples, dim=2).data
            selected_tensor = torch.zeros(sz, arguments.L2)

            for i, gn in enumerate(gannum.data):
                selected_tensor[i] = sample_tensor[i, :, gn[0]]
            selected_tensor = selected_tensor.cpu().numpy()


            tardat = tar.data.cpu().numpy()
            plt.plot(tardat[:, 0], tardat[:, 1], 'o')

            plt.scatter(all_samples[:, 0], all_samples[:, 1], c=all_pr_cat.squeeze())
            plt.plot(selected_tensor[:, 0], selected_tensor[:, 1], 'v')
            
            
            plt.legend()
            plt.rc('legend',**{'fontsize':6})
            plt.title('Situation at iteration {}'.format(ep))

            folder_name = 'toy_example_figures_' + arguments.exp_info
            if not os.path.exists(folder_name):
                os.mkdir(folder_name) 

            if 1:
                plt.savefig(os.path.join(folder_name, 'iter_{}_experiment_{}'.format(ep, arguments.exp_info))) 
                plt.close()


def adversarial_wasserstein_trainer(loader_mix, train_loader, 
                                    generator, discriminator, EP=5,
                                    **kwargs):
    arguments = kwargs['arguments']
    conditional_gen = kwargs['conditional_gen']
    source_num = kwargs['source_num']
    

    def drawgendata_toy():
        samples = out_g.data.cpu().numpy()
        targets = tar.data.cpu().numpy() 

        npts = 30
        tol = 17
        xmin = np.min(arguments.means[:,0]) - tol 
        xmax = np.max(arguments.means[:,0]) + tol
        xs = np.linspace(xmin, xmax, npts) 

        ymin = np.min(arguments.means[:,1]) - tol
        ymax = np.max(arguments.means[:,1]) + tol
        ys = np.linspace(ymin, ymax, npts)
        
        X, Y = np.meshgrid(xs, ys) 

        xy_pairs = torch.from_numpy(np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)).float()

        if hasattr(arguments, 'pack_num'):
            N = xy_pairs.size(0)
            Ncut = int(N/arguments.pack_num)
            xy_pairs = torch.cat([xy_pairs[:Ncut, :], xy_pairs[Ncut:, :]], dim=1)
        if arguments.cuda:
            xy_pairs = xy_pairs.cuda()
        sigmoids, _ = discriminator.forward(Variable(xy_pairs))
        sigmoids = sigmoids.data.cpu().numpy().reshape(npts, npts) 

        cs = plt.contour(X, Y, sigmoids) 
        plt.clabel(cs, inline=1, fontsize=9) 

        plt.plot(targets[:, 0], targets[:, 1], 'x', label='True Data')
        plt.plot(samples[:, 0], samples[:, 1], 'o', label='Generated Data')
        plt.legend()
        plt.rc('legend',**{'fontsize':6})
        plt.title('Situation at iteration {}'.format(ep))

        folder_name = 'toy_example_figures_' + arguments.exp_info
        if not os.path.exists(folder_name):
            os.mkdir(folder_name) 

        if 1:
            plt.savefig(os.path.join(folder_name, 'iter_{}_experiment_{}'.format(ep, arguments.exp_info))) 

    def drawgendata_2d():

        mode = 'isomap'
        
        all_data = torch.cat([tar[0], out_g.data], 0) 
        all_data_numpy = all_data.cpu().numpy()
        N = all_data.size()[0]

        disc_values, _ = discriminator.forward(Variable(all_data)) 
        disc_values = disc_values.data.cpu().numpy()
        disc_vals_real = disc_values[:N/2]
        disc_vals_fake = disc_values[(N/2):]

        lowdim_out = ut.dim_red(all_data_numpy, 2, mode)
        #lowdim_data = ut.dim_red(tar.data.numpy(), 2, mode)

        plt.subplot2grid((2,2),(0,0)) 
        ss = 1
        
        sc = plt.scatter(lowdim_out[:(N/2):ss, 0], lowdim_out[:(N/2):ss, 1], c=disc_vals_real[::ss], marker='o')
        sc = plt.scatter(lowdim_out[N/2::ss, 0], lowdim_out[N/2::ss, 1], c=disc_vals_fake[::ss], marker='x')
    
        plt.colorbar(sc)
        plt.title('Scatter plot of real vs generated data')
        
        plt.subplot2grid((2, 2), (0, 1))
        plt.hist2d(lowdim_out[:N/2, 0], lowdim_out[:N/2:,1], bins=100,
                   norm=colors.LogNorm())
        plt.colorbar()
        plt.plot(lowdim_out[N/2:, 0], lowdim_out[N/2:, 1], 'rx', label='generated data')
        plt.title('Histogram of training data vs generated data')
        plt.legend()
        
        plt.subplot2grid((2,2), (1, 0), colspan=2)
        genspec = out_g.data.cpu().numpy().transpose()[:, :200]
        lrd.specshow(genspec, y_axis='log', x_axis='time') 

        plt.suptitle('Situation at iteration {}'.format(ep))
        #plt.scatter(lowdim_out[::10, 0], lowdim_out[::10, 1], c=disc_values[::10]) 

        folder_name = 'figures_' + arguments.exp_info
        if not os.path.exists(folder_name):
            os.mkdir(folder_name) 

        if 1:
            plt.savefig(os.path.join(folder_name, 'source_{}, iter_{}'.format(source_num,
                                                                              ep))) 
    
    if arguments.optimizer == 'Adam':
        optimizerD = optim.Adam(discriminator.parameters(), lr=arguments.lr, betas=(0.9, 0.999))
        optimizerG = optim.Adam(generator.parameters(), lr=arguments.lr, betas=(0.9, 0.999))
    elif arguments.optimizer == 'RMSprop':
        optimizerD = optim.RMSprop(discriminator.parameters(), lr=arguments.lr)
        optimizerG = optim.RMSprop(generator.parameters(), lr=arguments.lr)
    else:
        raise ValueError('Whaaaat?')

    if not arguments.cuda and arguments.plot_training:
        my_dpi = 96
        plt.figure(figsize=(1200/my_dpi, 600/my_dpi), dpi=my_dpi)

    one = torch.FloatTensor([1])
    mone = one * -1

    for ep in range(EP):
        for (ft, tar, lens), mix in zip(train_loader, loader_mix):
            if arguments.cuda:
                ft = ft.cuda()
                tar = tar.cuda()
                lens = lens.cuda()
                one, mone = one.cuda(), mone.cuda()
                #mix = mix.cuda()

            
            for p in discriminator.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update


            if ep < 25 or ep % 500 == 0:
                Diters = 100
            else:
                Diters = 5

            # sort the tensors within batch
            if arguments.task == 'images' or arguments.task == 'toy_data':
                tar = tar.contiguous().view(-1, arguments.L2)
                tar, ft = Variable(tar), (ft)
            else:
                ft, tar = ut.sort_pack_tensors(ft, tar, lens)

            if arguments.task != 'toy_data':
                ft = ft[0]
                tar = tar[0]

            for disc_ep in range(Diters):
                
                for p in discriminator.parameters():
                    p.data.clamp_(arguments.clamp_lower, 
                                  arguments.clamp_upper)


                # discriminator gradient with real data
                discriminator.zero_grad()
                out_d, _ = discriminator.forward(tar)

                err_D_real = out_d.mean()
                err_D_real.backward(one)

                # discriminator gradient with generated data
                
                out_g = generator.forward(Variable(ft, volatile=True))
                out_d_g, _ = discriminator.forward(Variable(out_g.data))
                err_D_fake = out_d_g.mean()
                err_D_fake.backward(mone)

                err_D = err_D_real - err_D_fake
                optimizerD.step()


            # show the current generated output
            if not arguments.cuda and arguments.plot_training:
                if arguments.task == 'atomic_sourcesep':
                    if ep % 100 == 0:
                        drawnow(drawgendata_atomic)
                        drawnow(drawgendata_2d) 
                elif arguments.task == 'images':
                    drawnow(drawgendata)
                elif arguments.task == 'toy_data':
                    drawnow(drawgendata_toy)
                else:
                    raise ValueError('Whhhhhaaaaat')
            else:

                if arguments.plot_training and (ep % 50 == 0):
                    my_dpi = 96
                    fig = plt.figure(figsize=(1200/my_dpi, 600/my_dpi), dpi=my_dpi)

                    if arguments.task == 'atomic_sourcesep':
                        drawgendata_2d()
                        plt.close(fig)
                    elif arguments.task == 'toy_data':
                        drawgendata_toy()
                        plt.close(fig)


            generator_params = list(generator.parameters())
            print(generator_params[0].data.sum())

            for p in discriminator.parameters():
                p.requires_grad = False # to avoid computation

            if ep == 100:
                pdb.set_trace()
            # generator gradient
            generator.zero_grad()
            out_g = generator.forward(Variable(ft))
            out_d_g, _ = discriminator.forward(out_g)
            err_G = out_d_g.mean()
            err_G.backward(one)

            optimizerG.step()
           
            if arguments.verbose:
                print('[%d/%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f\r'%
                      (ep, EP, err_D.data[0], err_G.data[0], err_D_real.data[0], 
                      err_D_fake.data[0]))
                  

def adversarial_trainer(loader_mix, train_loader, 
                        generator, discriminator, EP=5,
                        **kwargs):
    arguments = kwargs['arguments']
    criterion = kwargs['criterion']
    conditional_gen = kwargs['conditional_gen']
    source_num = kwargs['source_num']
 
    L1, L2 = generator.L1, generator.L2
    #K = generator.K
    def drawgendata():
        I = 1
        N = 3 if arguments.task == 'mnist' else 2

        for i in range(I):
            plt.subplot(N, I, i+1)
            plt.imshow(out_g[i].view(arguments.nfts, arguments.T).data.numpy())

            plt.subplot(N, I, I+i+1) 
            plt.imshow(tar[i].view(arguments.nfts, arguments.T).data.numpy())

            if arguments.task == 'mnist':
                plt.subplot(N, I, 2*I+i+1) 
                plt.imshow(ft[i].view(arguments.nfts, arguments.T).numpy())
    def drawgendata_atomic():
        I = 1
        N = 2 
        
        genspec = out_g.data.numpy().transpose()[:, :200]
        target = tar[0].numpy().transpose()[:, :200]

        #genspec = out_g[].data.numpy().transpose()
        #target = tar.permute(0,2,1).contiguous().view(-1, L2).numpy().transpose()
        for i in range(I):
                        
            plt.subplot(N, I, i+1)
            lrd.specshow(genspec, y_axis='log', x_axis='time') 
            
            plt.subplot(N, I, i+2) 
            lrd.specshow(target, y_axis='log', x_axis='time')

    def drawgendata_2d():

        mode = 'isomap'
        
        all_data = torch.cat([tar[0], out_g.data], 0) 
        all_data_numpy = all_data.cpu().numpy()
        N = all_data.size()[0]

        disc_values, _ = discriminator.forward(Variable(all_data)) 
        disc_values = disc_values.data.cpu().numpy()
        disc_vals_real = disc_values[:N/2]
        disc_vals_fake = disc_values[(N/2):]

        lowdim_out = ut.dim_red(all_data_numpy, 2, mode)
        #lowdim_data = ut.dim_red(tar.data.numpy(), 2, mode)

        plt.subplot2grid((2,2),(0,0)) 
        ss = 1
        
        sc = plt.scatter(lowdim_out[:(N/2):ss, 0], lowdim_out[:(N/2):ss, 1], c=disc_vals_real[::ss], marker='o', vmin=0, vmax=1)
        sc = plt.scatter(lowdim_out[N/2::ss, 0], lowdim_out[N/2::ss, 1], c=disc_vals_fake[::ss], marker='x', vmin=0, vmax=1)
    
        plt.colorbar(sc)
        plt.title('Scatter plot of real vs generated data')
        
        plt.subplot2grid((2, 2), (0, 1))
        plt.hist2d(lowdim_out[:N/2, 0], lowdim_out[:N/2:,1], bins=100,
                   norm=colors.LogNorm())
        plt.colorbar()
        plt.plot(lowdim_out[N/2:, 0], lowdim_out[N/2:, 1], 'rx', label='generated data')
        plt.title('Histogram of training data vs generated data')
        plt.legend()
        
        plt.subplot2grid((2,2), (1, 0), colspan=2)
        genspec = out_g.data.cpu().numpy().transpose()[:, :200]
        lrd.specshow(genspec, y_axis='log', x_axis='time') 

        plt.suptitle('Situation at iteration {}'.format(ep))
        #plt.scatter(lowdim_out[::10, 0], lowdim_out[::10, 1], c=disc_values[::10]) 

        folder_name = 'figures_' + arguments.exp_info
        if not os.path.exists(folder_name):
            os.mkdir(folder_name) 

        if 1:
            plt.savefig(os.path.join(folder_name, 'iter_{}, source_{}'.format(ep, 
                                                                              source_num))) 

    def drawgendata_toy():
        samples = out_g.data.cpu().numpy()
        targets = tar.data.cpu().numpy() 

        npts = 30
        tol = 17
        xmin = np.min(arguments.means[:,0]) - tol 
        xmax = np.max(arguments.means[:,0]) + tol
        xs = np.linspace(xmin, xmax, npts) 

        ymin = np.min(arguments.means[:,1]) - tol
        ymax = np.max(arguments.means[:,1]) + tol
        ys = np.linspace(ymin, ymax, npts)
        
        #X, Y = np.meshgrid(xs, ys) 

        #xy_pairs = torch.from_numpy(np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)).float()
        #if arguments.cuda:
        #    xy_pairs = xy_pairs.cuda()
        #sigmoids, _ = discriminator.forward(Variable(xy_pairs))

        #if hasattr(arguments, 'pack_num'):
        #    sigmoids = sigmoids.data.cpu().numpy().reshape(npts, npts) 

        #cs = plt.contour(X, Y, sigmoids) 
        #plt.clabel(cs, inline=1, fontsize=9) 

        plt.plot(samples[:, 0], samples[:, 1], 'o', label='Generated Data')
        plt.plot(targets[:, 0], targets[:, 1], 'x', label='True Data')
        plt.legend()
        plt.rc('legend',**{'fontsize':6})
        plt.title('Situation at iteration {}'.format(ep))

        folder_name = 'toy_example_figures_' + arguments.exp_info
        if not os.path.exists(folder_name):
            os.mkdir(folder_name) 

        if 1:
            plt.savefig(os.path.join(folder_name, 'iter_{}_experiment_{}'.format(ep, arguments.exp_info))) 


    # end of drawnow function

    if arguments.optimizer == 'Adam':
        optimizerD = optim.Adam(discriminator.parameters(), lr=arguments.lr, betas=(0.9, 0.999))
        optimizerG = optim.Adam(generator.parameters(), lr=arguments.lr, betas=(0.9, 0.999))
    elif arguments.optimizer == 'RMSprop':
        optimizerD = optim.RMSprop(discriminator.parameters(), lr=arguments.lr)
        optimizerG = optim.RMSprop(generator.parameters(), lr=arguments.lr)
    else:
        raise ValueError('Whaaaat?')

    if not arguments.cuda and arguments.plot_training:
        my_dpi = 96
        plt.figure(figsize=(1200/my_dpi, 600/my_dpi), dpi=my_dpi)
    true, false = 1, 0
    for ep in range(EP):
        for (ft, tar, lens), mix in zip(train_loader, loader_mix):
            if arguments.cuda:
                ft = ft.cuda()
                tar = tar.cuda()
                lens = lens.cuda()
                #mix = mix.cuda()

            # sort the tensors within batch
            if arguments.task == 'images' or arguments.task == 'toy_data' \
               or arguments.task == 'toy_spectrogram':
                tar = tar.contiguous().view(-1, arguments.L2)
                tar, ft = Variable(tar), Variable(ft)
            else:
                ft, tar = ut.sort_pack_tensors(ft, tar, lens)

            if ep < 25 or ep % 500 == 0:
                Diters = 100
            else:
                Diters = 5

            for disc_ep in range(Diters):
                # discriminator gradient with real data
                discriminator.zero_grad()
                out_d, _ = discriminator.forward(tar)
                if hasattr(arguments, 'pack_num'):
                    sz = out_d.size(0)#int(out_d.size(0)/arguments.pack_num)
                labels = Variable(torch.ones(sz)*true).squeeze().float()
                if arguments.cuda:
                    labels = labels.cuda()
                err_D_real = criterion(out_d, labels)
                err_D_real.backward()

                # discriminator gradient with generated data
                #if conditional_gen: 
                #    inp = mix.contiguous().view(-1, L1)
                #else:
                #    inp = ft_rshape # fixed_noise.contiguous().view(-1, L)

                out_g = generator.forward(ft)
                out_d_g, _ = discriminator.forward(out_g.detach())
                labels = Variable(torch.ones(sz)*false).squeeze().float()
                if arguments.cuda:
                    labels = labels.cuda()
                err_D_fake = criterion(out_d_g, labels) 
                err_D_fake.backward()

                err_D = err_D_real + err_D_fake
                optimizerD.step()


            # show the current generated output
            if not arguments.cuda:
                if arguments.task == 'atomic_sourcesep':
                    if ep % 100 == 0:
                        drawnow(drawgendata_atomic)
                        drawnow(drawgendata_2d) 
                elif arguments.task == 'images':
                    drawnow(drawgendata)
                elif arguments.task == 'toy_data':
                    drawnow(drawgendata_toy)
                else:
                    raise ValueError('Whhhhhaaaaat')
            else:
                if arguments.plot_training and (ep % 50 == 0):
                    my_dpi = 96
                    fig = plt.figure(figsize=(1200/my_dpi, 600/my_dpi), dpi=my_dpi)

                    if arguments.task == 'atomic_sourcesep':
                        drawgendata_2d()
                        plt.close(fig)
                    elif arguments.task == 'toy_data':
                        drawgendata_toy()
                        plt.close(fig)


            cnt = 1
            for gent_ep in range(1):
                #generator_params = list(generator.parameters())
                #print(generator_params[0].data.sum())

                # generator gradient
                generator.zero_grad()
                if arguments.feat_match:
                    _, out_h_data = discriminator.forward(tar)    
                    _, out_h_g = discriminator.forward(out_g) 
                    err_G = ((out_h_data.mean(0) - out_h_g.mean(0))**2).sum()
                else:
                    out_d_g, _ = discriminator.forward(out_g)
                    labels = Variable(torch.ones(out_d.size(0))*true).squeeze().float()
                    if arguments.cuda:
                        labels = labels.cuda()
                    err_G = criterion(out_d_g, labels)
                err_G.backward(retain_variables=True)

                #if out_d_g.mean().data.cpu().numpy() > 0.3:
                #    break

                #print(generator_params[0].data.sum())

                optimizerG.step()
                #generator_params2 = list(generator.parameters())
                #diff = (generator_params[0] - generator_params2[0]).sum().data.cpu().numpy()
                #print('Diff is {}'.format(diff))

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
        I = 1
        N = 3 if arguments.task == 'mnist' else 2

        for i in range(I):
            plt.subplot(N, I, i+1)
            plt.imshow(out_g[i].view(arguments.nfts, arguments.T).data.numpy())

            plt.subplot(N, I, I+i+1) 
            plt.imshow(tar[i].view(arguments.nfts, arguments.T).numpy())

            if arguments.task == 'mnist':
                plt.subplot(N, I, 2*I+i+1) 
                plt.imshow(ft[i].view(arguments.nfts, arguments.T).numpy())
    def drawgendata_atomic():
        I = 1
        N = 5 
        
        genspec = out_g.data.numpy().transpose()[:, :200]
        target = tar.data.numpy().transpose()[:, :200]
        feat = ft[0].numpy().transpose()[:, :200]


        #genspec = out_g[].data.numpy().transpose()
        #target = tar.permute(0,2,1).contiguous().view(-1, L2).numpy().transpose()

        for i in range(I):
            plt.subplot(N, I, i+1)
            plt.imshow(genspec) 
            
            plt.subplot(N, I, I+i+1) 
            plt.imshow(target)
            
            plt.subplot(N, I, I+i+2)
            lrd.specshow(genspec, y_axis='log', x_axis='time') 
            
            plt.subplot(N, I, I+i+3) 
            lrd.specshow(target, y_axis='log', x_axis='time')

            plt.subplot(N, I, I+i+4) 
            lrd.specshow(feat, y_axis='log', x_axis='time')

    if arguments.optimizer == 'Adam':
        optimizerG = optim.Adam(generator.parameters(), lr=arguments.lr, betas=(0.9, 0.999))
    elif arguments.optimizer == 'RMSprop':
        optimizerG = optim.RMSprop(generator.parameters(), lr=arguments.lr)

    if not arguments.cuda and arguments.plot_training:
        figure(figsize=(4,4))
    true, false = 1, 0
    for ep in range(EP):
        for (ft, tar, lens), mix in zip(train_loader, loader_mix):
            if arguments.cuda:
                tar = tar.cuda()
                ft = ft.cuda()
                lens = lens.cuda()

            # sort the tensors within batch
            if arguments.task == 'images':
                tar = tar.contiguous().view(-1, arguments.L2)
                tar, ft = Variable(tar), Variable(ft)
            else:
                ft, tar = ut.sort_pack_tensors(ft, tar, lens)
                tar = Variable(tar[0])

            #if conditional_gen: 
            #    inp = mix.contiguous().view(-1, L)
            #else:
            #    inp = ft_rshape   # fixed_noise.contiguous().view(-1, L)

            # generator gradient
            generator.zero_grad()
            out_g = generator.forward(ft)
            err_G = criterion(out_g, tar)
            err_G.backward()

            # show the current generated output
            if not arguments.cuda:
                if arguments.task == 'atomic_sourcesep':
                    drawnow(drawgendata_atomic)
                else:
                    drawnow(drawgendata)

            optimizerG.step()

            print(err_G)
            print(ep)

def moment_trainer(loader_mix, train_loader, generator, discriminator, EP=5,
                   **kwargs):
    arguments = kwargs['arguments']
    criterion = kwargs['criterion']
    conditional_gen = kwargs['conditional_gen']
    source_num = kwargs['source_num']
 
    L1, L2 = generator.L1, generator.L2
    #K = generator.K
    def drawgendata():
        I = 1
        N = 3 if arguments.task == 'mnist' else 2

        for i in range(I):
            plt.subplot(N, I, i+1)
            plt.imshow(out_g[i].view(arguments.nfts, arguments.T).data.numpy())

            plt.subplot(N, I, I+i+1) 
            plt.imshow(tar[i].view(arguments.nfts, arguments.T).data.numpy())

            if arguments.task == 'mnist':
                plt.subplot(N, I, 2*I+i+1) 
                plt.imshow(ft[i].view(arguments.nfts, arguments.T).numpy())
    def drawgendata_atomic():
        I = 1
        N = 2 
        
        genspec = out_g.data.numpy().transpose()[:, :200]
        target = tar[0].numpy().transpose()[:, :200]

        #genspec = out_g[].data.numpy().transpose()
        #target = tar.permute(0,2,1).contiguous().view(-1, L2).numpy().transpose()
        for i in range(I):
                        
            plt.subplot(N, I, i+1)
            lrd.specshow(genspec, y_axis='log', x_axis='time') 
            
            plt.subplot(N, I, i+2) 
            lrd.specshow(target, y_axis='log', x_axis='time')

    def drawgendata_2d():

        mode = 'isomap'
        
        all_data = torch.cat([tar[0], out_g.data], 0) 
        all_data_numpy = all_data.cpu().numpy()
        N = all_data.size()[0]

        disc_values, _ = discriminator.forward(Variable(all_data)) 
        disc_values = disc_values.data.cpu().numpy()
        disc_vals_real = disc_values[:N/2]
        disc_vals_fake = disc_values[(N/2):]

        lowdim_out = ut.dim_red(all_data_numpy, 2, mode)
        #lowdim_data = ut.dim_red(tar.data.numpy(), 2, mode)

        plt.subplot2grid((2,2),(0,0)) 
        ss = 1
        
        sc = plt.scatter(lowdim_out[:(N/2):ss, 0], lowdim_out[:(N/2):ss, 1], c=disc_vals_real[::ss], marker='o', vmin=0, vmax=1)
        sc = plt.scatter(lowdim_out[N/2::ss, 0], lowdim_out[N/2::ss, 1], c=disc_vals_fake[::ss], marker='x', vmin=0, vmax=1)
    
        plt.colorbar(sc)
        plt.title('Scatter plot of real vs generated data')
        
        plt.subplot2grid((2, 2), (0, 1))
        plt.hist2d(lowdim_out[:N/2, 0], lowdim_out[:N/2:,1], bins=100,
                   norm=colors.LogNorm())
        plt.colorbar()
        plt.plot(lowdim_out[N/2:, 0], lowdim_out[N/2:, 1], 'rx', label='generated data')
        plt.title('Histogram of training data vs generated data')
        plt.legend()
        
        plt.subplot2grid((2,2), (1, 0), colspan=2)
        genspec = out_g.data.cpu().numpy().transpose()[:, :200]
        lrd.specshow(genspec, y_axis='log', x_axis='time') 

        plt.suptitle('Situation at iteration {}'.format(ep))
        #plt.scatter(lowdim_out[::10, 0], lowdim_out[::10, 1], c=disc_values[::10]) 

        folder_name = 'figures_' + arguments.exp_info
        if not os.path.exists(folder_name):
            os.mkdir(folder_name) 

        if 1:
            plt.savefig(os.path.join(folder_name, 'iter_{}, source_{}'.format(ep, 
                                                                              source_num))) 

    def drawgendata_toy():
        samples = out_g.data.cpu().numpy()
        targets = tar.data.cpu().numpy() 

        npts = 30
        tol = 17
        xmin = np.min(arguments.means[:,0]) - tol 
        xmax = np.max(arguments.means[:,0]) + tol
        xs = np.linspace(xmin, xmax, npts) 

        ymin = np.min(arguments.means[:,1]) - tol
        ymax = np.max(arguments.means[:,1]) + tol
        ys = np.linspace(ymin, ymax, npts)
        
        X, Y = np.meshgrid(xs, ys) 

        xy_pairs = torch.from_numpy(np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)).float()
        if arguments.cuda:
            xy_pairs = xy_pairs.cuda()
        sigmoids, _ = discriminator.forward(Variable(xy_pairs))
        sigmoids = sigmoids.data.cpu().numpy().reshape(npts, npts) 

        #cs = plt.contour(X, Y, sigmoids) 
        #plt.clabel(cs, inline=1, fontsize=9) 

        plt.plot(samples[:, 0], samples[:, 1], 'o', label='Generated Data')
        plt.plot(targets[:, 0], targets[:, 1], 'x', label='True Data')
        plt.legend()
        plt.rc('legend',**{'fontsize':6})
        plt.title('Situation at iteration {}'.format(ep))

        folder_name = 'toy_example_figures_' + arguments.exp_info
        if not os.path.exists(folder_name):
            os.mkdir(folder_name) 

        if 1:
            plt.savefig(os.path.join(folder_name, 'iter_{}_experiment_{}'.format(ep, arguments.exp_info))) 


    # end of drawnow function

    if arguments.optimizer == 'Adam':
        optimizerD = optim.Adam(discriminator.parameters(), lr=arguments.lr, betas=(0.9, 0.999))
        optimizerG = optim.Adam(generator.parameters(), lr=arguments.lr, betas=(0.9, 0.999))
    elif arguments.optimizer == 'RMSprop':
        optimizerD = optim.RMSprop(discriminator.parameters(), lr=arguments.lr)
        optimizerG = optim.RMSprop(generator.parameters(), lr=arguments.lr)
    else:
        raise ValueError('Whaaaat?')

    if not arguments.cuda and arguments.plot_training:
        my_dpi = 96
        plt.figure(figsize=(1200/my_dpi, 600/my_dpi), dpi=my_dpi)
    true, false = 1, 0
    for ep in range(EP):
        for (ft, tar, lens), mix in zip(train_loader, loader_mix):
            if arguments.cuda:
                ft = ft.cuda()
                tar = tar.cuda()
                lens = lens.cuda()
                #mix = mix.cuda()

            # sort the tensors within batch
            if arguments.task == 'images' or arguments.task == 'toy_data':
                tar = tar.contiguous().view(-1, arguments.L2)
                tar, ft = Variable(tar, volatile = False), Variable(ft)
            else:
                ft, tar = ut.sort_pack_tensors(ft, tar, lens)

            generator.zero_grad()

            out_g = generator.forward(ft)
        
            tar_mean = tar.mean(0)
            gen_mean = out_g.mean(0)

            tar_cov = torch.mm((tar - tar_mean).permute(1, 0), (tar - tar_mean))/tar.size(0)
            gen_cov = torch.mm((out_g - gen_mean).permute(1, 0), (out_g - gen_mean))/tar.size(0)

            errG = torch.mean(torch.abs(gen_cov - tar_cov)) + \
                   torch.mean(torch.abs(tar_mean - gen_mean))
            errG.backward()

            optimizerG.step()


            # show the current generated output
            if not arguments.cuda:
                if arguments.task == 'atomic_sourcesep':
                    if ep % 100 == 0:
                        drawnow(drawgendata_atomic)
                        drawnow(drawgendata_2d) 
                elif arguments.task == 'images':
                    drawnow(drawgendata)
                elif arguments.task == 'toy_data':
                    drawnow(drawgendata_toy)
                else:
                    raise ValueError('Whhhhhaaaaat')
            else:
                if arguments.plot_training and (ep % 50 == 0):
                    my_dpi = 96
                    fig = plt.figure(figsize=(1200/my_dpi, 600/my_dpi), dpi=my_dpi)

                    if arguments.task == 'atomic_sourcesep':
                        drawgendata_2d()
                        plt.close(fig)
                    elif arguments.task == 'toy_data':
                        drawgendata_toy()
                        plt.close(fig)

            # generator gradient
            print(errG.mean())
            print(ep)

def propose_parameters(generator1, generator2, st):
    for p1, p2 in zip(generator1.parameters(), generator2.parameters()):
        p1.data.copy_(p2.data + st*torch.randn(p2.data.size()))

def assign_parameters(generator1, generator2):
    for p1, p2 in zip(generator1.parameters(), generator2.parameters()):
        p1.data.copy_(p2.data) 

def VAE_trainer_MC(loader_mix, train_loader,
                   generator, EP = 5, 
                   **kwargs):

    def plot_VAE():
        if ep % 1 == 0:
            plt.subplot(1,2,1)
            tardat = tar.data.cpu().numpy()
            plt.plot(tardat[:, 0], tardat[:, 1], 'o')

            all_samples = out_g.data.cpu().numpy()
            plt.plot(all_samples[:, 0], all_samples[:, 1], 'x')
            
            plt.legend()
            plt.rc('legend',**{'fontsize':6})
            plt.title('Situation at iteration {}'.format(ep))

            if 0:
                folder_name = 'toy_example_figures_' + arguments.exp_info
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name) 

                plt.savefig(os.path.join(folder_name, 'iter_{}_experiment_{}'.format(ep, arguments.exp_info))) 

            plt.subplot(1,2,2)
            if hgen.size(1) == 1:
                hnp = hgen.data.cpu().numpy()
                plt.hist(hnp,50)
            elif hgen.size(1) == 2:
                hnp = hgen.data.cpu().numpy()
                plt.plot(hnp[:,0], hnp[:,1],'o')
            plt.title('p(h|x)')


        if ep == EP-1:
            plt.subplot(1,2,1)
            h = torch.randn(2000, generator.arguments.Kdisc)
            gendata = generator.decode(Variable(h))
            gendata = gendata.data.cpu().numpy()
            plt.plot(gendata[:, 0], gendata[:, 1], 'v', label='test_gen')
            plt.legend()
            pdb.set_trace()

    arguments = kwargs['arguments']
    criterion = kwargs['criterion']
    conditional_gen = kwargs['conditional_gen']


    L1 = generator.L1
    L2 = generator.L2
    K = generator.K

    generator_proposal = VAE(1, K=K, L1=L1, L2=L2, arguments=arguments)

    generator.train(mode=True)
    generator_proposal.train(mode=True)

    if not arguments.cuda and arguments.plot_training:
        figure(figsize=(4,4))
    true, false = 1, 0
    for ep in range(EP):
        for (ft, tar, lens), mix in zip(train_loader, loader_mix):
            if arguments.cuda:
                tar = tar.cuda()
                ft = ft.cuda()
                lens = lens.cuda()

            if arguments.task == 'images' or arguments.task == 'toy_data':
                tar = tar.contiguous().view(-1, arguments.L2)
                tar, ft = Variable(tar), Variable(ft)
            else:
                ft, tar = ut.sort_pack_tensors(ft, tar, lens)

            out_g, mu, logvar, hgen = generator.forward(tar)
            err_G = criterion(out_g, tar, mu, logvar)

            propose_parameters(generator_proposal, generator, 0.003)

            out_g_proposed, mu, logvar, h = generator_proposal.forward(tar)
            err_G_proposal = criterion(out_g_proposed, tar, mu, logvar)

            drawnow(plot_VAE)
            if err_G_proposal.data[0] < err_G.data[0]: 
                params1 = list(generator.parameters())[0]
                params2 = list(generator_proposal.parameters())[0]


                assign_parameters(generator, generator_proposal)
                print('[{}/{}] accept!, errG {}, errG_proposal {}'.format(ep, EP,
                                                                        err_G.data[0],
                                                                        err_G_proposal.data[0]))
                params1 = list(generator.parameters())
                params2 = list(generator_proposal.parameters())

                
            else:
                print('[{}/{}] reject!, errG {}, errG_proposal {}'.format(ep, EP,
                                                                        err_G.data[0],
                                                                        err_G_proposal.data[0]))



def VAE_trainer(loader_mix, train_loader, 
                generator, EP = 5,
                **kwargs):

    def drawgendata():
        I = 3
        N = 2

        for i in range(I):
            plt.subplot(N, I, i+1)
            img = out_g[i].data.contiguous().view(arguments.nfts, arguments.T)
            plt.imshow(img.numpy())

            plt.subplot(N, I, I+i+1) 
            plt.imshow(tar[i].data.view(arguments.nfts, arguments.T).numpy())

            
    def plot_VAE():
        if ep % 1 == 0:
            plt.subplot(1,2,1)
            tardat = tar.data.cpu().numpy()
            plt.plot(tardat[:, 0], tardat[:, 1], 'o', label='target')

            all_samples = out_g.data.cpu().numpy()
            plt.plot(all_samples[:, 0], all_samples[:, 1], 'x', label='train_gen')

            if arguments.out_type == 'implicit':
                mu_forward = generator.fc3.forward(mu).data.cpu().numpy()
            #plt.plot(mu_forward[:, 0], mu_forward[:, 1], 'v', label='means')

            #hgen = torch.randn(2000, generator.arguments.Kdisc)
            gendata = generator.decode((h))
            gendata = gendata.data.cpu().numpy()
            #plt.plot(gendata[:, 0], gendata[:, 1], 'v', label='test_gen')

            plt.legend()
            plt.rc('legend',**{'fontsize':6})
            plt.title('Situation at iteration {}'.format(ep))

            if 0:
                folder_name = 'toy_example_figures_' + arguments.exp_info
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name) 

                plt.savefig(os.path.join(folder_name, 'iter_{}_experiment_{}'.format(ep, arguments.exp_info))) 

            plt.subplot(1,2,2)
            hnp = h.data.cpu().numpy()
            if h.size(1) == 1:
                plt.hist(hnp,50)
            elif h.size(1) == 2:
                plt.plot(hnp[:,0], hnp[:,1],'o', label='h forward')
                if arguments.out_type == 'implicit':
                    hhatnp = hhat.data.cpu().numpy()
                    munp = mu.data.cpu().numpy()
                    plt.plot(hhatnp[:,0], hhatnp[:,1], 'o', label='hhat backward')
                    plt.plot(munp[:, 0], munp[:,1], 'v', label='mu forward')

            else: 
                end = 100
                plt.imshow(hnp[:end],interpolation='None')
            plt.title('p(h|x)')
            plt.legend()
        # generate some data and show
        if ep == EP-1:
            plt.subplot(1,2,1)
            #hgen = torch.randn(2000, generator.arguments.Kdisc)
            gendata = generator.decode(h)
            gendata = gendata.data.cpu().numpy()
            #plt.plot(gendata[:, 0], gendata[:, 1], 'v', label='test_gen')
            plt.legend()

            root = os.path.expanduser('~')
            path = os.path.join(root, 'Dropbox', 'GANs', 'BayesGAN', 'linear_mm.eps')
            plt.savefig(path, format='eps')

            pdb.set_trace()

    arguments = kwargs['arguments']
    criterion = kwargs['criterion']
    conditional_gen = kwargs['conditional_gen']

    generator.train(mode=True)

    L1 = generator.L1
    L2 = generator.L2
    K = generator.K

    if arguments.optimizer == 'Adam':
        optimizerG = optim.Adam(generator.parameters(), lr=arguments.lr, betas=(0.9, 0.999))
    elif arguments.optimizer == 'RMSprop':
        optimizerG = optim.RMSprop(generator.parameters(), lr=arguments.lr)
    elif arguments.optimizer == 'SGD':
        optimizerG = optim.SGD(generator.parameters(), lr=arguments.lr)
    elif arguments.optimizer == 'LBFGS':
        optimizerG = optim.LBFGS(generator.parameters(), lr=arguments.lr)


    if not arguments.cuda and arguments.plot_training:
        plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    true, false = 1, 0
    for ep in range(EP):
        for (ft, tar, lens), mix in zip(train_loader, loader_mix):
            if arguments.cuda:
                tar = tar.cuda()
                ft = ft.cuda()
                lens = lens.cuda()

            if arguments.task == 'images' or arguments.task == 'toy_data':
                tar = tar.contiguous().view(-1, arguments.L2)
                tar, ft = Variable(tar), Variable(ft)
            else:
                ft, tar = ut.sort_pack_tensors(ft, tar, lens)

            # generator gradient
            generator.zero_grad()
            if arguments.out_type == 'noise':
                out_g, mu, logvar, h = generator.forward(tar)
                err_G = criterion(out_g, tar, mu, logvar)
            elif arguments.out_type == 'implicit':
                hhat, mu, logvar, h, logdet, out_g = generator.forward(tar)
                err_G = criterion(hhat, logdet, mu, logvar, tar)

            err_G.backward()

            param_norms = []
            for p in generator.parameters():
                param_norms.append(p.data.abs().mean())

            # step 
            optimizerG.step()
            print('EP [{}/{}, error = {}]'.format(ep, EP, 
                                                  err_G.data[0])) 

            
            if arguments.task == 'toy_data':
                drawnow(plot_VAE)
            else:
                drawnow(drawgendata)

                                                                                    

def reconstruct_tester(generators, source_num, 
                       loader_mix, EP, **kwargs):
    generator1, generator2 = generators
    

    arguments = kwargs['arguments']
    optimizer = arguments.optimizer
    loss = kwargs['loss']
    exp_info = kwargs['exp_info']
    L1 = generator1.L1
    L2 = generator1.L2

    for i, (MSabs, MSphase, SPCS1abs, SPCS2abs, wavfls1, wavfls2, lens1, lens2) in enumerate(islice(loader_mix, 0, 1, 1)): 

        source_tar = SPCS1abs if source_num == 1 else SPCS2abs

        eps = 1e-20
        if arguments.cuda:
            source_tar = source_tar.cuda()

        print('Processing source ',i)
        Nmix = source_tar.size(0)
        T = source_tar.size(1)
        source_tar = source_tar.contiguous().view(-1, L2) 

        if arguments.test_method == 'optimize':
            c = 1
            if arguments.cuda:
                x = Variable(c*torch.randn(Nmix*T, L1).cuda(), requires_grad=True)
            else:
                x = Variable(c*torch.randn(Nmix*T, L1), requires_grad=True)

            if optimizer == 'Adam':
                optimizer_sourcesep = optim.Adam([x], lr=1e-3, betas=(0.5, 0.999))
            elif optimizer == 'RMSprop':
                optimizer_sourcesep = optim.RMSprop([x], lr=1e-3)
            else:
                raise ValueError('Whaaaaaaaat')

            for ep in range(EP):

                source_hat = generator1.forward(x) if source_num == 1 else generator2.forward(x)

                if loss == 'Euclidean': 
                    err = torch.mean((Variable(source_tar) - source_hat)**2)
                elif loss == 'Poisson':
                    err = torch.mean(-Variable(source_tar)*torch.log(source_hat+eps) + source_hat)

                if arguments.smooth_source_estimates:
                    serr = 0*torch.mean(torch.abs(source_hat[1:] - source_hat[:-1]))
                    err = err + serr
                err.backward()

                optimizer_sourcesep.step()

                x.grad.data.zero_()

                print('Step in batch [{:d}\{:d}]'.format(ep+1, EP))
                print('The error is ', err)
                if arguments.smooth_source_estimates:
                    print('The smoothness error is ', serr)


            # get the final source estimate
            source_hat = generator1.forward(x) if source_num == 1 else generator2.forward(x)
            curdir = os.getcwd() 
            reconspath = os.path.join(curdir, 'reconstructions')
            
            if not os.path.exists(reconspath):
                os.mkdir(reconspath)

            fn = np.sqrt
            plt.subplot(211)
            lrd.specshow(fn(source_tar.cpu().numpy()).transpose(), y_axis='log', x_axis='time')
            plt.title('Original source')

            plt.subplot(212)
            lrd.specshow(fn(source_hat.data.cpu().numpy()).transpose(), y_axis='log', x_axis='time')
            plt.title('Reconstructed source')
            plt.savefig(os.path.join(reconspath, 
                        '_'.join([exp_info, 
                            'reconstruction_{}'.format(source_num)])+'.png'))
     

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
    num_ims, c = 30, 2
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

def ML_separate_audio_sources(generators, loader_mix, EP, **kwargs):

    generator1, generator2 = generators
    discriminator1, discriminator2 = kwargs['discriminators']

    arguments = kwargs['arguments']
    optimizer = arguments.optimizer
    alpha = kwargs['alpha'] 
    loss = kwargs['loss']
    exp_info = kwargs['exp_info']
    L1 = generator1.L1
    L2 = generator1.L2
    if arguments.tr_method == 'VAE':
        generator1.eval()
        generator2.eval()

    s1s, s2s = [], [] 
    s1hats, s2hats = [], []
    mixes = []
    all_mags1, all_mags2 = [], []
    for i, (MSabs, MSphase, SPCS1abs, SPCS2abs, wavfls1, wavfls2, lens1, lens2) in enumerate(islice(loader_mix, 0, 1, 1)): 

        eps = 1e-20
        if arguments.cuda:
            MSabs = MSabs.cuda()
            SPCS1abs = SPCS1abs.cuda()
            SPCS2abs = SPCS2abs.cuda()

        print('Processing source ',i)
        Nmix = MSabs.size(0)
        T = MSabs.size(1)
        MSabs = MSabs.contiguous().view(-1, L2) 

        if arguments.test_method == 'optimize':
            c = 1
            if arguments.cuda:
                x1 = Variable(c*torch.randn(Nmix*T, L1).cuda(), requires_grad=True)
                x2 = Variable(c*torch.randn(Nmix*T, L1).cuda(), requires_grad=True)
            else:
                x1 = Variable(c*torch.randn(Nmix*T, L1), requires_grad=True)
                x2 = Variable(c*torch.randn(Nmix*T, L1), requires_grad=True)

            if optimizer == 'Adam':
                optimizer_sourcesep = optim.Adam([x1, x2], lr=1e-3, betas=(0.5, 0.999))
            elif optimizer == 'RMSprop':
                optimizer_sourcesep = optim.RMSprop([x1, x2], lr=1e-3)
            else:
                raise ValueError('Whaaaaaaaat')

            for ep in range(EP):
                if arguments.tr_method == 'VAE':
                    source1hat, _, _ = generator1.forward(x1) 
                    source2hat, _, _ = generator2.forward(x2)  
                else:
                    source1hat = generator1.forward(x1)
                    source2hat = generator2.forward(x2)

                mix_sum = source1hat + source2hat
                if loss == 'Euclidean': 
                    err = torch.mean((Variable(MSabs) - mix_sum)**2)
                elif loss == 'Poisson':
                    err = torch.mean(-Variable(MSabs)*torch.log(mix_sum+eps) + mix_sum)

                if arguments.smooth_source_estimates:
                    serr = 0.1*torch.mean(torch.abs(source1hat[1:] - source1hat[:-1]) \
                              +torch.abs(source2hat[1:] - source2hat[:-1]))
                    err = err + serr
                # if adversarial use discriminator information
                if arguments.tr_method in ['adversarial', 'adversarial_wasserstein']:
                    if arguments.feat_match: 
                        _, source1hat_feat = discriminator1.forward(source1hat)
                        _, source2hat_feat = discriminator2.forward(source2hat)

                        source1hat_feat = source1hat_feat.mean(0)
                        source2hat_feat = source2hat_feat.mean(0)

                        _, source1_feat = discriminator1.forward(Variable(SPCS1abs.contiguous().view(-1, arguments.L2)))
                        _, source2_feat = discriminator2.forward(Variable(SPCS2abs.contiguous().view(-1, arguments.L2)))
                        source1_feat = source1_feat.mean(0)
                        source2_feat = source2_feat.mean(0)

                        source1hat_cl = -((source1_feat - source1hat_feat)**2).mean()
                        source2hat_cl = -((source2_feat - source2hat_feat)**2).mean()
                    else:
                        source1hat_cl, _ = discriminator1.forward(source1hat)
                        source2hat_cl, _ = discriminator2.forward(source2hat)

                        if arguments.tr_method == 'adversarial': 
                            source1hat_cl = torch.log(source1hat_cl + eps)
                            source2hat_cl = torch.log(source2hat_cl + eps)

                    err = err - alpha*(source1hat_cl.mean() + source2hat_cl.mean()) 
                err.backward()

                optimizer_sourcesep.step()

                x1.grad.data.zero_()
                x2.grad.data.zero_()

                print('Step in batch [{:d}\{:d}]'.format(ep+1, EP))
                print('The error is ', err)
                if arguments.smooth_source_estimates:
                    print('The smoothness error is ', serr)

            if arguments.tr_method == 'VAE':
                temp1, _, _ = generator1.forward(x1)
                temp1 = temp1.data.cpu().numpy() 

                temp2, _, _ = generator2.forward(x2)
                temp2 = temp2.data.cpu().numpy() 
            else:
                temp1 = generator1.forward(x1).data.cpu().numpy() 
                temp2 = generator2.forward(x2).data.cpu().numpy() 

        elif arguments.test_method == 'sample':
            pf_nsamples = 5 
            x1s, x2s = [], []
            for n, mix_frame in enumerate(MSabs):
                source1_smps = sample_outputs(generator1, pf_nsamples, arguments) 
                source2_smps = sample_outputs(generator2, pf_nsamples, arguments) 
                all_pairs = it.product(source1_smps, source2_smps)

                all_errs = []
                for pair in all_pairs:
                    mix_sum = pair[0] + pair[1]
                    
                    loss = 'Poisson'
                    if loss == 'Euclidean': 
                        err = torch.mean((mix_frame - mix_sum)**2)
                    elif loss == 'Poisson':
                        eps = 1e-20
                        err = torch.mean(-mix_frame*torch.log(mix_sum+eps) + mix_sum)
                    all_errs.append(err)
                all_pairs = it.product(source1_smps, source2_smps)
                min_ind = np.argmin(all_errs)
                x1_t, x2_t = next(it.islice(all_pairs, min_ind, min_ind+1))
                x1s.append(x1_t), x2s.append(x2_t)
                print(n)
            temp1 = torch.cat(x1s, 0).cpu().numpy()
            temp2 = torch.cat(x2s, 0).cpu().numpy()
        else:
            raise ValueError('Whaaaaat')
       
        if arguments.wiener_recons:
            temp1_audio, mags1 = ut.mag2spec_and_audio_wiener(temp1, 
                                                              temp1+temp2,
                                                              MSabs,
                                                              MSphase, arguments)
        else: 
            temp1_audio, mags1 = ut.mag2spec_and_audio(temp1, MSphase, arguments)

        all_mags1.extend(mags1)
        s1hats.extend(temp1_audio)

        if arguments.wiener_recons:
            temp2_audio, mags2 = ut.mag2spec_and_audio_wiener(temp2, 
                                                              temp1+temp2,
                                                              MSabs,
                                                              MSphase, arguments)
        else:
            temp2_audio, mags2 = ut.mag2spec_and_audio(temp2, MSphase, arguments)
        all_mags2.extend(mags2)
        s2hats.extend(temp2_audio)

        s1s.extend(np.split(wavfls1.cpu().numpy(), Nmix, 0))
        s2s.extend(np.split(wavfls2.cpu().numpy(), Nmix, 0))

    curdir = os.getcwd() 
    magpath = os.path.join(curdir, '_'.join(['spectrograms', exp_info]))
    soundpath = os.path.join(curdir, '_'.join(['sounds', exp_info]))
    
    if not os.path.exists(magpath):
        os.mkdir(magpath)
    if not os.path.exists(soundpath):
        os.mkdir(soundpath)
    
    for i, (s1, s2, mag1, mag2) in enumerate(zip(s1s, s2s, all_mags1, all_mags2)):
        print('Processing the mixture {}'.format(i))

        mag1, mag2 = mag1.transpose(), mag2.transpose()
        s1, s2 = s1.squeeze(), s2.squeeze()
        fn = np.sqrt

        if arguments.save_files:

            plt.subplot(3, 2, 1)
            lrd.specshow(fn(np.abs(lr.stft(s1+s2, n_fft=1024))), y_axis='log', x_axis='time')
            plt.title('Observed Mixture')

            plt.subplot(3, 2, 3)
            lrd.specshow(fn(np.abs(lr.stft(s1, n_fft=1024))), 
                         y_axis='log', x_axis='time')
            plt.title('Source 1')

            plt.subplot(3, 2, 5) 
            lrd.specshow(fn(np.abs(lr.stft(s2, n_fft=1024))),
                         y_axis='log', x_axis='time')
            plt.title('Source 2')

            plt.subplot(3, 2, 2)
            lrd.specshow(fn(mag1+mag2),
                         y_axis='log', x_axis='time')
            plt.title('Reconstruction')

            plt.subplot(3, 2, 4)
            lrd.specshow(fn(mag1), y_axis='log', x_axis='time')
            plt.title('Sourcehat 1')

            plt.subplot(3, 2, 6) 
            lrd.specshow(fn(mag2),  y_axis='log', x_axis='time')
            plt.title('Sourcehat 2')
        
            figname = 'spectrograms_{}'.format(i)
            plt.savefig(os.path.join(magpath, figname)) 

            # save file 1
            filename1 = arguments.exp_info + 'source1hat_{}.wav'.format(i)
            filepath1 = os.path.join(soundpath, filename1)  

            # save file 2
            filename2 = arguments.exp_info + 'source2hat_{}.wav'.format(i)
            filepath2 = os.path.join(soundpath, filename2)  
            
            lr.output.write_wav(filepath2, s2hats[i], arguments.fs)
            lr.output.write_wav(filepath1, s1hats[i], arguments.fs)

    recordspath = os.path.join(curdir, 'records')
    if not os.path.exists(recordspath):
        os.mkdir(recordspath)
    
    bss_evals = ut.audio_to_bsseval(s1hats, s2hats, s1s, s2s)
    bss_df = ut.compile_bssevals(bss_evals) 

    if arguments.save_records:
        bss_df.to_csv(os.path.join(recordspath, '_'.join(['bss_evals', exp_info]) + '.csv' ))
    return bss_df
           


def sample_outputs(generator, Nsamples, arguments):
    inp = torch.randn(Nsamples, arguments.L1) 
    if arguments.cuda:
        inp = inp.cuda()

    out = generator.forward(Variable(inp))
    if arguments.task == 'images':
        out = out.contiguous().view(-1, arguments.nfts, arguments.T)
    return torch.split(out.data, split_size=1, dim=0)



