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
import torch.nn.utils.rnn as rnn_utils
from itertools import islice
import os
import torch.nn.init as torchinit
import librosa as lr
import librosa.display as lrd
import utils as ut

def initializationhelper(param, nltype):
    torchinit.xavier_uniform(param.weight, gain=torchinit.calculate_gain(nltype))
    c = 0.01
    torchinit.uniform(param.bias, a=-c, b=c)

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
        initializationhelper(self.l1, 'tanh')
        self.l1_bn = nn.BatchNorm1d(self.K+pl)

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

        h = F.softplus((self.l1(inp)))
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
        self.arguments = kwargs['arguments']

        self.l1 = nn.Linear(self.L, self.K, bias=True)
        initializationhelper(self.l1, 'tanh')
        self.l1_bn = nn.BatchNorm1d(self.K)

        #self.l2 = nn.Linear(self.K, self.K, bias=True) 
        #initializationhelper(self.l2, 'relu')
        #self.l2_bn = nn.BatchNorm1d(self.K)

        self.l3 = nn.Linear(self.K, 1, bias=True)
        initializationhelper(self.l3, 'relu') 

    def forward(self, inp):
        #if inp.dim() > 2:
        #    inp = inp.permute(0, 2, 1)
        #inp = inp.contiguous().view(-1, self.L) 

        if not (type(inp) == Variable):
            inp = Variable(inp[0])

        h1 = F.relu((self.l1(inp)))
        
        #h2 = F.tanh(self.l2_bn(self.l2(h1)))

        output = F.sigmoid(self.l3(h1))
        return output, h1

def adversarial_trainer(loader_mix, train_loader, 
                        generator, discriminator, EP=5,
                        **kwargs):
    arguments = kwargs['arguments']
    criterion = kwargs['criterion']
    conditional_gen = kwargs['conditional_gen']
 
    L1, L2 = generator.L1, generator.L2
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
        N = 4 
        
        genspec = out_g.data.numpy().transpose()[:, :700]
        target = tar[0].numpy().transpose()[:, :700]

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

    #lrd.specshow(lr.amplitude_to_db(SPC, ref=np.max),
    #             y_axis='log', x_axis='time')
    # end of drawnow function

    if arguments.optimizer == 'Adam':
        optimizerD = optim.Adam(discriminator.parameters(), lr=arguments.lr, betas=(0.9, 0.999))
        optimizerG = optim.Adam(generator.parameters(), lr=arguments.lr, betas=(0.9, 0.999))
    elif arguments.optimizer == 'RMSprop':
        optimizerD = optim.RMSprop(discriminator.parameters(), lr=arguments.lr)
        optimizerG = optim.RMSprop(generator.parameters(), lr=arguments.lr)
    else:
        raise ValueError('Whaaaat?')

    if not arguments.cuda:
        figure(figsize=(4,4))
    true, false = 1, 0
    for ep in range(EP):
        for (ft, tar, lens), mix in zip(train_loader, loader_mix):
            if arguments.cuda:
                ft = ft.cuda()
                tar = tar.cuda()
                lens = lens.cuda()
                #mix = mix.cuda()

            # sort the tensors within batch
            ft, tar = ut.sort_pack_tensors(ft, tar, lens)

            # discriminator gradient with real data
            discriminator.zero_grad()
            out_d, _ = discriminator.forward(tar)
            labels = Variable(torch.ones(out_d.size(0))*true).squeeze().float()
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
            out_d_g, _ = discriminator.forward(out_g)
            labels = Variable(torch.ones(out_d.size(0))*false).squeeze().float()
            if arguments.cuda:
                labels = labels.cuda()
            err_D_fake = criterion(out_d_g, labels) 
            err_D_fake.backward(retain_variables=True)

            err_D = err_D_real + err_D_fake
            optimizerD.step()


            # show the current generated output
            if not arguments.cuda:
                if arguments.task == 'atomic_sourcesep':
                    drawnow(drawgendata_atomic)
                else:
                    drawnow(drawgendata)

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
        target = tar[0].numpy().transpose()[:, :200]
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

    if not arguments.cuda:
        figure(figsize=(4,4))
    true, false = 1, 0
    for ep in range(EP):
        for (ft, tar, lens), mix in zip(train_loader, loader_mix):
            if arguments.cuda:
                tar = tar.cuda()
                ft = ft.cuda()
                lens = lens.cuda()

            # sort the tensors within batch
            ft, tar = ut.sort_pack_tensors(ft, tar, lens)

            #if conditional_gen: 
            #    inp = mix.contiguous().view(-1, L)
            #else:
            #    inp = ft_rshape   # fixed_noise.contiguous().view(-1, L)

            # generator gradient
            generator.zero_grad()
            out_g = generator.forward(ft)
            err_G = criterion(out_g, Variable(tar[0]))
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
    arguments = kwargs['arguments']
    optimizer = arguments.optimizer
    loss = kwargs['loss']
    L1 = generator1.L1
    L2 = generator1.L2

    s1s, s2s = [], [] 
    s1hats, s2hats = [], []
    mixes = []
    all_mags1, all_mags2 = [], []
    for i, (MSabs, MSphase, wavfls1, wavfls2, lens1, lens2) in enumerate(islice(loader_mix, 0, 1, 1)): 
        if arguments.cuda:
            MSabs = MSabs.cuda()

        print('Processing source ',i)
        Nmix = MSabs.size(0)
        T = MSabs.size(1)
        MSabs = MSabs.contiguous().view(-1, L2) 

        x1, x2 = torch.rand(Nmix*T, L1), torch.rand(Nmix*T, L1)

        if arguments.cuda:
            x1 = Variable(torch.rand(Nmix*T, L1).cuda(), requires_grad=True)
            x2 = Variable(torch.rand(Nmix*T, L1).cuda(), requires_grad=True)
        else:
            x1 = Variable(torch.rand(Nmix*T, L1), requires_grad=True)
            x2 = Variable(torch.rand(Nmix*T, L1), requires_grad=True)

        if optimizer == 'Adam':
            optimizer_sourcesep = optim.Adam([x1, x2], lr=1e-3, betas=(0.5, 0.999))
        elif optimizer == 'RMSprop':
            optimizer_sourcesep = optim.RMSprop([x1, x2], lr=1e-3)
        else:
            raise ValueError('Whaaaaaaaat')


        for ep in range(EP):
           
            mix_sum = generator1.forward(x1) + generator2.forward(x2) 
            if loss == 'Euclidean': 
                err = torch.mean((Variable(MSabs) - mix_sum)**2)
            elif loss == 'Poisson':
                eps = 1e-20
                err = torch.mean(-Variable(MSabs)*torch.log(mix_sum+eps) + mix_sum)

            err.backward()

            optimizer_sourcesep.step()

            x1.grad.data.zero_()
            x2.grad.data.zero_()

            print('Step in batch [{:d}\{:d}]'.format(ep+1, EP))
            print('The error is ', err)
        
        temp1 = generator1.forward(x1).data.cpu().numpy() 
        temp1_audio, mags1 = ut.mag2spec_and_audio(temp1, MSphase)
        all_mags1.extend(mags1)
        s1hats.extend(temp1_audio)

        temp2 = generator2.forward(x2).data.cpu().numpy() 
        temp2_audio, mags2 = ut.mag2spec_and_audio(temp2, MSphase)
        all_mags2.extend(mags2)
        s2hats.extend(temp2_audio)

        s1s.extend(np.split(wavfls1.cpu().numpy(), Nmix, 0))
        s2s.extend(np.split(wavfls2.cpu().numpy(), Nmix, 0))

    curdir = os.getcwd() 
    exp_info = '_'.join([arguments.tr_method, arguments.data, 
                         arguments.input_type, arguments.optimizer, 
                         'feat_match', str(arguments.feat_match)])
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

        plt.subplot(3, 2, 1)
        lrd.specshow(np.abs(lr.stft(s1+s2, n_fft=1024)), y_axis='log', x_axis='time')
        plt.title('Observed Mixture')

        plt.subplot(3, 2, 3)
        lrd.specshow(np.abs(lr.stft(s1, n_fft=1024)), y_axis='log', x_axis='time')
        plt.title('Source 1')

        plt.subplot(3, 2, 5) 
        lrd.specshow(np.abs(lr.stft(s2, n_fft=1024)), y_axis='log', x_axis='time')
        plt.title('Source 2')

        plt.subplot(3, 2, 2)
        lrd.specshow(mag1+mag2, y_axis='log', x_axis='time')
        plt.title('Reconstruction')

        plt.subplot(3, 2, 4)
        lrd.specshow(mag1, y_axis='log', x_axis='time')
        plt.title('Sourcehat 1')

        plt.subplot(3, 2, 6) 
        lrd.specshow(mag2,  y_axis='log', x_axis='time')
        plt.title('Sourcehat 2')
        
        if arguments.save_files:
            figname = 'spectrograms_{}'.format(i)
            plt.savefig(os.path.join(magpath, figname)) 

            # save file 1
            filename1 = 'source1hat_{}.wav'.format(i)
            filepath1 = os.path.join(soundpath, filename1)  

            # save file 2
            filename2 = 'source2hat_{}.wav'.format(i)
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


