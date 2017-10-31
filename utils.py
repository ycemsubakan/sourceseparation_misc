import torch
import numpy as np
import pdb
import torch.nn.utils.rnn as rnn_utils 
import librosa as lr
import torch.utils.data as data_utils
import os
import torch.nn.init as torchinit
import mir_eval.separation as mevalsep 
import pandas as pd
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import timit_utilities as tu
import scipy as sp
import sklearn as skt
import itertools as it

def append_dirs(directories):
    all_dirs = [ ''.join(dr) for dr in directories]
    all_dirs_str = '_' + ''.join(all_dirs) + '_'
    return all_dirs_str 

def list_timit_dirs():
    home = os.path.expanduser('~')
    p = os.path.join(home, 'Dropbox', 'RNNs', 'timit', 'timit-wav', 'train') 

    directories = os.listdir(p)
    possible_dirs = []
    for dr in directories:
        path = os.path.join(p, dr)

        males = [name for name in os.listdir(path) if name[0] == 'm']
        females = [name for name in os.listdir(path) if name[0] == 'f']

        possible_dirs = it.chain(possible_dirs, it.product([dr], males, females))

    return possible_dirs
        

def prepare_mixture_gm_data(arguments):
    dataset = []
    
    num_means = 1
    arguments.L2 = 2
    arguments.L1 = 2
    arguments.K = 50
    sig0 = 5
    sig = 0.1

    means = 5*torch.randn(num_means, arguments.L2) 
    arguments.means = means.numpy()

    N = 2000

    mixinds = torch.multinomial(torch.ones(num_means), N, replacement=True) 
    obsnoise = torch.randn(N, arguments.L2) 

    data = means[mixinds] + obsnoise
    inp = torch.randn(N, arguments.L1) 

    dataset1 = TensorDataset(inp, data, [1]*N)
    datasetmix = dataset1 

    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader1 = data_utils.DataLoader(dataset1, batch_size=arguments.batch_size, shuffle=False, **kwargs)
    loader_mix = data_utils.DataLoader(datasetmix, batch_size=arguments.batch_size, shuffle=False, **kwargs)

    return loader1, loader_mix
   

def save_image_samples(samples, save_path, exp_info, mode, arguments):
    N = len(samples)
    sqrtN = int(np.sqrt(N))
    for n, sample in enumerate(samples):
        plt.subplot(sqrtN, sqrtN, n+1)
        sample = sample.contiguous().view(arguments.nfts, arguments.T)
        plt.imshow(sample.cpu().numpy(), cmap='binary')
        plt.clim(0,1)

    plt.savefig(os.path.join(save_path, exp_info + '_' + mode +'.png'))

def save_models(generators, discriminators, exp_info, save_folder, arguments):
    
    for n, generator in enumerate(generators):
        torch.save(generator.state_dict(), os.path.join(save_folder, 'generator' + str(n) + '_'  + exp_info + '.trc'))    

    for n, discriminator in enumerate(discriminators):
        torch.save(discriminator.state_dict(), os.path.join(save_folder, 'discriminator' + str(n) + '_' + exp_info + '.trc'))    


def compile_bssevals(bss_evals): 
    sdrs1, sdrs2 = [], []
    sirs1, sirs2 = [], []
    sars1, sars2 = [], []

    for i, (sdr, sir, sar, _) in enumerate(bss_evals):
        sdrs1.append(sdr[0]), sirs1.append(sir[0]), sars1.append(sar[0])
        sdrs2.append(sdr[1]), sirs2.append(sir[1]), sars2.append(sar[1])  

    df = pd.DataFrame({'sdr1': sdrs1, 'sdr2': sdrs2, 
                       'sar1': sars1, 'sar2': sars2,
                       'sir1': sirs1, 'sir2': sirs2})
    return df

def audio_to_bsseval(s1hats, s2hats, s1s, s2s):
    bss_evals = []
    bss_evals_paris = []
    for i, (s1hat, s2hat, s1, s2) in enumerate(zip(s1hats, s2hats, s1s, s2s)):

        print('Computing bssevals for mixture {}'.format(i))

        sourcehat_mat = np.concatenate([s1hat.reshape(1, -1), s2hat.reshape(1, -1)], 0)
        source_mat = np.concatenate([s1.reshape(1, -1), s2.reshape(1, -1)], 0)

        Nhat, N = sourcehat_mat.shape[1], source_mat.shape[1]
        Nmin = min([N, Nhat])

        bss_evals.append(mevalsep.bss_eval_sources(source_mat[:, :Nmin], 
                                                   sourcehat_mat[:, :Nmin]))
        bss_evals_paris.append([tu.bss_eval(sourcehat_mat[0, :Nmin], 0, 
                                            source_mat[:, :Nmin]), 
                                tu.bss_eval(sourcehat_mat[1, :Nmin], 1,
                                            source_mat[:, :Nmin])])
        print(bss_evals)
        print(bss_evals_paris) 


    return bss_evals

def mag2spec_and_audio_wiener(xhat, recons, MS, MSphase, arguments):

    #xhat = xhat.cpu().numpy()
    #recons = recons.cpu().numpy()
    try:   # pytorch case
        MS = MS.cpu().numpy()
        MSphase = MSphase.cpu().numpy()
        Nmix = MSphase.shape[0]

        maghats = np.split(xhat, Nmix, axis=0) 
        reconss = np.split(recons, Nmix, axis=0) 
        mixmags = np.split(MS, Nmix, axis=0) 
        phases = np.split(MSphase, Nmix, axis=0)

    except:
        maghats = [xhat]
        reconss = [recons]
        mixmags = [MS]
        phases = [MSphase]

   
    all_audio = []
    eps = 1e-20
    for maghat, recons, mixmag, phase in zip(maghats, reconss, mixmags, phases):
        mask = (maghat / (recons + eps))
        all_audio.append(lr.istft((mask*mixmag*np.exp(1j*phase)).transpose(), 
                                  win_length=arguments.win_length))

    return all_audio, maghats


def mag2spec_and_audio(xhat, MSphase, arguments):

    MSphase = MSphase.cpu().numpy()
    Nmix = MSphase.shape[0]
    mags = np.split(xhat, Nmix, axis=0) 
    phases = np.split(MSphase, Nmix, axis=0)

    all_audio = []
    for mag, phase in zip(mags, phases):
        all_audio.append(lr.istft((mag*np.exp(1j*phase.squeeze())).transpose(), 
                                  win_length=arguments.win_length))

    return all_audio, mags

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
        
    dataset1 = TensorDataset(data_tensor=inp1,
                                        target_tensor=dataset1,
                                        lens=[1]*Nmix)
    dataset2 = data_utils.TensorDataset(data_tensor=inp2,
                                        target_tensor=dataset2)
    dataset_mix = data_utils.TensorDataset(data_tensor=dataset_mix,
                                        target_tensor=torch.ones(Nmix))

    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader1 = data_utils.DataLoader(dataset1, batch_size=arguments.batch_size, shuffle=False, **kwargs)
    loader2 = data_utils.DataLoader(dataset2, batch_size=arguments.batch_size, shuffle=False, **kwargs)
    loader_mix = data_utils.DataLoader(dataset_mix, batch_size=arguments.batch_size, shuffle=False, **kwargs)

    return loader1, loader2, loader_mix

def get_loaders(loader_batchsize, **kwargs):
    arguments=kwargs['arguments']
    data = arguments.data

    if data == 'mnist':
        kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               #transforms.Normalize((0,), (1,))
                           ])),
            batch_size=loader_batchsize, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               #transforms.Normalize((7,), (0.3081,))
                           ])),
            batch_size=loader_batchsize, shuffle=True, **kwargs)

    return train_loader, test_loader

def sort_pack_tensors(ft, tar, lens):
    _, inds = torch.sort(lens, dim=0, descending=True)
    ft, tar, lens = ft[inds], tar[inds], list(lens[inds])

    ft_packed = rnn_utils.pack_padded_sequence(ft, lens, batch_first=True)
    tar_packed = rnn_utils.pack_padded_sequence(tar, lens, batch_first=True)
    return ft_packed, tar_packed

def do_pca(X, K):
    L = X.shape[0]

    X_mean = X.mean(1) 
    X_zeromean = X - X_mean.reshape(L, 1)

    U, S, V = sp.linalg.svd(X_zeromean) 

    U = U[:, :K] 
    X_Kdim = np.dot(U.transpose(), X_zeromean) 

def dim_red(X, K, mode):
    if mode == 'isomap':
        X_low = skt.manifold.Isomap(5, K).fit_transform(X) 
    elif mode == 'mds':
        X_low = skt.manifold.MDS(K).fit_transform(X)
    elif mode == 'tsne':
        X_low = skt.manifold.TSNE(K, init='pca', random_state=0).fit_transform(X)

    #plt.plot(X_low[:, 0], X_low[:, 1], 'o')
    #plt.show()
    return X_low

def preprocess_timit_files(arguments, dr=None):

    L, T, step = 150, 200, 50  

    #random.seed( s)
    #we pick the set according to trial number 
    Z_temp = tu.sound_set(3, dr = dr) 
    Z = Z_temp[0:4]
    mf = Z_temp[4]
    ff = Z_temp[5]


    # Front-end details
    #if hp is None:
    sz = 1024       
    win_length = sz

    #source 1
    #M1paris, P1 = FE.fe( Z[0] )
    S1 = lr.stft( Z[0], n_fft=sz, win_length=win_length).transpose()
    M1, P1 = np.abs(S1), np.angle(S1) 
    M1, P1, lens1 = [M1], [P1], [M1.shape[0]]

    #dim_red(M1[0], 2, 'tsne') 

    # source 2
    S2 = lr.stft( Z[1], n_fft=sz, win_length=win_length).transpose()
    M2, P2 = np.abs(S2), np.angle(S2) 
    M2, P2, lens2 = [M2], [P2], [M2.shape[0]] 

    #dim_red(M2[0], 2, 'tsne') 


    #mixtures
    M = lr.stft( Z[2]+Z[3], n_fft=sz, win_length=win_length).transpose()
    M_t, P_t = np.abs(M), np.angle(M) 
    M_t, P_t, lens_t = [M_t], [P_t], [M_t.shape[0]]

    M_t1 = [np.abs(lr.stft( Z[2], n_fft=sz, win_length=win_length).transpose())]
    M_t2 = [np.abs(lr.stft( Z[3], n_fft=sz, win_length=win_length).transpose())]

    arguments.n_fft = sz
    arguments.L2 = M.shape[1]
    #arguments.K = 100
    arguments.smooth_output = False
    arguments.dataname = '_'.join([mf, ff])
    arguments.win_length = win_length
    arguments.fs = 16000

    T = 200

    if arguments.plot_training:
        plt.subplot(211)
        lr.display.specshow(M1[0][:T].transpose(), y_axis='log') 
        
        plt.subplot(212)
        lr.display.specshow(M2[0][:T].transpose(), y_axis='log') 

        fs = arguments.fs
        lr.output.write_wav('timit_train1_pt.wav', Z[0], fs)
        lr.output.write_wav('timit_train2_pt.wav', Z[1], fs)
        lr.output.write_wav('timit_test1_pt.wav', Z[2], fs)
        lr.output.write_wav('timit_test2_pt.wav', Z[3], fs)

    loader1 = form_torch_audio_dataset(M1, P1, lens1, arguments, 'source') 
    loader2 = form_torch_audio_dataset(M2, P2, lens2, arguments, 'source')
    loadermix = form_torch_mixture_dataset(M_t, P_t, 
                                           M_t1, M_t2,  
                                           [Z[2]], [Z[3]], 
                                           [Z[2].size], [Z[3].size], 
                                           arguments)
    return loader1, loader2, loadermix


def preprocess_audio_files(arguments):
    '''preprocess audio files to form mixtures 
    and training sequences'''

    if arguments.data == 'synthetic_sounds':
        #dataname = 'generated_sounds_43_71_35_43_64_73'
        dataname = 'generated_sounds_20_71_43_51_64_73'
        #dataname = 'generated_sounds_20_71_30_40_64_73'
        #dataname = 'generated_sounds_20_71_64_73_64_73'
        #dataname = 'generated_sounds_20_71_50_60_50_60'
        audio_path = os.getcwd().replace('someplaying_around', 
                                         dataname) 

        arguments.dataname = dataname
        arguments.K = 200

        files = os.listdir(audio_path)
        files_source1 = [fl for fl in files if 'source1' in fl]
        files_source2 = [fl for fl in files if 'source2' in fl]
        #files_mixture = [fl for fl in files if 'mixture' in fl]
    elif arguments.data == 'spoken_digits':
        digit1, digit2 = 0, 1

        path = os.getcwd().replace('someplaying_around', 'free-spoken-digit-dataset') 
        audio_path = os.path.join(path, 'recordings')
        arguments.K = 200

        files = os.listdir(audio_path)
        files_source1 = [fl for fl in files if str(digit1)+'_' in fl]
        files_source2 = [fl for fl in files if str(digit2)+'_' in fl]

        N1, N2 = len(files_source1), len(files_source2)
        N = min([N1, N2])
        files_source1, files_source2 = files_source1[:N], files_source2[:N]
    else:
        raise ValueError('Whaaat?')

    n_fft = 1024
    win_length = 1024
    arguments.n_fft, arguments.win_length = n_fft, win_length
    # first load the files and append zeros
    SPCS1abs, SPCS2abs, MSabs = [], [], []
    SPCS1phase, SPCS2phase, MSphase = [], [], [] 
    wavfls1, wavfls2 = [], []
    lens1, lens2 = [], []
    for i, (fl1, fl2) in enumerate(zip(files_source1, files_source2)):
        wavfl1, fs = lr.load(os.path.join(audio_path, fl1))
        wavfl2, fs = lr.load(os.path.join(audio_path, fl2))
        wavfls1.append(wavfl1), wavfls2.append(wavfl2)

        SPC1 = lr.core.stft(wavfl1, n_fft=n_fft, win_length=win_length).transpose()
        SPC2 = lr.core.stft(wavfl2, n_fft=n_fft, win_length=win_length).transpose()

        #lens1.append(SPC1.shape[1]), lens2.append(SPC2.shape[1])
        form_np_audio_list(SPC1, SPCS1abs, SPCS1phase)
        form_np_audio_list(SPC2, SPCS2abs, SPCS2phase)
        print(i)

    arguments.fs = fs  # add the sampling rate here
    wavfls1, wavfls2, mixes, wavlens1, wavlens2 = append_zeros_all(wavfls1, wavfls2, mode='audio')
    SPCS1abs, SPCS2abs, SPCmixes, lens1, lens2 = append_zeros_all(SPCS1abs, SPCS2abs, mode='specs')

    # then compute the spectrograms 
    for i, mixes in enumerate(mixes):
        M = lr.core.stft(mixes, n_fft=n_fft, win_length=win_length).transpose()
        form_np_audio_list(M, MSabs, MSphase) 

    if arguments.task == 'spoken_digits':
        arguments.L2 = SPC1.shape[0]*SPC1.shape[1]
        arguments.nfts = SPC1.shape[0]
        arguments.T = SPC1.shape[1]
    elif arguments.task == 'atomic_sourcesep':
        arguments.L2 = SPC1.shape[1]

    loader1 = form_torch_audio_dataset(SPCS1abs, SPCS1phase, lens1, arguments, 'source') 
    loader2 = form_torch_audio_dataset(SPCS2abs, SPCS2phase, lens2, arguments, 'source')
    loadermix = form_torch_mixture_dataset(MSabs, MSphase, 
                                           SPCS1abs, SPCS2abs,  
                                           wavfls1, wavfls2, 
                                           wavlens1, wavlens2, 
                                           arguments)

    return loader1, loader2, loadermix

def form_np_audio_list(SPC, SPCSabs, SPCSphase):
    SPCSabs.append((np.abs(SPC)))
    SPCSphase.append((np.angle(SPC)))


def form_torch_audio_dataset(SPCSabs, SPCSphase, lens, arguments, loadertype):
    
    SPCSabs = torch.from_numpy(np.array(SPCSabs))
    if loadertype == 'mixture':
        SPCSphase = torch.from_numpy(np.array(SPCSphase))
        dataset = TensorDataset(data_tensor=SPCSabs,
                                target_tensor=SPCSphase,
                                lens=lens)
    elif loadertype == 'source':
        if arguments.input_type == 'noise':
            if arguments.noise_type == 'gamma': 
                a, b = 1, 10
                b = 1/float(b)
                sz = (SPCSabs.size(0), SPCSabs.size(1), arguments.L1)
                inp_np = np.random.gamma(a, b, sz)
                plt.matshow(inp_np.squeeze().transpose()[:, :50])
                inp = torch.from_numpy(inp_np).float()
            elif arguments.noise_type == 'bernoulli':
                sz = (SPCSabs.size(0), SPCSabs.size(1), arguments.L1)
                mat = (1/float(8))*torch.ones(sz)
                inp = torch.bernoulli(mat) 

                
            elif arguments.noise_type == 'gaussian':
                inp = torch.randn(SPCSabs.size(0), SPCSabs.size(1), arguments.L1)
            else:
                raise ValueError('Whaaaat?')
        elif arguments.input_type == 'autoenc':
            inp = SPCSabs
            arguments.L1 = arguments.L2
        else:
            raise ValueError('Whaaaaaat input_type?')
        dataset = TensorDataset(data_tensor=inp,
                                target_tensor=SPCSabs,
                                lens=lens)
    else:
        raise ValueError('Whaaaat?') 
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader = data_utils.DataLoader(dataset, batch_size=arguments.batch_size, shuffle=True, **kwargs)

    return loader

def form_torch_mixture_dataset(MSabs, MSphase, 
                               SPCS1abs, SPCS2abs,
                               wavfls1, wavfls2, 
                               lens1, lens2, 
                               arguments):

    MSabs = torch.from_numpy(np.array(MSabs))
    MSphase = torch.from_numpy(np.array(MSphase)) 
    SPCS1abs = torch.from_numpy(np.array(SPCS1abs)) 
    SPCS2abs = torch.from_numpy(np.array(SPCS2abs)) 
    wavfls1 = torch.from_numpy(np.array(wavfls1))
    wavfls2 = torch.from_numpy(np.array(wavfls2))
    
    dataset = MixtureDataset(MSabs, MSphase, SPCS1abs, SPCS2abs, 
                             wavfls1, wavfls2, lens1, lens2)

    kwargs = {'num_workers': 1, 'pin_memory': True} if arguments.cuda else {}
    loader = data_utils.DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False, **kwargs)

    return loader


def append_zeros_all(fls1, fls2, mode):
    lens1, lens2 = [], []
    for fl1, fl2 in zip(fls1, fls2):
        if mode == 'audio':
            lens1.append(fl1.shape[0]), lens2.append(fl2.shape[0])
        elif mode == 'specs':
            lens1.append(fl1.shape[0]), lens2.append(fl2.shape[0])
        else:
            raise ValueError('Whaaat?')

    inds1, lens1 = list(np.flip(np.argsort(lens1),0)), np.flip(np.sort(lens1),0)
    inds2, lens2 = list(np.flip(np.argsort(lens2),0)), np.flip(np.sort(lens2),0)
    fls1, fls2 = np.array(fls1)[inds1], np.array(fls2)[inds2]
    maxlen = max([max(lens1), max(lens2)])
    
    mixes = []
    for i, (fl1, fl2) in enumerate(zip(fls1, fls2)):
        if mode == 'audio':
            fls1[i] = np.pad(fl1, (0, maxlen - fl1.shape[0]), 'constant')
            fls2[i] = np.pad(fl2, (0, maxlen - fl2.shape[0]), 'constant')
            mixes.append(fls1[i] + fls2[i])
        elif mode == 'specs':
            fls1[i] = np.pad(fl1, ((0, maxlen - fl1.shape[0]), (0, 0)), 'constant')
            fls2[i] = np.pad(fl2, ((0, maxlen - fl2.shape[0]), (0, 0)), 'constant')
        else:
            raise ValueError('Whaaat?')

    return list(fls1), list(fls2), mixes, lens1, lens2


class TensorDataset(data_utils.Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data_tensor, target_tensor, lens):
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.lens = lens

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.lens[index]

    def __len__(self):
        return self.data_tensor.size(0)

class MixtureDataset(data_utils.Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, MSabs, MSphase, SPCS1abs, SPCS2abs, 
                 wavfls1, wavfls2, lens1, lens2):
        assert MSabs.size(0) == wavfls1.size(0)
        assert wavfls1.size(0) == wavfls2.size(0)

        self.MSabs = MSabs
        self.MSphase = MSphase
        self.SPCS1abs = SPCS1abs
        self.SPCS2abs = SPCS2abs
        self.wavfls1 = wavfls1
        self.wavfls2 = wavfls2
        self.lens1 = lens1
        self.lens2 = lens2

    def __getitem__(self, index):
        return self.MSabs[index], self.MSphase[index], \
               self.SPCS1abs[index], self.SPCS2abs[index], \
               self.wavfls1[index], self.wavfls2[index], \
               self.lens1[index], self.lens2[index]

    def __len__(self):
        return self.MSabs.size(0)


