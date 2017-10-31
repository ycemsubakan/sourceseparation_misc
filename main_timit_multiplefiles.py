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
import torch.multiprocessing as _mp
import multiprocessing as mp
import time
from deep_sep_expr_shared import sep_run

#mp = _mp.get_context('spawn')

def get_bssevals_nmf(dr, arguments):
    loader1, loader2, loader_mix = ut.preprocess_timit_files(arguments, dr=dr) 

    return sep_run([loader1, loader2, loader_mix], [arguments.K, arguments.K], 
                   arguments=arguments)

    
def get_bssevals(dr, arguments):
    loader1, loader2, loader_mix = ut.preprocess_timit_files(arguments, dr=dr) 

    exp_info = '_'.join([arguments.tr_method,
                         arguments.test_method,
                         arguments.data, 
                         arguments.dataname,
                         arguments.input_type, 
                         arguments.optimizer, 
                         'feat_match', str(arguments.feat_match),
                         'K', str(arguments.K),
                         'Kdisc', str(arguments.Kdisc),
                         'L1', str(arguments.L1),
                         'smooth_estimate', str(arguments.smooth_source_estimates),
                         'nfft', str(arguments.n_fft)])
    arguments.exp_info = exp_info

    ngpu = 1
    L1 = arguments.L1
    L2 = arguments.L2
    K = arguments.K
    Kdisc = arguments.Kdisc
    smooth_output = arguments.smooth_output
    tr_method = arguments.tr_method
    loss = 'Poisson'
    alpha_range = arguments.alpha_range

    if tr_method == 'VAE':
        generator1 = VAE(ngpu, K=K, L1=L1, L2=L2, arguments=arguments)
        generator2 = VAE(ngpu, K=K, L1=L1, L2=L2, arguments=arguments)
    else:
        generator1 = netG(ngpu, K=K, L1=L1, L2=L2, arguments=arguments)
        generator2 = netG(ngpu, K=K, L1=L1, L2=L2, arguments=arguments)

    discriminator1 = netD(ngpu, K=Kdisc, L=L2, arguments=arguments)
    discriminator2 = netD(ngpu, K=Kdisc, L=L2, arguments=arguments)

    if arguments.cuda:
        generator1.cuda()
        discriminator1.cuda()

        generator2.cuda()
        discriminator2.cuda()

    # Train the generative models for the sources

    if arguments.load_models:
        modelfldr = 'model_parameters'
        generator1.load_state_dict(torch.load(os.path.join(modelfldr, 'generator0_' + exp_info + '.trc')))
        generator2.load_state_dict(torch.load(os.path.join(modelfldr, 'generator1_' + exp_info + '.trc')))

        discriminator1.load_state_dict(torch.load(os.path.join(modelfldr, 'discriminator0_' + exp_info + '.trc')))
        discriminator2.load_state_dict(torch.load(os.path.join(modelfldr, 'discriminator1_' + exp_info + '.trc')))
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

            adversarial_trainer(loader_mix=loader_mix,
                                train_loader=loader2,
                                generator=generator2, 
                                discriminator=discriminator2, 
                                EP=EP,
                                arguments=arguments,
                                criterion=criterion,
                                conditional_gen=False,
                                source_num=2)

            
        elif tr_method == 'adversarial_wasserstein':


            adversarial_wasserstein_trainer(loader_mix=loader_mix,
                                            train_loader=loader1,
                                            generator=generator1, 
                                            discriminator=discriminator1, 
                                            EP=EP,
                                            arguments=arguments,
                                            conditional_gen=False,
                                            source_num=1)

            adversarial_wasserstein_trainer(loader_mix=loader_mix,
                                            train_loader=loader2,
                                            generator=generator2, 
                                            discriminator=discriminator2, 
                                            EP=EP,
                                            arguments=arguments,
                                            conditional_gen=False,
                                            source_num=2)

        elif tr_method == 'VAE':
            
            def loss_function(recon_x, x, mu, logvar):
                eps = 1e-20
                criterion = lambda lam, tar: torch.mean(-tar*torch.log(lam+eps) + lam)

                BCE = criterion(recon_x, x)

                # see Appendix B from VAE paper:
                # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
                # https://arxiv.org/abs/1312.6114
                # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                # Normalise by same number of elements as in reconstruction
                KLD /= x.size(0) * arguments.L2

                return BCE + KLD

            VAE_trainer(loader_mix=loader_mix,
                        train_loader=loader1,
                        generator=generator1, 
                        EP=EP,
                        arguments=arguments,
                        criterion=loss_function,
                        conditional_gen=False)
            VAE_trainer(loader_mix=loader_mix,
                        train_loader=loader2,
                        generator=generator2, 
                        EP=EP,
                        arguments=arguments,
                        criterion=loss_function,
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
                               conditional_gen=False)
            generative_trainer(loader_mix=loader_mix,
                               train_loader=loader2,
                               generator=generator2, 
                               EP=EP,
                               arguments=arguments,
                               criterion=criterion,
                               conditional_gen=False)

        # save models
        savepath = os.path.join(os.getcwd(), 'model_parameters')
        if not os.path.exists(savepath):
            os.mkdir(savepath) 

        ut.save_models([generator1, generator2], [discriminator1, discriminator2], 
                        exp_info, savepath, arguments)

        
    # Separate out the sources 
    bss_evals = []
    for alpha in alpha_range:
        print('The current tradeoff parameter is {}'.format(alpha))
        bss_eval = ML_separate_audio_sources(generators=[generator1, generator2],
                                             discriminators=[discriminator1, discriminator2],
                                             loader_mix=loader_mix,
                                             EP=arguments.EP_test,
                                             arguments=arguments,
                                             conditional=False,
                                             tr_method=tr_method,
                                             loss=loss, alpha=float(alpha),
                                             exp_info=exp_info)
    
        bss_evals.append(bss_eval)
    return bss_evals

# Training settings
parser = argparse.ArgumentParser(description='Source separation experiments with GANs/Autoencoders')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
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
parser.add_argument('--task', type=str, default='atomic_sourcesep', metavar='task',
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
parser.add_argument('--data', type=str, default='TIMIT', help='spoken_digits or synthetic_sounds')
parser.add_argument('--L1', type=int, default=200)
parser.add_argument('--feat_match', type=int, default=0) 
parser.add_argument('--load_models', type=int, default=0)
parser.add_argument('--adjust_tradeoff', type=int, default=0)
parser.add_argument('--plot_training', type=int, default=0)
parser.add_argument('--smooth_source_estimates', type=int, default=0)
parser.add_argument('--wiener_recons', type=int, default=0)
parser.add_argument('--noise_type', type=str, default='gaussian')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--dir_start', type=int, default=0)
parser.add_argument('--dir_end', type=int, default=25)
parser.add_argument('--K', type=int, default=100)
parser.add_argument('--Kdisc', type=int, default=90)
parser.add_argument('--nmf', type=int, default=0)
parser.add_argument('--notes', type=str, default='')

arguments = parser.parse_args()

arguments.cuda = not arguments.no_cuda and torch.cuda.is_available()

torch.manual_seed(arguments.seed)
if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)
np.random.seed(arguments.seed)
timestamp = round(time.time())
arguments.timestamp = timestamp

# set the range for the tradeoff parameter
if arguments.adjust_tradeoff: 
    alpha_range = [0] + list(np.logspace(-8, 0, 10, base=2))
else:
    alpha_range = [0]*10
arguments.alpha_range = alpha_range

directories = list(ut.list_timit_dirs())
Ncombs = len(directories)
inds = np.sort(np.random.choice(Ncombs, size=25, replace=False))
directories = np.array(directories)[inds]

all_directories = ut.append_dirs(directories) 


num_dirs_at_once = 5
all_bss_evals = []
for i, dr in enumerate(directories[arguments.dir_start:arguments.dir_end]):
    print('processing directory {}'.format(i))
    if arguments.nmf:
        bss_evals = get_bssevals_nmf(dr, arguments)
    else:
        bss_evals = get_bssevals(dr, arguments)
   
    all_bss_evals.append({'bss_evals': bss_evals,
                          'dataname': arguments.dataname})

    # only save the bss evals here if we adjust the tradeoff parameter
    exp_info_all = '_'.join(['notes', arguments.notes,
                             arguments.tr_method,
                             arguments.test_method,
                             arguments.data, 
                             arguments.input_type, arguments.optimizer, 
                             'feat_match', str(arguments.feat_match),
                             'adjust_tradeoff', str(arguments.adjust_tradeoff),
                             'L1', str(arguments.L1),
                             'K', str(arguments.K),
                             'Kdisc', str(arguments.Kdisc),
                             'smooth_estimate', str(arguments.smooth_source_estimates),
                             'dir_start', str(arguments.dir_start),
                             'dir_end', str(arguments.dir_end),
                             'nfft', str(arguments.n_fft),
                             str(timestamp)])

    if arguments.nmf:
        exp_info_all = '_'.join(['NMF',
                             arguments.data, 
                             'K', str(arguments.K),
                             'dir_start', str(arguments.dir_start),
                             'dir_end', str(arguments.dir_end),
                             'nfft', str(arguments.n_fft),
                             str(timestamp)])


    curdir = os.getcwd()
    recordspath = os.path.join(curdir, 'records')
    if not os.path.exists(recordspath):
        os.mkdir(recordspath)

    bss_evals_path = os.path.join(recordspath,
                                  '_'.join(['bss_evals_all', exp_info_all]) + '.pk')
    pickle.dump({'alpha_range': alpha_range,
                 'bss_evals': all_bss_evals,
                 'dataname': directories, 
                 'arguments': arguments}, 
                  open(bss_evals_path, 'wb')) 
