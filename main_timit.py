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

arguments = parser.parse_args()

arguments.cuda = not arguments.no_cuda and torch.cuda.is_available()

torch.manual_seed(arguments.seed)
if arguments.cuda:
    torch.cuda.manual_seed(arguments.seed)
np.random.seed(arguments.seed)

tr_method = arguments.tr_method
loss = 'Poisson'

#directories = list(ut.list_timit_dirs())
#Ncombs = len(directories)
#inds = np.sort(np.random.choice(Ncombs, size=25, replace=False))
#directories = np.array(directories)[inds]

loader1, loader2, loader_mix = ut.preprocess_timit_files(arguments) 

exp_info = '_'.join([arguments.tr_method,
                     arguments.test_method,
                     arguments.data, 
                     arguments.dataname,
                     arguments.input_type, arguments.optimizer, 
                     'noise_type', arguments.noise_type,
                     'L1', str(arguments.L1),
                     'feat_match', str(arguments.feat_match),
                     'nfft', str(arguments.n_fft)])
arguments.exp_info = exp_info

ngpu = 1
L1 = arguments.L1
L2 = arguments.L2
K = arguments.K
smooth_output = arguments.smooth_output

if arguments.tr_method == 'ML':
    generator1 = netG(ngpu, K=K, L1=L1, L2=L2, arguments=arguments)
    generator2 = netG(ngpu, K=K, L1=L1, L2=L2, arguments=arguments)
elif arguments.tr_method in ['adversarial', 'adversarial_wasserstein']:
    generator1 = netG(ngpu, K=K, L1=L1, L2=L2, arguments=arguments)
    generator2 = netG(ngpu, K=K, L1=L1, L2=L2, arguments=arguments)

    #generator1 = netG_onelayer(ngpu, K=K, L1=L1, L2=L2, arguments=arguments)
    #generator2 = netG_onelayer(ngpu, K=K, L1=L1, L2=L2, arguments=arguments)
else:
    raise ValueError('Whaaaaat?')

discriminator1 = netD(ngpu, K=K, L=L2, arguments=arguments)
discriminator2 = netD(ngpu, K=K, L=L2, arguments=arguments)

if arguments.cuda:
    generator1.cuda()
    generator2.cuda()

    discriminator1.cuda()
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

# test the generative models by reconstructing
for i in range(1,3):
    reconstruct_tester(generators=[generator1, generator2],
                       source_num=i,    
                       loader_mix=loader_mix,
                       EP=arguments.EP_test,
                       arguments=arguments,
                       conditional=False,
                       tr_method=tr_method,
                       loss=loss,
                       exp_info=exp_info)
pdb.set_trace()



# Separate out the sources 

if arguments.adjust_tradeoff: 
    #alpha_range = list(np.linspace(0.01, 4, 10))
    alpha_range = [0] + list(np.logspace(-8, 1, 10, base=2))
else:
    alpha_range = [0]*3

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

# save the bss evals here 
curdir = os.getcwd()
recordspath = os.path.join(curdir, 'records')
if not os.path.exists(recordspath):
    os.mkdir(recordspath)

bss_evals_path = os.path.join(recordspath, '_'.join(['bss_evals_all', exp_info]) + '.pk')
pickle.dump({'alpha_range': alpha_range, 'bss_evals': bss_evals}, 
            open(bss_evals_path, 'w')) 


# python main.py --optimizer RMSprop --tr_method adversarial --input_type noise --save_files 0 --save_records 0 --L1 512 --EP_train 400 --feat_match 1 --data synthetic_sounds --batch_size 20 --test_method optimize --load_model 0 --adjust_tradeoff 1

#python main.py --input_type autoenc --tr_method ML --save_files 1  --optimizer RMSprop --tr_method ML --EP_train 400 --save_records 1 --data synthetic_sounds --feat_match 1 --load_models 0 --adjust_tradeoff 0 --batch_size 20

####
