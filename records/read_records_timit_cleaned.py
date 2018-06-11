import pandas as pd 
import numpy as np
import pickle
import pdb
import os
import matplotlib.pyplot as plt
import seaborn as sns

def merge_pks(string):
    curdir = os.getcwd()
    files = os.listdir(curdir)

    relevant_files = sorted([fl for fl in files if string in fl])
    dfs = [pickle.load(open(fl, 'rb')) for fl in relevant_files]

    merged_dfs = {}
    for df in dfs:
        for key,value in df.items():
            if key == 'bss_evals':
                try:
                    merged_dfs[key].extend(value)
                except:
                    merged_dfs[key] = []
                    merged_dfs[key].extend(value)
        else:
            merged_dfs[key] = value
    return merged_dfs


def pick_max_sdrs(df, mode = 'max', **kwargs):
    '''this function picks the best sdrs for adversarial source separation network'''
    bss_evals = df['bss_evals']

    try:
        bss_evals = [pd.concat(bss_eval['bss_evals']) for bss_eval in bss_evals]
    except:
        bss_evals = [(bss_eval['bss_evals']) for bss_eval in bss_evals]

    try:
        if kwargs['cutshort']:
            bss_evals = [bss_eval.iloc[:3] for bss_eval in bss_evals]
    except:
        pass

    sdrsums = [bss_eval.sdr1.values + bss_eval.sdr2.values for bss_eval in bss_evals]


    
    sdr_argmaxes = [np.argmax(sdrsum) for sdrsum in sdrsums]  
    if mode == 'max':
        bss_eval_maxes = [pd.DataFrame(bss_eval.iloc[sdr_argmax]).transpose() for bss_eval, sdr_argmax in zip(bss_evals, sdr_argmaxes)]
        return pd.concat(bss_eval_maxes)
    else:
        bss_evals = [pd.DataFrame(bss_eval.iloc[kwargs['alpha_col']]).transpose() for bss_eval, sdr_argmax in zip(bss_evals, sdr_argmaxes)]
        return pd.concat(bss_evals)


def compile_results(algorithm_names, dfs):
    
    sdrs, sirs, sars = [], [], []
    for df in dfs:
        sdrs.append(list(0.5*(df.sdr1 + df.sdr2).values))
        sirs.append(list(0.5*(df.sir1 + df.sir2).values))
        sars.append(list(0.5*(df.sar1 + df.sar2).values))

    sdr_df = pd.DataFrame({algo: sdr for algo, sdr in zip(algorithm_names, sdrs)}, columns=algorithm_names)
    sir_df = pd.DataFrame({algo: sir for algo, sir in zip(algorithm_names, sirs)}, columns=algorithm_names)
    sar_df = pd.DataFrame({algo: sar for algo, sar in zip(algorithm_names, sars)}, columns=algorithm_names)

    return sdr_df, sir_df, sar_df

df_advwas_2 = merge_pks('bss_evals_all_notes_testing_multiple_inits_per_alpha_eq0.1_adversarial_wasserstein_optimize_TIMIT_noise_RMSprop_feat_match_0_adjust_tradeoff_0_L1_513_K_100_Kdisc_90_smooth_estimate')

df_vae = merge_pks('bss_evals_all_VAE_optimize_TIMIT_autoenc_RMSprop_feat_match_0_adjust_tradeoff_0_L1_513_K_100_Kdisc_20_smooth_estimate_1_dir')
df_nmf = merge_pks('bss_evals_all_NMF_TIMIT_K_100_dir_start_0_dir_end_25_nfft_1024_1509138126.0.pk')
df_adv_gauss = merge_pks('bss_evals_all_adversarial_optimize_TIMIT_noise_RMSprop_feat_match_0_adjust_tradeoff_1_L1_513_K_100_Kdisc_90_smooth_estimate_1_dir_')
df_advwas_gauss = merge_pks('bss_evals_all_adversarial_wasserstein_optimize_TIMIT_noise_RMSprop_feat_match_0_adjust_tradeoff_1_L1_513_K_100_smooth_estimate_1_dir_start')
df_advwas_autoenc = merge_pks('bss_evals_all_adversarial_wasserstein_optimize_TIMIT_autoenc_RMSprop_feat_match_0_adjust_tradeoff_1_L1_513_smooth_estimate_1_dir_st')
df_ML = merge_pks('bss_evals_all_ML_optimize_TIMIT_autoenc_RMSprop_feat_match_0_adjust_tradeoff_0_L1_513_smooth_estimate_1_dir_start')

mode = 'max'

all_advwas2 = pick_max_sdrs(df_advwas_2, mode='max',alpha_col=10) 

all_vae = pick_max_sdrs(df_vae)
all_bss_nmf = pick_max_sdrs(df_nmf)
all_bss_adv_gauss = pick_max_sdrs(df_adv_gauss, mode='max', alpha_col=4)
all_bss_advwas_gauss = pick_max_sdrs(df_advwas_gauss, mode='max', alpha_col=4, cutshort=0)
all_bss_advwas_autoenc = pick_max_sdrs(df_advwas_autoenc, mode=mode, alpha_col=2)
all_ML = pick_max_sdrs(df_ML)

algos = ['NMF', 'GAN', 'Gaussian \n WGAN', 'AE\n WGAN', 'MLAE', 'VAE']
algo_dfs = [all_bss_nmf, all_bss_adv_gauss, all_advwas2, all_bss_advwas_autoenc, all_ML, all_vae]
sdr_df, sir_df, sar_df = compile_results(algos, algo_dfs)

my_dpi = 96
plt.figure(figsize=(1600/my_dpi, 400/my_dpi), dpi=my_dpi)

print(all_bss_advwas_gauss)
print(all_bss_advwas_autoenc)
print(all_ML)

print(all_advwas2.shape)
print(all_advwas2.median())
print(all_bss_adv_gauss.median(0))
print(all_bss_advwas_gauss.median(0))
print(all_bss_advwas_autoenc.median(0))
print(all_ML.median(0))
print(all_vae.median(0))

sns.set_style("whitegrid")

ax = plt.subplot(131) 
sns.violinplot(data=sdr_df)
plt.title('SDRs')
plt.ylabel('dB')

plt.subplot(132)
sns.violinplot(data=sir_df)
plt.title('SIRs')
plt.ylabel('dB')

plt.subplot(133)
sns.violinplot(data=sar_df)
plt.title('SARs')
plt.ylabel('dB')

home = os.path.expanduser('~')
plt.savefig('results.eps', format='eps', dpi=my_dpi, bbox_inches='tight')


