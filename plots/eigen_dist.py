import numpy as np
import json
import codecs
import os
import fnmatch
import sys
import matplotlib.pyplot as plt
import numpy.linalg as LA
from genotypes import Genotype
import seaborn as sns

def get_moving_average_3(eigens):
    ma = []
    start_index = 0
    for i in range(0, len(eigens)):
        if start_index == 0:
            ma.append(np.mean(eigens[0:2]))
        #elif start_index == 1:
        #    ma.append(np.mean(eigens[0:3]))
        elif start_index == len(eigens) - 1:
            ma.append(np.mean(eigens[-2:]))
        #elif start_index == len(eigens) - 2:
        #    ma.append(np.mean(eigens[-4:]))
        else:
            ma.append(np.mean(eigens[start_index-1:start_index+2]))
        start_index += 1
    return ma

def get_moving_average_5(eigens):
    ma = []
    start_index = 0
    for i in range(0, len(eigens)):
        if start_index == 0:
            ma.append(np.mean(eigens[0:3]))
        elif start_index == 1:
            ma.append(np.mean(eigens[0:4]))
        elif start_index == len(eigens) - 1:
            ma.append(np.mean(eigens[-3:]))
        elif start_index == len(eigens) - 2:
            ma.append(np.mean(eigens[-4:]))
        else:
            ma.append(np.mean(eigens[start_index-2:start_index+3]))
        start_index += 1
    return ma

counter = 0
def stop_criteria(moving_averages, log_file, factor=1.3):
    global counter
    with open(log_file) as f:
        lines = [eval(x[x.find('genotype') + len('genotype = '): -1])
                 for x in f.readlines() if 'genotype' in x]

    for i in range(5, len(moving_averages)-4):
        if moving_averages[i+4]/moving_averages[i] > factor:
            counter += 1
            return 2*i, lines[2*i]
    return 49, None

filename_paths_1 = ['True-0.0-0.0003-*',
                    'True-0.0-0.0009-*',
                    'True-0.0-0.0027-*',
                    'True-0.0-0.0081-*',
                    'True-0.0-0.0243-*',
                    ]

filename_paths_2 = ['True-0.0-0.0003-*',
                    'True-0.2-0.0003-*',
                    'True-0.4-0.0003-*',
                    'True-0.6-0.0003-*',
                    ]

settings = {'wd': filename_paths_1, 'dp': filename_paths_2}

c = ['b', 'r', 'g', 'y', 'c']
wd = [0.0003, 0.0009, 0.0027, 0.0081, 0.0243]
dp = [0.0, 0.2, 0.4, 0.6]
ymax = [0.1, 0.2, 0.3, 0.4, 0.5]

#sns.set()
sns.set_style("whitegrid")
sns.set_style("ticks")
def main(space, reg, dataset):
    script_path = '/home/zelaa/NIPS19/ANALYSIS_HESSIANFLOW_final_pt031/search/S{}/{}/'.format(space, dataset)
    #f, ax = plt.subplots(1, 2, sharex=True, figsize=(14, 6))
    for seed in [1]:
        f, ax = plt.subplots(1, 1, sharex=True, figsize=(4, 4))
        for i, dirs in enumerate(settings[reg]):
            for d in os.listdir(script_path):
                if fnmatch.fnmatch(d, dirs):
                    f = d
            data = [json.loads(x) for x in
                        codecs.open(os.path.join(script_path,
                                                 f+'/derivatives_{}.json'.format(seed)), 'r',
                                    encoding='utf-8').readlines()]
            xticks = [e['epoch'] for e in data]

            spectrum = [LA.eigvals(e['H']) for e in data][-1][:30]
            if reg == 'wd' and eval(reg)[i] == 0.0243 and space == 4 and dataset == 'svhn':
                print('##############')
                continue

            if reg == 'wd':
                #ax.plot(spectrum, c=c[i], label=r'$L_2$'+'=%.4f'%eval(reg)[i])
                if space == 1 and dataset == 'cifar10':
                    sns.distplot(spectrum, color=c[i], hist=False, label=r'$L_2$'+'=%.4f'%eval(reg)[i])
                else:
                    sns.distplot(spectrum, color=c[i], hist=False)
            else:
                #ax.plot(spectrum, c=c[i], label=reg+'=%.4f'%eval(reg)[i])
                if space == 1 and dataset == 'cifar10':
                    sns.distplot(spectrum, color=c[i], hist=False, label=reg+'=%.4f'%eval(reg)[i])
                else:
                    sns.distplot(spectrum, color=c[i], hist=False)
        ax.grid(True)
        #ax.set_ylabel('', fontsize=12)
        #ax.set_xlabel('i-th eigenvalue', fontsize=12)
        ax.set_title('Eigen. distribution: S{} {}'.format(space, dataset), fontsize=12)

        if space == 1 and dataset == 'cifar10':
            ax.legend()

        save_dir='/home/zelaa/NIPS19/ANALYSIS_HESSIANFLOW_final_pt031/plots/plots_eigen_dist'
        os.makedirs(save_dir, exist_ok=True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, reg+'_S{}_{}_dist.pdf'.format(space,
                                                                             dataset)),
                                 figsize=(4,4))
        #plt.show()
    print(counter)

if __name__ == '__main__':
    for r in settings.keys():
        for s in range(1, 5):
            for d in ['cifar10', 'cifar100', 'svhn']:
                main(s, r, d)
