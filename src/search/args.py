import os
import yaml
import argparse
import numpy as np
import torch.utils
import torchvision.datasets as dset

from copy import copy
from src import utils

class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser("RobustDARTS")

        # general options
        parser.add_argument('--data',                    type=str,            default='./data',       help='location of the data corpus')
        parser.add_argument('--space',                   type=str,            default='s1',           help='space index')
        parser.add_argument('--dataset',                 type=str,            default='cifar10',      help='dataset')
        parser.add_argument('--gpu',                     type=int,            default=0,              help='gpu device id')
        parser.add_argument('--model_path',              type=str,            default='saved_models', help='path to save the model')
        parser.add_argument('--seed',                    type=int,            default=2,              help='random seed')
        parser.add_argument('--resume',                  action='store_true', default=False,          help='resume search')
        parser.add_argument('--debug',                   action='store_true', default=False,          help='use one-step unrolled validation loss')
        parser.add_argument('--job_id',                  type=int,            default=1,              help='SLURM_ARRAY_JOB_ID number')
        parser.add_argument('--task_id',                 type=int,            default=1,              help='SLURM_ARRAY_TASK_ID number')

        # training options
        parser.add_argument('--epochs',                  type=int,            default=50,             help='num of training epochs')
        parser.add_argument('--batch_size',              type=int,            default=64,             help='batch size')
        parser.add_argument('--learning_rate',           type=float,          default=0.025,          help='init learning rate')
        parser.add_argument('--learning_rate_min',       type=float,          default=0.001,          help='min learning rate')
        parser.add_argument('--momentum',                type=float,          default=0.9,            help='momentum')
        parser.add_argument('--weight_decay',            type=float,          default=3e-4,           help='weight decay')
        parser.add_argument('--grad_clip',               type=float,          default=5,              help='gradient clipping')
        parser.add_argument('--train_portion',           type=float,          default=0.5,            help='portion of training data')
        parser.add_argument('--arch_learning_rate',      type=float,          default=3e-4,           help='learning rate for arch encoding')
        parser.add_argument('--arch_weight_decay',       type=float,          default=1e-3,           help='weight decay for arch encoding')
        parser.add_argument('--unrolled',                action='store_true', default=False,          help='use one-step unrolled validation loss')

        # one-shot model options
        parser.add_argument('--init_channels',           type=int,            default=16,             help='num of init channels')
        parser.add_argument('--layers',                  type=int,            default=8,              help='total number of layers')
        parser.add_argument('--nodes',                   type=int,            default=4,              help='number of intermediate nodes per cell')

        # augmentation options
        parser.add_argument('--cutout',                  action='store_true', default=False,          help='use cutout')
        parser.add_argument('--cutout_length',           type=int,            default=16,             help='cutout length')
        parser.add_argument('--cutout_prob',             type=float,          default=1.0,            help='cutout probability')
        parser.add_argument('--drop_path_prob',          type=float,          default=0.2,            help='drop path probability')

        # logging options
        parser.add_argument('--save',                    type=str,            default='experiments/search_logs',  help='experiment name')
        parser.add_argument('--results_file_arch',       type=str,            default='results_arch', help='filename where to write architectures')
        parser.add_argument('--results_file_perf',       type=str,            default='results_perf', help='filename where to write val errors')
        parser.add_argument('--report_freq',             type=float,          default=50,             help='report frequency')
        parser.add_argument('--report_freq_hessian',     type=float,          default=50,             help='report frequency hessian')

        # early stopping
        parser.add_argument('--early_stop',              type=int,            default=0,              choices=[0, 1, 2, 3],
                            help='early stop DARTS based on dominant eigenvalue. 0: no 1: yes 2: simulate 3: adaptive regularization')
        parser.add_argument('--window',                  type=int,            default=5,              help='window size of the local average')
        parser.add_argument('--es_start_epoch',          type=int,            default=10,             help='when to start considering early stopping')
        parser.add_argument('--delta',                   type=int,            default=4,              help='number of previous local averages to consider in early stopping')
        parser.add_argument('--factor',                  type=float,          default=1.3,            help='early stopping factor')
        parser.add_argument('--extra_rollback_epochs',   type=int,            default=0,              help='number of extra rollback epochs when deciding to increse regularization')
        parser.add_argument('--compute_hessian',         action='store_false',default=True,           help='compute or not Hessian')
        parser.add_argument('--max_weight_decay',        type=float,          default=243e-4,         help='maximum weight decay')
        parser.add_argument('--mul_factor',              type=float,          default=3.0,            help='multiplication factor')

        # randomNAS
        parser.add_argument('--eval_only',               action='store_true', default=False,          help='eval only')
        parser.add_argument('--randomnas_rounds',        type=int,            default=None,           help='number of evaluation rounds in RandomNAS')
        parser.add_argument('--n_samples',               type=int,            default=1000,           help='number of discrete architectures to sample during eval')

        self.args = parser.parse_args()
        utils.print_args(self.args)


class Helper(Parser):
    def __init__(self):
        super(Helper, self).__init__()

        self.args._save = copy(self.args.save)
        self.args.save = '{}/{}/{}/{}_{}-{}'.format(self.args.save,
                                                    self.args.space,
                                                    self.args.dataset,
                                                    self.args.drop_path_prob,
                                                    self.args.weight_decay,
                                                    self.args.job_id)

        utils.create_exp_dir(self.args.save)

        config_filename = os.path.join(self.args._save, 'config.yaml')
        if not os.path.exists(config_filename):
            with open(config_filename, 'w') as f:
                yaml.dump(self.args_to_log, f, default_flow_style=False)

        if self.args.dataset != 'cifar100':
            self.args.n_classes = 10
        else:
            self.args.n_classes = 100

        # set cutout to False if the drop_prob is 0
        if self.args.drop_path_prob == 0:
            self.args.cutout = False

    @property
    def config(self):
        return self.args

    @property
    def args_to_log(self):
        list_of_args = [
            "epochs",
            "batch_size",
            "learning_rate",
            "learning_rate_min",
            "momentum",
            "grad_clip",
            "train_portion",
            "arch_learning_rate",
            "arch_weight_decay",
            "unrolled",
            "init_channels",
            "layers",
            "nodes",
            "cutout_length",
            "report_freq_hessian",
            "early_stop",
            "window",
            "es_start_epoch",
            "delta",
            "factor",
            "extra_rollback_epochs",
            "compute_hessian",
            "mul_factor",
            "max_weight_decay",
        ]

        args_to_log = dict(filter(lambda x: x[0] in list_of_args,
                                  self.args.__dict__.items()))
        return args_to_log

    def get_train_val_loaders(self):
        if self.args.dataset == 'cifar10':
            train_transform, valid_transform = utils._data_transforms_cifar10(self.args)
            train_data = dset.CIFAR10(root=self.args.data, train=True, download=True, transform=train_transform)
        elif self.args.dataset == 'cifar100':
            train_transform, valid_transform = utils._data_transforms_cifar100(self.args)
            train_data = dset.CIFAR100(root=self.args.data, train=True, download=True, transform=train_transform)
        elif self.args.dataset == 'svhn':
            train_transform, valid_transform = utils._data_transforms_svhn(self.args)
            train_data = dset.SVHN(root=self.args.data, split='train', download=True, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(self.args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=self.args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=self.args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=2)

        return train_queue, valid_queue, train_transform, valid_transform
