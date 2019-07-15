import os
import sys
import time
import glob
import yaml
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, '../RobustDARTS')

from src import utils
from src.utils import Genotype
from src.evaluation.model import Network
from src.evaluation.args import Helper


parser = argparse.ArgumentParser("DARTS evaluation")
parser.add_argument('--data', type=str, default='/home/zelaa/NIPS19/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default=None, help='path of saved model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='drop path probability')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--save', type=str, default='evals', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--job_id', type=int, default=1, help='SLURM_ARRAY_JOB_ID number')
parser.add_argument('--task_id', type=int, default=1, help='SLURM_ARRAY_TASK_ID number')
# search options
parser.add_argument('--space', type=str, default='s1',
                    help='space index')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset')
parser.add_argument('--setting', type=str, default='wd',
                    help='regularizer')
parser.add_argument('--setting_value', type=float, default=0.003,
                    help='value of regularizer')
parser.add_argument('--results_file', type=str, default='results.yaml')

args = parser.parse_args()

args.save = '../{}/{}/{}/{}-{}'.format(args.save,
                                       args.space,
                                       args.dataset,
                                       args.setting_value,
                                       args.job_id)
utils.create_exp_dir(args.save)#, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log_{}.txt'.format(args.task_id)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset != 'cifar100':
  CIFAR_CLASSES = 10
else:
  CIFAR_CLASSES = 100

# load configuration file
with open('config.yaml', 'r') as f:
  cfg = yaml.load(f)

configuration = '_'.join([args.space, args.dataset, args.setting])
args.init_channels, args.layers = dict(cfg)[configuration].values()

# load architectures to evaluate
with open('archs.yaml', 'r') as f:
  archs = yaml.load(f)

arch = dict(archs)[configuration][args.setting_value]

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval(arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  if args.model_path is not None:
    utils.load(model, args.model_path, genotype)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  if args.dataset == 'cifar10':
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'cifar100':
    train_transform, valid_transform = utils._data_transforms_cifar100(args)
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
  elif args.dataset == 'svhn':
    train_transform, valid_transform = utils._data_transforms_svhn(args)
    train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
    valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = CosineAnnealingLR(optimizer, float(args.epochs))

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    with open(os.path.join(args.save, 'ckpt_{}.txt'.format(args.task_id)), 'a') as file:
        file.write(str(valid_acc))
        file.write('\n')

    #utils.save(model, os.path.join(args.save, 'weights_{}.pt'.format(args.task_id)))
  try:
    with open(args.results_file, 'r') as f:
      result = yaml.load(f)
    if configuration in result.keys():
      result[configuration].update({args.setting_value: valid_acc})
    else:
      result.update({configuration: {args.setting_value: valid_acc}})
    with open(args.results_file, 'w') as f:
      yaml.dump(result, f, default_flow_style=False)
  except:
    result = {
        configuration: {
            args.setting_value: valid_acc
        }
    }
    with open(args.results_file, 'w') as f:
      yaml.dump(result, f, default_flow_style=False)


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.cuda()
    target = target.cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = input.cuda()
      target = target.cuda(async=True)

      logits, _ = model(input)
      loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

