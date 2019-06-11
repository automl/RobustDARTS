import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import json
import codecs
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from copy import deepcopy
from collections import OrderedDict
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from torch.optim.lr_scheduler import CosineAnnealingLR
from genotypes import PRIMITIVES, primitives_dict

from analyze import Analyzer
#import hessianflow as hf
import numpy.linalg as LA


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/home/zelaa/NIPS19/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--report_freq_hessian', type=float, default=50, help='report frequency hessian')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--nodes', type=int, default=4, help='number of intermediate nodes per cell')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--debug', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--resume', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--job_id', type=int, default=1, help='SLURM_ARRAY_JOB_ID number')
parser.add_argument('--task_id', type=int, default=1, help='SLURM_ARRAY_TASK_ID number')

parser.add_argument('--regularize', action='store_true', default=False,
                    help='use co and droppath')
parser.add_argument('--space', type=str, default='S1', help='space index')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
args = parser.parse_args()

args.save = '../search/{}/{}/{}-{}-{}-{}'.format(args.space,
                                                 args.dataset,
                                                 args.unrolled,
                                                 args.drop_path_prob,
                                                 args.weight_decay,
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

def main(primitives, iteration, final_genotype=None):
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

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion,
                  primitives, steps=args.nodes)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  if args.dataset == 'cifar10':
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  elif args.dataset == 'cifar100':
    train_transform, valid_transform = utils._data_transforms_cifar100(args)
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  elif args.dataset == 'svhn':
    train_transform, valid_transform = utils._data_transforms_svhn(args)
    train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)

  architect = Architect(model, args)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  analyser = Analyzer(args, model)

  if args.resume:
    if args.regularize:
      model.drop_path_prob = args.drop_path_prob
      train_transform.transforms[-1].cutout_prob = args.cutout_prob
    filename = os.path.join(args.save, 'checkpoint_{}.pth.tar'.format(args.task_id))
    if os.path.isfile(filename):
      logging.info("=> loading checkpoint '{}'".format(filename))
      checkpoint = torch.load(filename)
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      model.alphas_normal.data, model.alphas_reduce.data = checkpoint['arch_parameters'][0].clone().data, checkpoint['arch_parameters'][1].clone().data

    input_train, target_train = next(iter(train_queue))
    input_train = Variable(input_train, requires_grad=False).cuda()
    target_train = Variable(target_train, requires_grad=False).cuda(async=True)

    input_valid, target_valid = next(iter(valid_queue))
    input_valid = Variable(input_valid, requires_grad=False).cuda()
    target_valid = Variable(target_valid, requires_grad=False).cuda(async=True)

    model.train()

    H = analyser.compute_Hw(input_train, target_train, input_valid,
                            target_valid,
                            args.learning_rate_min, optimizer, False)
    g = analyser.compute_dw(input_train, target_train, input_valid,
                            target_valid,
                            args.learning_rate_min, optimizer, False)
    g = torch.cat([x.view(-1) for x in g])

    _state = {'epoch': 50,
             'H': H.cpu().data.numpy().tolist(),
             'g': g.cpu().data.numpy().tolist()}

    with codecs.open(os.path.join(args.save,
                                  'derivatives_final_{}.json'.format(args.task_id)),
                                  'a', encoding='utf-8') as file:
        json.dump(_state, file, separators=(',', ':'))
        file.write('\n')

    return analyser

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    if args.regularize:
      model.drop_path_prob = args.drop_path_prob * epoch / (args.epochs - 1)
      train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
      logging.info('epoch %d lr %e drop_prob %e cutout_prob %e', epoch, lr,
                    model.drop_path_prob,
                    train_transform.transforms[-1].cutout_prob)
    else:
      logging.info('epoch %d lr %e', epoch, lr)

    # training
    train_acc, train_obj = train(epoch, primitives, train_queue,
                                 valid_queue, model, architect, criterion,
                                 optimizer, lr, analyser)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    genotype, _ = model.genotype()

    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    with open(os.path.join(args.save, 'ckpt_{}.txt'.format(args.task_id)), 'a') as file:
      file.write(str(valid_acc))
      file.write('\n')

    #state = {'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'arch_parameters': model.arch_parameters(),
    #        }

    #utils.save_checkpoint(state, False, args.save, args.task_id)

  with open('genotypes.py', 'a') as file:
    file.write('DARTS_%s_%s_%d_%d = %s'%(args.space,
                                         args.dataset,
                                         args.job_id,
                                         args.task_id,
                                         genotype))
    file.write('\n')

  return analyser


def train(epoch, primitives, train_queue, valid_queue, model, architect,
          criterion, optimizer, lr, analyser):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    if architect is not None:
      # get a random minibatch from the search queue with replacement
      input_search, target_search = next(iter(valid_queue))
      input_search = Variable(input_search, requires_grad=False).cuda()
      target_search = Variable(target_search, requires_grad=False).cuda(async=True)

      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      if args.debug:
        break

  if (epoch % args.report_freq_hessian == 0) or (epoch == (args.epochs - 1)):
    _data_loader = deepcopy(train_queue)
    input, target = next(iter(_data_loader))

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get gradient information
    param_grads = [p.grad for p in model.parameters() if p.grad is not None]
    param_grads = torch.cat([x.view(-1) for x in param_grads])
    param_grads = param_grads.cpu().data.numpy()
    grad_norm = LA.norm(param_grads)

    #gradient_vector = torch.cat([x.view(-1) for x in gradient_vector]) 
    #grad_norm = LA.norm(gradient_vector.cpu())
    logging.info('\nCurrent grad norm based on Train Dataset: %.4f',
                 grad_norm)

    H = analyser.compute_Hw(input, target, input_search, target_search,
                            lr, optimizer, False)
    g = analyser.compute_dw(input, target, input_search, target_search,
                            lr, optimizer, False)
    g = torch.cat([x.view(-1) for x in g])

    #del _data_loader

    state = {'epoch': epoch,
             'H': H.cpu().data.numpy().tolist(),
             'g': g.cpu().data.numpy().tolist(),
             'g_train': float(grad_norm),
             #'eig_train': eigenvalue,
            }

    with codecs.open(os.path.join(args.save,
                                  'derivatives_{}.json'.format(args.task_id)),
                                  'a', encoding='utf-8') as file:
        json.dump(state, file, separators=(',', ':'))
        file.write('\n')

    #print(analyser.compute_eigenvalues())

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      if args.debug:
        break

  return top1.avg, objs.avg


if __name__ == '__main__':
  _primitives = primitives_dict[args.space]
  analyser = main(_primitives, 1, None)

