import os
import sys
import glob
import numpy as np
import torch
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
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, '../RobustDARTS')

from src import utils
from src.spaces import spaces_dict
from src.search.model_search import Network
from src.search.architect import Architect
from src.search.analyze import Analyzer
from src.search.args import Helper


helper = Helper()
args = helper.config

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log_{}.txt'.format(args.task_id)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main(primitives):
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

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  model = Network(args.init_channels, args.n_classes, args.layers, criterion,
                  primitives, steps=args.nodes)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  architect = Architect(model, args)

  scheduler = CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  analyser = Analyzer(args, model)

  train_queue, valid_queue, train_transform, valid_transform = helper.get_train_val_loaders()

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

  errors_dict = {'train_acc': [], 'train_loss': [], 'valid_acc': [],
                 'valid_loss': []}

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

    # update the errors dictionary
    errors_dict['train_acc'].append(train_acc)
    errors_dict['train_loss'].append(train_obj)
    errors_dict['valid_acc'].append(valid_acc)
    errors_dict['valid_loss'].append(valid_obj)

    genotype = model.genotype()

    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    #state = {'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'arch_parameters': model.arch_parameters(),
    #        }

    #utils.save_checkpoint(state, False, args.save, args.task_id)

  with codecs.open(os.path.join(args.save,
                                'errors_{}.json'.format(args.task_id)),
                                'w', encoding='utf-8') as file:
    json.dump(errors_dict, file, separators=(',', ':'))

  with open(os.path.join(args.save,
                         'arch_{}'.format(args.task_id)),
            'w') as file:
    file.write(str(genotype))

  utils.write_yaml_results(args, args.results_file_arch, str(genotype))
  utils.write_yaml_results(args, args.results_file_perf, 100-valid_acc)

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
    #param_grads = [p.grad for p in model.parameters() if p.grad is not None]
    #param_grads = torch.cat([x.view(-1) for x in param_grads])
    #param_grads = param_grads.cpu().data.numpy()
    #grad_norm = np.linalg.norm(param_grads)

    #gradient_vector = torch.cat([x.view(-1) for x in gradient_vector]) 
    #grad_norm = LA.norm(gradient_vector.cpu())
    #logging.info('\nCurrent grad norm based on Train Dataset: %.4f',
    #             grad_norm)

    H = analyser.compute_Hw(input, target, input_search, target_search,
                            lr, optimizer, False)
    g = analyser.compute_dw(input, target, input_search, target_search,
                            lr, optimizer, False)
    g = torch.cat([x.view(-1) for x in g])

    #del _data_loader

    state = {'epoch': epoch,
             'H': H.cpu().data.numpy().tolist(),
             'g': g.cpu().data.numpy().tolist(),
             #'g_train': float(grad_norm),
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
  space = spaces_dict[args.space]
  analyser = main(space)

