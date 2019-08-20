import sys
import time
import math
import copy
import random
import logging
import os
import gc
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

sys.path.append('../RobustDARTS')

#import genotypes
from src.spaces import spaces_dict
from src.search.model_search import Network
from src.search.args import Helper
from src import utils

logger = logging.getLogger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsWrapper(Helper):
    def __init__(self):
        super(DartsWrapper, self).__init__()

        args = AttrDict(self.args.__dict__)
        self.args = args
        self.seed = args.seed

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = False
        cudnn.enabled=True
        cudnn.deterministic=True
        torch.cuda.manual_seed_all(args.seed)

        self.train_queue, self.valid_queue, _, _ = super(DartsWrapper,
                                                         self).get_train_val_loaders()
        setattr(self.train_queue, 'num_workers', 0)
        setattr(self.train_queue, 'worker_init_fn', np.random.seed(args.seed))
        setattr(self.valid_queue, 'num_workers', 0)
        setattr(self.valid_queue, 'worker_init_fn', np.random.seed(args.seed))

        self.train_iter = iter(self.train_queue)
        self.valid_iter = iter(self.valid_queue)

        self.steps = 0
        self.epochs = 0
        self.total_loss = 0
        self.start_time = time.time()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        self.criterion = criterion

        self.primitives = spaces_dict[args.space]

        model = Network(args.init_channels, args.n_classes,
                        args.layers, self.criterion,
                        self.primitives, steps=args.nodes)

        model = model.cuda()
        self.model = model

        optimizer = torch.optim.SGD(
          self.model.parameters(),
          args.learning_rate,
          momentum=args.momentum,
          weight_decay=args.weight_decay)
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, float(args.epochs), eta_min=args.learning_rate_min)


    def train_batch(self, arch, errors_dict):
      args = self.args
      if self.steps % len(self.train_queue) == 0:
        self.scheduler.step()
        self.objs = utils.AvgrageMeter()
        self.top1 = utils.AvgrageMeter()
        self.top5 = utils.AvgrageMeter()
      lr = self.scheduler.get_lr()[0]

      weights = self.get_weights_from_arch(arch)
      self.set_model_weights(weights)

      step = self.steps % len(self.train_queue)
      input, target = next(self.train_iter)

      self.model.train()
      n = input.size(0)

      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda(async=True)

      # get a random minibatch from the search queue with replacement
      self.optimizer.zero_grad()
      logits = self.model(input, discrete=True)
      loss = self.criterion(logits, target)

      loss.backward()
      nn.utils.clip_grad_norm(self.model.parameters(), args.grad_clip)
      self.optimizer.step()

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      self.objs.update(loss.data[0], n)
      self.top1.update(prec1.data[0], n)
      self.top5.update(prec5.data[0], n)

      if step % args.report_freq == 0:
        logger.info('train %03d %e %f %f', step, self.objs.avg, self.top1.avg, self.top5.avg)

      self.steps += 1
      if self.steps % len(self.train_queue) == 0:
        self.epochs += 1
        self.train_iter = iter(self.train_queue)
        valid_err, valid_obj = self.evaluate(arch)
        logger.info('epoch %d  |  train_acc %f  |  valid_acc %f' % (self.epochs, self.top1.avg, 1-valid_err))
        self.save()
        errors_dict['train_acc'].append(self.top1.avg)
        errors_dict['train_loss'].append(self.objs.avg)
        errors_dict['valid_acc'].append(1-valid_err)
        errors_dict['valid_loss'].append(valid_obj)


    def evaluate(self, arch, split=None):
      # Return error since we want to minimize obj val
      logger.info(arch)
      objs = utils.AvgrageMeter()
      top1 = utils.AvgrageMeter()
      top5 = utils.AvgrageMeter()

      weights = self.get_weights_from_arch(arch)
      self.set_model_weights(weights)

      self.model.eval()

      if split is None:
        n_batches = 10
      elif self.args.debug:
        n_batches = 1
      else:
        n_batches = len(self.valid_queue)

      for step in range(n_batches):
        try:
          input, target = next(self.valid_iter)
        except Exception as e:
          logger.info('looping back over valid set')
          self.valid_iter = iter(self.valid_queue)
          input, target = next(self.valid_iter)
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(async=True)

        logits = self.model(input, discrete=True)
        loss = self.criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % self.args.report_freq == 0:
          logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

      return 1-top1.avg, objs.avg

    def save(self):
        utils.save(self.model, os.path.join(self.args.save,
                                            'weights_{}.pt'.format(self.args.task_id)))

    def load(self):
        utils.load(self.model, os.path.join(self.args.save,
                                            'weights_{}.pt'.format(self.args.task_id)))

    def get_weights_from_arch(self, arch):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))
        num_ops = len(self.primitives['primitives_normal'][0])
        n_nodes = self.model._steps

        alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)
        alphas_reduce = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)

        offset = 0
        for i in range(n_nodes):
            normal1 = arch[0][2*i]
            normal2 = arch[0][2*i+1]
            reduce1 = arch[1][2*i]
            reduce2 = arch[1][2*i+1]
            alphas_normal[offset+normal1[0], normal1[1]] = 1
            alphas_normal[offset+normal2[0], normal2[1]] = 1
            alphas_reduce[offset+reduce1[0], reduce1[1]] = 1
            alphas_reduce[offset+reduce2[0], reduce2[1]] = 1
            offset += (i+2)

        arch_parameters = [
          alphas_normal,
          alphas_reduce,
        ]
        return arch_parameters

    def set_model_weights(self, weights):
      self.model.alphas_normal = weights[0]
      self.model.alphas_reduce = weights[1]
      self.model._arch_parameters = [self.model.alphas_normal, self.model.alphas_reduce]

    def sample_arch(self):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))
        num_ops = len(self.primitives['primitives_normal'][0])
        n_nodes = self.model._steps

        normal = []
        reduction = []
        for i in range(n_nodes):
            ops = np.random.choice(range(num_ops), 4)
            nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
            nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])

        return (normal, reduction)


    def perturb_arch(self, arch):
        new_arch = copy.deepcopy(arch)
        num_ops = len(self.primitives['primitives_normal'][0])

        cell_ind = np.random.choice(2)
        step_ind = np.random.choice(self.model._steps)
        nodes_in = np.random.choice(step_ind+2, 2, replace=False)
        ops = np.random.choice(range(num_ops), 2)

        new_arch[cell_ind][2*step_ind] = (nodes_in[0], ops[0])
        new_arch[cell_ind][2*step_ind+1] = (nodes_in[1], ops[1])
        return new_arch


