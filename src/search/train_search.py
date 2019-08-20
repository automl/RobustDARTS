import os
import sys
import glob
import numpy as np
import torch
import json
import codecs
import pickle
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from copy import deepcopy
from numpy import linalg as LA
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

schedule_of_params = []


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

  model_init = Network(args.init_channels, args.n_classes, args.layers, criterion,
                       primitives, steps=args.nodes)
  model_init = model_init.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model_init))

  optimizer_init = torch.optim.SGD(
      model_init.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  architect_init = Architect(model_init, args)

  scheduler_init = CosineAnnealingLR(
        optimizer_init, float(args.epochs), eta_min=args.learning_rate_min)

  analyser_init = Analyzer(args, model_init)
  la_tracker = utils.EVLocalAvg(args.window, args.report_freq_hessian,
                                args.epochs)

  train_queue, valid_queue, train_transform, valid_transform = helper.get_train_val_loaders()


  errors_dict = {'train_acc': [], 'train_loss': [], 'valid_acc': [],
                 'valid_loss': []}

  #for epoch in range(args.epochs):
  def train_epochs(epochs_to_train, iteration,
                   args=args, model=model_init, optimizer=optimizer_init,
                   scheduler=scheduler_init,
                   train_queue=train_queue, valid_queue=valid_queue,
                   train_transform=train_transform,
                   valid_transform=valid_transform,
                   architect=architect_init, criterion=criterion,
                   primitives=primitives, analyser=analyser_init,
                   la_tracker=la_tracker,
                   errors_dict=errors_dict, start_epoch=-1):

    logging.info('STARTING ITERATION: %d', iteration)
    logging.info('EPOCHS TO TRAIN: %d', epochs_to_train - start_epoch - 1)

    la_tracker.stop_search = False

    if epochs_to_train - start_epoch - 1 <= 0:
        return model.genotype(), -1
    for epoch in range(start_epoch+1, epochs_to_train):
      # set the epoch to the right one
      #epoch += args.epochs - epochs_to_train

      scheduler.step(epoch)
      lr = scheduler.get_lr()[0]
      if args.drop_path_prob != 0:
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
                                   optimizer, lr, analyser, la_tracker,
                                   iteration)
      logging.info('train_acc %f', train_acc)

      # validation
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
      logging.info('valid_acc %f', valid_acc)

      # update the errors dictionary
      errors_dict['train_acc'].append(100 - train_acc)
      errors_dict['train_loss'].append(train_obj)
      errors_dict['valid_acc'].append(100 - valid_acc)
      errors_dict['valid_loss'].append(valid_obj)

      genotype = model.genotype()

      logging.info('genotype = %s', genotype)

      print(F.softmax(model.alphas_normal, dim=-1))
      print(F.softmax(model.alphas_reduce, dim=-1))

      state = {'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'alphas_normal': model.alphas_normal.data,
               'alphas_reduce': model.alphas_reduce.data,
               'arch_optimizer': architect.optimizer.state_dict(),
               'lr': lr,
               'ev': la_tracker.ev,
               'ev_local_avg': la_tracker.ev_local_avg,
               'genotypes': la_tracker.genotypes,
               'la_epochs': la_tracker.la_epochs,
               'la_start_idx': la_tracker.la_start_idx,
               'la_end_idx': la_tracker.la_end_idx,
               #'scheduler': scheduler.state_dict(),
              }

      utils.save_checkpoint(state, False, args.save, epoch, args.task_id)

      if args.compute_hessian:
        ev = -1
      else:
        ev = la_tracker.ev[-1]
      params = {'iteration': iteration,
                'epoch': epoch,
                'wd': args.weight_decay,
                'ev': ev,
               }

      schedule_of_params.append(params)

      # limit the number of iterations based on the maximum regularization
      # value predefined by the user
      final_iteration = round(
          np.log(args.max_weight_decay) / np.log(args.weight_decay), 1) == 1.

      if la_tracker.stop_search and not final_iteration:
        if args.early_stop == 1:
          # set the following to the values they had at stop_epoch
          errors_dict['valid_acc'] = errors_dict['valid_acc'][:la_tracker.stop_epoch + 1]
          genotype = la_tracker.stop_genotype
          valid_acc = 100 - errors_dict['valid_acc'][la_tracker.stop_epoch]
          logging.info(
              'Decided to stop the search at epoch %d (Current epoch: %d)',
              la_tracker.stop_epoch, epoch
          )
          logging.info(
              'Validation accuracy at stop epoch: %f', valid_acc
          )
          logging.info(
              'Genotype at stop epoch: %s', genotype
          )
          break

        elif args.early_stop == 2:
          # simulate early stopping and continue search afterwards
          simulated_errors_dict = errors_dict['valid_acc'][:la_tracker.stop_epoch + 1]
          simulated_genotype = la_tracker.stop_genotype
          simulated_valid_acc = 100 - simulated_errors_dict[la_tracker.stop_epoch]
          logging.info(
              '(SIM) Decided to stop the search at epoch %d (Current epoch: %d)',
              la_tracker.stop_epoch, epoch
          )
          logging.info(
              '(SIM) Validation accuracy at stop epoch: %f', simulated_valid_acc
          )
          logging.info(
              '(SIM) Genotype at stop epoch: %s', simulated_genotype
          )

          with open(os.path.join(args.save,
                                 'arch_early_{}'.format(args.task_id)),
                    'w') as file:
            file.write(str(simulated_genotype))

          utils.write_yaml_results(args, 'early_'+args.results_file_arch,
                                   str(simulated_genotype))
          utils.write_yaml_results(args, 'early_stop_epochs',
                                   la_tracker.stop_epoch)

          args.early_stop = 0

        elif args.early_stop == 3:
          # adjust regularization
          simulated_errors_dict = errors_dict['valid_acc'][:la_tracker.stop_epoch + 1]
          simulated_genotype = la_tracker.stop_genotype
          simulated_valid_acc = 100 - simulated_errors_dict[la_tracker.stop_epoch]
          stop_epoch = la_tracker.stop_epoch
          start_again_epoch = stop_epoch - args.extra_rollback_epochs
          logging.info(
              '(ADA) Decided to increase regularization at epoch %d (Current epoch: %d)',
              stop_epoch, epoch
          )
          logging.info(
              '(ADA) Rolling back to epoch %d', start_again_epoch
          )
          logging.info(
              '(ADA) Restoring model parameters and continuing for %d epochs',
              epochs_to_train - start_again_epoch - 1
          )

          del model
          del architect
          del optimizer
          del scheduler
          del analyser

          model_new = Network(args.init_channels, args.n_classes, args.layers, criterion,
                          primitives, steps=args.nodes)
          model_new = model_new.cuda()

          optimizer_new = torch.optim.SGD(
              model_new.parameters(),
              args.learning_rate,
              momentum=args.momentum,
              weight_decay=args.weight_decay)

          architect_new = Architect(model_new, args)

          analyser_new = Analyzer(args, model_new)

          la_tracker = utils.EVLocalAvg(args.window, args.report_freq_hessian,
                                        args.epochs)

          lr = utils.load_checkpoint(model_new, optimizer_new, None,
                                     architect_new, args.save, la_tracker,
                                     start_again_epoch, args.task_id)

          args.weight_decay *= args.mul_factor
          for param_group in optimizer_new.param_groups:
              param_group['weight_decay'] = args.weight_decay

          scheduler_new = CosineAnnealingLR(
              optimizer_new, float(args.epochs),
              eta_min=args.learning_rate_min)


          logging.info(
              '(ADA) Validation accuracy at stop epoch: %f', simulated_valid_acc
          )
          logging.info(
              '(ADA) Genotype at stop epoch: %s', simulated_genotype
          )

          logging.info(
              '(ADA) Adjusting L2 regularization to the new value: %f',
              args.weight_decay
          )

          genotype, valid_acc = train_epochs(args.epochs,
                                             iteration + 1, model=model_new,
                                             optimizer=optimizer_new,
                                             architect=architect_new,
                                             scheduler=scheduler_new,
                                             analyser=analyser_new,
                                             start_epoch=start_again_epoch)
          args.early_stop = 0
          break

    return genotype, valid_acc

  # call train_epochs recursively
  genotype, valid_acc = train_epochs(args.epochs, 1)

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

  with open(os.path.join(args.save,
                         'schedule_{}.pickle'.format(args.task_id)),
            'ab') as file:
    pickle.dump(schedule_of_params, file, pickle.HIGHEST_PROTOCOL)


def train(epoch, primitives, train_queue, valid_queue, model, architect,
          criterion, optimizer, lr, analyser, local_avg_tracker, iteration=1):
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

      architect.step(input, target, input_search, target_search, lr, optimizer,
                     unrolled=args.unrolled)

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

  if args.compute_hessian:
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

      if not args.debug:
        H = analyser.compute_Hw(input, target, input_search, target_search,
                                lr, optimizer, False)
        g = analyser.compute_dw(input, target, input_search, target_search,
                                lr, optimizer, False)
        g = torch.cat([x.view(-1) for x in g])

        del _data_loader

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

        # early stopping
        ev = max(LA.eigvals(H.cpu().data.numpy()))
      else:
        ev = 0.1
        if epoch >= 8 and iteration==1:
          ev = 2.0
      logging.info('CURRENT EV: %f', ev)
      local_avg_tracker.update(epoch, ev, model.genotype())

      if args.early_stop and epoch != (args.epochs - 1):
        local_avg_tracker.early_stop(epoch, args.factor, args.es_start_epoch,
                                     args.delta)

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
  main(space)

