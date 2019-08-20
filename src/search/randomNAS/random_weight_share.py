import sys
import os
import ast
import shutil
import logging
import codecs
import json
import inspect
import pickle
import argparse
import numpy as np

sys.path.append('../RobustDARTS')

from src.search.randomNAS.darts_wrapper_discrete import DartsWrapper
from src.search.randomNAS.parse_cnn_arch import parse_arch_to_darts
from src import utils

class Rung:
    def __init__(self, rung, nodes):
        self.parents = set()
        self.children = set()
        self.rung = rung
        for node in nodes:
            n = nodes[node]
            if n.rung == self.rung:
                self.parents.add(n.parent)
                self.children.add(n.node_id)

class Node:
    def __init__(self, parent, arch, node_id, rung):
        self.parent = parent
        self.arch = arch
        self.node_id = node_id
        self.rung = rung
    def to_dict(self):
        out = {'parent':self.parent, 'arch': self.arch, 'node_id': self.node_id, 'rung': self.rung}
        if hasattr(self, 'objective_val'):
            out['objective_val'] = self.objective_val
        return out

class Random_NAS:
    def __init__(self, B, model, seed, save_dir):
        self.save_dir = save_dir

        self.B = B
        self.model = model
        self.args = model.args
        self.seed = seed

        self.iters = 0

        self.arms = {}
        self.node_id = 0

    def get_arch(self):
        arch = self.model.sample_arch()
        self.arms[self.node_id] = Node(self.node_id, arch, self.node_id, 0)
        self.node_id += 1
        return arch

    def save(self):
        to_save = {a: self.arms[a].to_dict() for a in self.arms}
        # Only replace file if save successful so don't lose results of last pickle save
        with open(
            os.path.join(self.save_dir,'results_tmp_{}.pkl'.format(self.args.task_id)),'wb'
        ) as f:
            pickle.dump(to_save, f)
        shutil.copyfile(
            os.path.join(self.save_dir,
                         'results_tmp_{}.pkl'.format(
                             self.args.task_id
                         )), os.path.join(self.save_dir,
                                          'results_{}.pkl'.format(self.args.task_id))
        )

        self.model.save()

    def run(self):
        errors_dict = {'train_acc': [], 'train_loss': [], 'valid_acc': [],
                       'valid_loss': []}
        while self.iters < self.B:
            arch = self.get_arch()
            self.model.train_batch(arch, errors_dict)
            self.iters += 1
            if self.iters % 500 == 0:
                self.save()
            if (self.iters % self.args.report_freq) and self.args.debug:
                break

        self.save()
        with codecs.open(os.path.join(self.args.save,
                                      'errors_{}.json'.format(self.args.task_id)),
                         'w', encoding='utf-8') as file:
            json.dump(errors_dict, file, separators=(',', ':'))


    def get_eval_arch(self, rounds=None, n_samples=1000):
        #n_rounds = int(self.B / 7 / 1000)
        if rounds is None:
            n_rounds = max(1,int(self.B/10000))
        else:
            n_rounds = rounds
        best_rounds = []
        for r in range(n_rounds):
            sample_vals = []
            for _ in range(n_samples):
                arch = self.model.sample_arch()
                try:
                    ppl, _ = self.model.evaluate(arch)
                except Exception as e:
                    ppl = 1000000
                logging.info(arch)
                logging.info('objective_val: %.3f' % ppl)
                sample_vals.append((arch, ppl))
            sample_vals = sorted(sample_vals, key=lambda x:x[1])

            full_vals = []
            if 'split' in inspect.getargspec(self.model.evaluate).args:
                for i in range(10):
                    arch = sample_vals[i][0]
                    try:
                        ppl, _ = self.model.evaluate(arch, split='valid')
                    except Exception as e:
                        ppl = 1000000
                    full_vals.append((arch, ppl))
                full_vals = sorted(full_vals, key=lambda x:x[1])
                logging.info('best arch: %s, best arch valid performance: %.3f' % (' '.join([str(i) for i in full_vals[0][0]]), full_vals[0][1]))
                best_rounds.append(full_vals[0])
            else:
                best_rounds.append(sample_vals[0])
        return best_rounds

def main(wrapper):
    args = wrapper.args
    model = wrapper.model
    save_dir = args.save

    try:
        wrapper.load()
        logging.info('loaded previously saved weights')
    except Exception as e:
        print(e)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info('Args: {}'.format(args))

    if args.eval_only:
        assert save_dir is not None

    data_size = 25000
    time_steps = 1

    B = int(args.epochs * data_size / args.batch_size / time_steps)

    searcher = Random_NAS(B, wrapper, args.seed, save_dir)
    logging.info('budget: %d' % (searcher.B))
    if not args.eval_only:
        searcher.run()
        archs = searcher.get_eval_arch(args.randomnas_rounds, args.n_samples)
    else:
        np.random.seed(args.seed+1)
        archs = searcher.get_eval_arch(2)
    logging.info(archs)
    #arch = ' '.join([str(a) for a in archs[0][0]])
    arch = str(archs[0][0])
    arch = parse_arch_to_darts('cnn', ast.literal_eval(arch), args.space)
    with open(os.path.join(args.save, 'arch_{}'.format(args.task_id)),'w') as f:
        f.write(str(arch))

    logging.info(str(arch))
    utils.write_yaml_results(args, args.results_file_arch, str(arch))
    return arch


if __name__ == "__main__":
    wrapper = DartsWrapper()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(wrapper.args.save,
                                          'log_{}.txt'.format(wrapper.args.task_id)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main(wrapper)


