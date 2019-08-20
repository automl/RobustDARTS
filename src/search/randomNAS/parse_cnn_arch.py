import pickle
import sys
import ast

sys.path.append('../RobustDARTS')

from src.spaces import primitives_1
from src.utils import Genotype


def get_op(node_idx, prev_node_idx, op_idx, normal=True):
    PRIMITIVES = primitives_1['primitives_normal' if normal else 'primitives_reduct']

    node_to_in_edges = {
        0: (0, 2),
        1: (2, 5),
        2: (5, 9),
        3: (9, 14)
    }

    in_edges = node_to_in_edges[node_idx]
    op = PRIMITIVES[in_edges[0]: in_edges[1]][prev_node_idx][op_idx]
    return op


def parse_arch_to_darts(benchmark, arch, space='s1'):
    if space == 's2':
        op_dict = {
            0: 'skip_connect',
            1: 'sep_conv_3x3'
        }
    elif space == 's3':
        op_dict = {
            0: 'none',
            1: 'skip_connect',
            2: 'sep_conv_3x3'
        }
    elif space == 's4':
        op_dict = {
            0: 'noise',
            1: 'sep_conv_3x3'
        }
    else:
        op_dict = {
        0: 'none',
        1: 'max_pool_3x3',
        2: 'avg_pool_3x3',
        3: 'skip_connect',
        4: 'sep_conv_3x3',
        5: 'sep_conv_5x5',
        6: 'dil_conv_3x3',
        7: 'dil_conv_5x5'
        }

    darts_arch = [[], []]

    for i, (cell, normal) in enumerate(zip(arch, [True, False])):
        for j, n in enumerate(cell):
            if space == 's1':
                darts_arch[i].append((get_op(j//2, n[0], n[1], normal),
                                      n[0]))
            else:
                darts_arch[i].append((op_dict[n[1]], n[0]))


    arch_str = 'Genotype(normal=%s, normal_concat=[2,3,4,5], reduce=%s, reduce_concat=[2,3,4,5])' % (str(darts_arch[0]), str(darts_arch[1]))
    print(arch_str)

    return eval(arch_str)

if __name__=="__main__":
    args = sys.argv[1:]
    print(args[0])
    arch = ast.literal_eval(args[0])
    parse_arch_to_darts('cnn', arch)
