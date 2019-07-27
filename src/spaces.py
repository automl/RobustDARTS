from collections import OrderedDict




primitives_1 = OrderedDict([('primitives_normal', [['skip_connect',
                                                    'dil_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['max_pool_3x3',
                                                     'skip_connect'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['max_pool_3x3',
                                                     'skip_connect'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['dil_conv_3x3',
                                                     'dil_conv_5x5'],
                                                    ['dil_conv_3x3',
                                                     'dil_conv_5x5']]),
                             ('primitives_reduct', [['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'dil_conv_3x3'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5']])])

primitives_2 = OrderedDict([('primitives_normal', 14 * [['skip_connect',
                                                         'sep_conv_3x3']]),
                            ('primitives_reduct', 14 * [['skip_connect',
                                                         'sep_conv_3x3']])])

primitives_3 = OrderedDict([('primitives_normal', 14 * [['none',
                                                         'skip_connect',
                                                         'sep_conv_3x3']]),
                            ('primitives_reduct', 14 * [['none',
                                                         'skip_connect',
                                                         'sep_conv_3x3']])])

primitives_4 = OrderedDict([('primitives_normal', 14 * [['noise',
                                                         'sep_conv_3x3']]),
                            ('primitives_reduct', 14 * [['noise',
                                                         'sep_conv_3x3']])])

PRIMITIVES = [
    #'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

primitives_5 = OrderedDict([('primitives_normal', 14 * [PRIMITIVES]),
                            ('primitives_reduct', 14 * [PRIMITIVES])])

spaces_dict = {
    's1': primitives_1,
    's2': primitives_2,
    's3': primitives_3,
    's4': primitives_4,
    's5': primitives_5, # original DARTS space
}

