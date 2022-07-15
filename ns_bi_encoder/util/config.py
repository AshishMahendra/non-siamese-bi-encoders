import os
from os.path import join as os_join

from stefutil import *

from ns_bi_encoder.util.util import *


config_dict = {
    'datasets': dict(
        banking77=dict(
            path=dict(
                train=os_join(u.dset_path, 'banking_train.json'),
                test=dict(
                    text=os_join(u.dset_path, 'banking_test.json'),
                    label=os_join(u.dset_path, 'banking_test_label.json')
                )
            )
        ),
        clinc_150=dict(
            path=dict(
                train=os_join(u.dset_path, 'clinc_full_training_pos.json'),
                test=dict(
                    text=os_join(u.dset_path, 'clinc_full_testing_pos.json'),
                    label=os_join(u.dset_path, 'clinc_test_label.json')
                )
            )
        ),
        sgd=dict(
            path=dict(
                train=os_join(u.dset_path, 'sgd_train.json'),
                test=dict(
                    text=os_join(u.dset_path, 'sgd_test.json'),
                    label=os_join(u.dset_path, 'sgd_test_label.json')
                )
            )
        ),
        snips=dict(
            path=dict(
                train=os_join(u.dset_path, 'snips_full_training_pos.json'),
                test=dict(
                    text=os_join(u.dset_path, 'snips_full_testing_pos.json'),
                    label=os_join(u.dset_path, 'snips_testing_label.json')
                )
            )
        )
    ),
    'random-seed': 77
}


for dnm in config_dict['datasets'].keys():
    d_path = get(config_dict, f'datasets.{dnm}.path')
    set_(config_dict, f'datasets.{dnm}.label', dict(  # Add labels for each dataset
        train=sorted(load_json(d_path['train'])['dataset'].keys()),
        test=sorted(load_json(get(d_path, 'test.text'))['candidates'])
    ))


if __name__ == '__main__':
    import json

    from stefutil import *

    path_util = os_join(u.proj_path, u.pkg_nm, 'util')
    mic(os.listdir(path_util))
    with open(os_join(path_util, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
