"""
BERT sequence classifier upperbound
"""

import json
from os.path import join as os_join

import pandas as pd
import datasets

from stefutil import *
from ns_bi_encoder.util import *


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def process_dataset(dataset_name: str):
    """
    Clean up dataset from json to (text, label) pair format, with set of labels
    """
    d_path = sconfig(f'datasets.{dataset_name}.path')
    path_tr, d_path_ts = d_path['train'], d_path['test']
    path_ts_txt, path_ts_lb = d_path_ts['text'], d_path_ts['label']
    tr = load_json(path_tr)['dataset']
    dset = dict(train=sum(([dict(text=t, label=lb) for t in txts] for lb, txts in tr.items()), start=[]))
    labels = set() | tr.keys()

    d_ts_txt, ts_lb = load_json(path_ts_txt), load_json(path_ts_lb)
    ts_txt = d_ts_txt['contexts']
    assert len(ts_txt) == len(ts_lb)  # sanity check
    dset['test'] = [dict(text=t, label=lb) for t, lb in zip(ts_txt, ts_lb)]
    if dataset_name == 'sgd':
        labels |= set(d_ts_txt['candidates'])  # union of labels in train & test split so that inference runs
    else:
        assert set(d_ts_txt['candidates']) == labels
    return dict(pairs=dset, labels=sorted(labels))


def datasets_to_disk():
    logger = get_logger('Dset2Disk')
    for dnm in sconfig('datasets').keys():
        logger.info(f'Processing dataset {logi(dnm)}... ')
        dsets = datasets.DatasetDict()
        d_dset = process_dataset(dnm)
        d_pairs, labels = d_dset['pairs'], d_dset['labels']
        d_log = {'#sample': {s: len(ps) for s, ps in d_pairs.items()}, '#label': len(labels), 'labels': labels}
        logger.info(f'Refactored with {logi(d_log)}')

        for split, pairs in d_pairs.items():
            df = pd.DataFrame(pairs)
            lb2id = {lb: i for i, lb in enumerate(labels)}
            df.label = df.label.map(lb2id)
            feats = datasets.Features(text=datasets.Value(dtype='string'), label=datasets.ClassLabel(names=labels))
            dsets[split] = datasets.Dataset.from_pandas(df, features=feats)
        save_path = os_join(u.dset_path, 'processed', dnm)
        logger.info(f'Writing to {logi(save_path)}... ')
        dsets.save_to_disk(save_path)


if __name__ == '__main__':
    datasets_to_disk()