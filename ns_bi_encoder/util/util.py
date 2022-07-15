import json
from typing import List

import pandas as pd
import datasets

from ns_bi_encoder.util import *

from os.path import join as os_join

from stefutil import *

from ns_bi_encoder.util.data_path import BASE_PATH, PROJ_DIR, DSET_DIR, PKG_NM, MODEL_DIR


__all__ = ['sconfig', 'u', 'load_json', 'get_output_base']

sconfig = StefConfig(config_file=os_join(BASE_PATH, PROJ_DIR, PKG_NM, 'util', 'config.json')).__call__
u = StefUtil(
    base_path=BASE_PATH, project_dir=PROJ_DIR, package_name=PKG_NM, dataset_dir=DSET_DIR, model_dir=MODEL_DIR
)
u.dset_path = os_join(u.base_path, u.proj_dir, u.dset_dir)


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


def get_output_base():
    # For remote machines, save heavy-duty data somewhere else to save `/home` disk space
    hnm = get_hostname()
    if 'clarity' in hnm:  # Clarity lab
        return '/data'
    elif 'arc-ts' in hnm:  # Great Lakes; `profmars0` picked arbitrarily among [`profmars0`, `profmars1`]
        # Per https://arc.umich.edu/greatlakes/user-guide/
        return os_join('/scratch', 'profmars_root', 'profmars0', 'stefanhg')
    else:
        return u.base_path


def check_label_overlap(dataset_name: str, text: str) -> List[str]:
    """
    :return: labels in the dataset that overlap with the text
    """
    d_lbs = sconfig(f'datasets.{dataset_name}.label')
    labels = sorted(set().union(*(d_lbs.values())))
    if dataset_name == 'clinc_150':
        labels = [lb.replace('_', ' ') for lb in labels]

    words = set(text.split())

    def overlap(label: str) -> bool:
        inter = words.intersection(label.split())
        return len(inter) > 0

    return [lb for lb in labels if overlap(lb)]


if __name__ == '__main__':
    # datasets_to_disk()

    dnm = 'clinc_150'
    # txt = 'what do i need to make a cajun chili'
    # txt = 'you need to speak softer'
    # txt = 'what\'s my vacation day total'
    # txt = 'are there specific shots i need before traveling to japan'
    txt = 'what shots do i need to get in order to travel to khartoum'
    mic(check_label_overlap(dnm, txt))
