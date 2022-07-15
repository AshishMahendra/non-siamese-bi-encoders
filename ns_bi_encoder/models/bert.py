"""
BERT sequence classifier upperbound
"""

import os
import math
from os.path import join as os_join

import numpy as np
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, IntervalStrategy
)
import datasets

from stefutil import *
from ns_bi_encoder.util import *


MODEL_NAME = 'BERT Seq CLS'
HF_MODEL_NAME = 'bert-base-uncased'


def compute_metrics(eval_pred):
    if not hasattr(compute_metrics, 'acc'):
        compute_metrics.acc = datasets.load_metric('accuracy')
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return dict(acc=compute_metrics.acc.compute(predictions=preds, references=labels)['accuracy'])


if __name__ == '__main__':
    import transformers

    seed = sconfig('random-seed')

    logger = get_logger(f'{MODEL_NAME} Train')
    # dnm = 'banking77'
    dnm = 'snips'
    dset = datasets.load_from_disk(os_join(u.dset_path, 'processed', dnm))
    tr, ts = dset['train'], dset['test']
    logger.info(f'Loaded dataset {logi(dnm)} with {logi(dset)} ')

    bsz, n_ep, lr = 16, 50, 1e-4

    feat_label = tr.features['label']

    tokenizer = BertTokenizer.from_pretrained(HF_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(HF_MODEL_NAME, num_labels=feat_label.num_classes)
    mic(tokenizer, type(model))

    def tokenize(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    tr, ts = tr.shuffle(seed=seed), ts.shuffle(seed=seed)
    map_args = dict(batched=True, batch_size=16, num_proc=os.cpu_count())
    tr, ts = tr.map(tokenize, **map_args), ts.map(tokenize, **map_args)
    warmup_steps = math.ceil(len(tr) * n_ep * 0.1)  # 10% of train data

    dir_nm = f'{now(for_path=True)}_{MODEL_NAME}-{dnm}'
    output_path = os_join(get_output_base(), u.proj_dir, u.model_dir, dir_nm)
    proj_output_path = os_join(u.base_path, u.proj_dir, u.model_path, dir_nm, 'trained')
    mic(dir_nm, proj_output_path)
    d_log = {
        'learning rate': lr, 'batch size': bsz, 'epochs': n_ep,
        'warmup steps': warmup_steps, 'save path': output_path
    }
    logger.info(f'Launching {MODEL_NAME} training with {log_dict(d_log)}... ')

    training_args = TrainingArguments(
        learning_rate=lr,
        output_dir=output_path,
        num_train_epochs=n_ep,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        evaluation_strategy=IntervalStrategy.EPOCH
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tr, eval_dataset=ts, compute_metrics=compute_metrics
    )

    transformers.set_seed(seed)
    trainer.train()
    mic(trainer.evaluate())
    trainer.save_model(proj_output_path)
    tokenizer.save_pretrained(proj_output_path)
    os.listdir(proj_output_path)
