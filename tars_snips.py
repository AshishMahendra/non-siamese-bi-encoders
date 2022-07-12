import time
# import numpy as np
from flair.models import TARSClassifier
from flair.data import Corpus, Sentence
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer
import torch
import json
torch.cuda.empty_cache()
train_text = []
train_label = []
test_text = []
test_label = []
with open("full_training.txt", "r", encoding='utf-8') as f:
    text_data = json.load(f)
c = 1
for cont, cand in zip(text_data['contexts'], text_data['candidates']):
    if c == 1:
        train_text.append(cont)
        train_label.append(cand)
        c = 0
    else:
        c = 1
with open("full_testing.txt", "r", encoding='utf-8') as f:
    text_data = json.load(f)
for cont, cand in zip(text_data['contexts'], text_data['candidates']):
    test_text.append(cont)
    test_label.append(cand)
train_ds = []
test_ds = []

for datapoint, class_val in zip(train_text, train_label):
    train_ds.append(Sentence(datapoint.lower()).add_label(
        'snips_mobile_data', class_val))
train_ds = SentenceDataset(train_ds)
for datapoint, class_val in zip(test_text, test_label):
    test_ds.append(Sentence(datapoint.lower()).add_label(
        'snips_mobile_data', class_val))
test_ds = SentenceDataset(test_ds)
print(train_ds[0])
print(test_ds[0])
print("data_load Completed")
corpus = Corpus(train=train_ds, test=test_ds)
tars = TARSClassifier(embeddings="prajjwal1/bert-tiny")
tars.add_and_switch_to_new_task("snips_mobile_data",
                                label_dictionary=corpus.make_label_dictionary(
                                    label_type="snips_mobile_data"),
                                label_type="snips_mobile_data")
trainer = ModelTrainer(tars, corpus)

start_time = time.time()

data = trainer.train(base_path='taggers/snips_small',
                     learning_rate=0.02,
                     mini_batch_size=16,
                     max_epochs=10,
                     monitor_train=False,
                     embeddings_storage_mode="cuda",
                     train_with_dev=True
                     )

print(
    f"""\n\nTime taken to complete the model training
    : {time.time()-start_time}\n\n""")
print(data)
