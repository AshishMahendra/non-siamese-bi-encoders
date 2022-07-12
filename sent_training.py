import torch
from sentence_transformers import InputExample, SentencesDataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import models
from sentence_transformers.util import cos_sim

# from sentence_transformers.util import dot_score
import math
import json


train_examples = []
test_examples = []
with open("clinc_full_testing.txt", "r", encoding="utf-8") as f:
    data = json.load(f)
    test_contexts = data["contexts"]
    test_candidates = data["candidates"]


def create_model(model_name="prajjwal1/bert-tiny", max_seq_length=512):
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # dense_model = models.Dense(
    #     in_features=pooling_model.get_sentence_embedding_dimension(),
    #     out_features=256, activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


with open("clinc_full_training_pos.txt", "r", encoding="utf-8") as f:
    dataset = json.load(f)["dataset"]
# print(dataset.keys())
for cand in dataset.keys():
    for con in dataset[cand]:
        train_examples.append(InputExample(texts=[con, cand], label=1))

model = create_model()
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
# train_loss = losses.SoftmaxLoss(
#     model=model,
# sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
# num_labels=len(test_candidates))

train_loss = losses.ContrastiveLoss(model=model)
print(train_loss)
num_epochs = 50
batch_size = 32
model_save_path = "test/"
#  Configure the training ####
# 10% of train data for warm-up
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
print("Warmup-steps: {}".format(warmup_steps))


#  Train the SBERT model ####
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
)

context_embeddings = model.encode(test_contexts)
response_embeddings = model.encode(test_candidates)

predicted_candidates = []
distances = cos_sim(context_embeddings, response_embeddings)
for i in range(len(context_embeddings)):
    predicted_candidates.append(test_candidates[torch.argmax(distances[i]).item()])
print(predicted_candidates)
