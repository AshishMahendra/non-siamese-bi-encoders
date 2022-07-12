# flake8: noqa
import os
import torch
from typing import List, Union
from fastapi import HTTPException
from utils.evaluate import get_embeddings
from transformers import AutoModel, AutoConfig, AutoTokenizer
from utils.models import BiEncoder
import traceback
import numpy as np
from utils.train import train_model
import random
import json
import shutil

# device = torch.device("cpu")
# uncomment this if you wish to use GPU to train
# this is commented out because this causes issues with
# unittest on machines with GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
arch = 'bi'

# funtion to set seed for the module


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def config_setup():
    """
    Loading configurations from utils/config.cfg and
    initialize tokenizer and model
    """
    global model, tokenizer, model_config, train_config
    dirname = os.path.dirname(__file__)
    m_config_fname = os.path.join(dirname, "utils/model_config.json")
    t_config_fname = os.path.join(dirname, "utils/train_config.json")
    with open(m_config_fname, "r") as jsonfile:
        model_config = json.load(jsonfile)
    with open(t_config_fname, "r") as jsonfile:
        train_config = json.load(jsonfile)
    train_config.update({'device': device.type})
    trf_config = AutoConfig.from_pretrained(model_config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"],
                                              do_lower_case=True,
                                              clean_text=False)
    if arch == 'cross':

        bert = AutoModel.from_config(trf_config)

        # model = CrossEncoder(config=trf_config,
        #                      bert=bert,)

        # model.to(train_config['device'])
        # set_seed(train_config['seed'])
    else:
        if model_config["shared"] is True:
            cont_bert = AutoModel.from_config(trf_config)
            cand_bert = cont_bert
            print("shared model created")
        else:
            cont_bert = AutoModel.from_config(trf_config)
            cand_bert = AutoModel.from_config(trf_config)
            print("non shared model created")
    model = BiEncoder(config=trf_config,
                      cont_bert=cont_bert,
                      cand_bert=cand_bert,
                      shared=model_config["shared"],
                      loss_type=model_config["loss_type"],
                      loss_function=model_config["loss_function"])

    model.to(train_config['device'])
    set_seed(train_config['seed'])


config_setup()


def cosine_sim(vec_a: List[float], vec_b: List[float]):
    """
    Caculate the cosine similarity score of two given vectors
    Param 1 - First vector
    Param 2 - Second vector
    Return - float between 0 and 1
    """

    result = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) *
                                     np.linalg.norm(vec_b))
    return result.astype(float)


def get_context_emb(contexts: List):
    """
    Take list of context and returns the embeddings
    """
    model.eval()
    embedding = get_embeddings(
        model=model,
        tokenizer=tokenizer,
        text_data=contexts,
        embed_type="context",
        train_config=train_config)
    return embedding

# API for geting Candidates Embedding


def get_candidate_emb(candidates: List):
    """
    Take list of candidates and returns the embeddings
    """
    model.eval()
    embedding = get_embeddings(
        model, tokenizer,
        text_data=candidates,
        embed_type="candidate",
        train_config=train_config)
    return embedding


def dot_prod(vec_a: List[float], vec_b: List[float]):
    """
    Caculate the dot product of two given vectors
    Param 1 - First vector
    Param 2 - Second vector
    Return - dot product
    """
    dot_product = np.matmul(vec_a, vec_b)
    return dot_product.astype(float)


def infer(contexts: Union[List[str], List[List]],
          candidates: Union[List[str], List[List]], loss="dot"):
    """
    Take list of context, candidate and return nearest candidate to the context
    """
    model.eval()
    predicted_candidates = []
    try:
        # if loss == "dot":
        #     predicted_candidates = get_inference(model, tokenizer,
        #                                          contexts=contexts,
        #                                          candidates=candidates,
        #                                          train_config=train_config,
        #                                          arch=arch)
        # elif loss == "cos":
        con_embed = []
        cand_embed = []
        for cont in contexts:
            con_embed.append([get_context_emb(cont)])
        cand_embed = get_candidate_emb(candidates)
        for data in con_embed:
            score_dat = []
            for lbl in cand_embed:
                if loss == "cos":
                    score_dat.append(cosine_sim(
                        vec_a=data, vec_b=lbl))
                else:
                    score_dat.append(dot_prod(vec_a=data, vec_b=lbl))
            predicted_candidates.append(
                candidates[np.argmax(score_dat)])
        return predicted_candidates

    except Exception as e:
        traceback.print_exc()
        print(e)

# API for training


def train(contexts: List, candidates: List, labels: List[int]):
    """
    Take list of context, candidate, labels and trains the model
    """
    global model
    model.train()
    try:
        model = train_model(
            model=model,
            tokenizer=tokenizer,
            contexts=contexts,
            candidates=candidates,
            labels=labels, train_config=train_config
        )
        return "Model Training is complete."
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    with open("clinc_full_training_pos.txt", "r", encoding='utf-8') as f:
        snips_data = json.load(f)
    train(contexts=snips_data['contexts'],
          candidates=snips_data['candidates'],
          labels=snips_data['labels'])
    with open("clinc_full_testing_pos.txt", "r", encoding='utf-8') as f:
        snips_data = json.load(f)
    pred = infer(loss="dot", contexts=snips_data['contexts'],
        candidates=snips_data['candidates'])
    print(pred)