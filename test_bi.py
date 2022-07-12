import os
from sklearn.linear_model import LogisticRegression
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from model import SelectionDataset, SelectionSequentialTransform, SelectionJoinTransform, EvalDataset
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from model.encoder import BiEncoder,PolyEncoder
import argparse
import random
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #   torch.cuda.manual_seed_all(args.seed)


def eval_running_model(dataloader, model, mode="context"):
    loss_fct = CrossEntropyLoss()
    model.eval()
    eval_loss, eval_hit_times = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for step, batch in enumerate(dataloader, start=1):
        batch = tuple(t.to(device) for t in batch)
        context_token_ids_list_batch,  context_input_masks_list_batch, \
            response_token_ids_list_batch,  response_input_masks_list_batch, labels_batch = batch

        with torch.no_grad():
            logits = model(context_token_ids_list_batch,  context_input_masks_list_batch,
                           response_token_ids_list_batch,  response_input_masks_list_batch)
            # 5 is a coef
            # loss = loss_fct(logits * 5, torch.argmax(labels_batch, 1))
        # print(logits)
        eval_hit_times += (logits.argmax(-1) ==
                           torch.argmax(labels_batch, 1)).sum().item()
        # eval_loss += loss.item()

        nb_eval_examples += labels_batch.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_hit_times / nb_eval_examples
    result = {
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    # parser.add_argument("--bert_model", default='ckpt/pretrained/distilbert-base-uncased', type=str)
    # parser.add_argument("--model_type", default='distilbert', type=str)
    parser.add_argument(
        "--bert_model", default='ckpt/pretrained/bert-small-uncased', type=str)
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--use_pretrain", action="store_true")
    parser.add_argument("--architecture", required=True,
                        type=str, help='[poly, bi]')

    parser.add_argument("--max_contexts_length", default=128, type=int)
    parser.add_argument("--max_response_length", default=64, type=int)

    parser.add_argument("--eval_batch_size", default=2,
                        type=int, help="Total batch size for eval.")
    parser.add_argument('--seed', type=int, default=12345,
                        help="random seed for initialization")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--poly_m", default=16, type=int,
                        help="Total batch size for eval.")
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    set_seed(args)

    MODEL_CLASSES = {
        'bert': (BertConfig, BertTokenizer, BertModel)
    }
    ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]
    # intent_list = [
    #     "bookrestaurant",
    #     "requestride",
    #     "getplacedetails",
    #     "getdirections",
    #     "compareplaces",
    #     "sharecurrentlocation",
    #     "searchplace",
    #     "shareeta",
    #     "getweather",
    #     "gettrafficinformation",
    # ]

    # test_examples = []
    # labels = []
    # with open("dev_clinc.txt", "r", encoding="utf") as f:
    #     for line in f:
    #         split = line.strip().split('\t')
    #         lbl, context, response = int(split[0]), split[1:-1], split[-1]
    #         test_examples.append(context[0])
    #         labels.append(response)

    # init dataset and bert model
    tokenizer = TokenizerClass.from_pretrained(os.path.join(
        args.bert_model, "bert/vocab.txt"), do_lower_case=True, clean_text=False)
    context_transform = SelectionJoinTransform(
        tokenizer=tokenizer, max_len=args.max_contexts_length)
    response_transform = SelectionSequentialTransform(
        tokenizer=tokenizer, max_len=args.max_response_length)
    val_dataset = SelectionDataset(os.path.join('dev_snips_small.txt'),
                                   context_transform, response_transform, sample_cnt=None, mode=args.architecture)

    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size,
                                collate_fn=val_dataset.batchify_join_str, shuffle=False, num_workers=0)
    log_wf = open(os.path.join(args.output_dir, 'log.txt'),
                  'a', encoding='utf-8')

    state_save_path = os.path.join(args.output_dir, 'pytorch_model.bin')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################################
    # build BERT encoder
    ########################################
    bert_config = ConfigClass.from_json_file(
        os.path.join(args.bert_model, 'bert/config.json'))
    model_state_dict = None
    if args.use_pretrain:
        bert_model_file = os.path.join(
            args.bert_model, "bert/pytorch_model.bin")
        resp_bert_model_file = os.path.join(
            args.bert_model, "resp_bert/pytorch_model.bin")
        print('Loading parameters from', bert_model_file)
        log_wf.write('Loading parameters from %s' % bert_model_file + '\n')
        bert_state_dict = torch.load(bert_model_file, map_location="cpu")
        resp_bert_state_dict = torch.load(
            resp_bert_model_file, map_location="cpu")
        bert = BertModelClass.from_pretrained(
            args.bert_model+"/bert", state_dict=bert_state_dict)
        resp_bert = BertModelClass.from_pretrained(
            args.bert_model+"/resp_bert", state_dict=resp_bert_state_dict)
        del model_state_dict
        # print(bert)
    else:
        bert = BertModelClass(bert_config)
        resp_bert = BertModelClass(bert_config)

    if args.architecture == 'bi':
        model = BiEncoder(bert_config, bert=bert, resp_bert=resp_bert)
    elif args.architecture == 'poly':
        model = PolyEncoder(bert_config, bert=bert,
                            resp_bert=resp_bert, poly_m=args.poly_m)
    else:
        raise Exception('Unknown architecture.')
    model.to(device)
    import time
    start = time.time()
    cand_result = eval_running_model(val_dataloader, model, mode="candidate")
    print(cand_result)
    print(f"time taken : {time.time() - start}")
    # cont_result = eval_running_model(context_dataloader, model, mode="context")
    # index = 0
    # hit = 0
    # print(f"time taken : {time.time() - start}")
    # print(len(cand_result))
    # print(len(cont_result))
    # for context_embedding in cont_result[:1]:
    #     result = []
    #     # print("===================")
    #     for response_embedding in cand_result[:2]:
    #         try:
    #             result.append(F.cosine_similarity(
    #                 context_embedding, response_embedding))
    #         except Exception as e:
    #             print(e)
    #     print(result)
    #     if np.argmax(result) == intent_list.index(labels[index]):
    #         hit += 1
    #     index += 1
    # print(f"time taken : {time.time() - start}")
    # print(
    #     f"Total Test Data : {len(cont_result)}\nhit : {hit}\nAccuracy : {hit/len(cont_result)}")
