import os
from sklearn.linear_model import LogisticRegression
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from model import SelectionDataset, SelectionSequentialTransform, SelectionJoinTransform, EvalDataset
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from model.encoder import BiEncoder
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
max_contexts_length, max_candidate_length = 128, 64


def get_embeddings(model, tokenizer, text_data, embed_type="context"):
    global max_history, max_contexts_length, max_candidate_length, device
    if embed_type == "context":
        context_transform = SelectionJoinTransform(
            tokenizer=tokenizer,
            max_len=max_contexts_length)
        context_data = EvalDataset(
            text_data, context_transform=context_transform, response_transform=None, mode=embed_type)
    else:
        candidate_transform = SelectionSequentialTransform(
            tokenizer=tokenizer,
            max_len=max_candidate_length)
        context_data = EvalDataset(
            text_data, context_transform=None, response_transform=candidate_transform, mode=embed_type)
    dataloader = DataLoader(context_data, batch_size=1,
                            collate_fn=context_data.eval_str, shuffle=False, num_workers=0)
    for step, batch in enumerate(dataloader, start=1):
        batch = tuple(t.to(device) for t in batch)
        token_ids_list_batch,  input_masks_list_batch = batch

        with torch.no_grad():
            if embed_type == "context":
                embeddings = model(context_input_ids=token_ids_list_batch,
                                   context_input_masks=input_masks_list_batch, get_embedding=embed_type, mode="eval")
            else:
                embeddings = model(candidate_input_ids=token_ids_list_batch,
                                   candidate_input_masks=input_masks_list_batch, get_embedding=embed_type, mode="eval")
                embeddings = embeddings.squeeze(0)

    return embeddings.squeeze(0).detach().tolist()


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
        print(logits)
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

    parser.add_argument("--eval_batch_size", default=1,
                        type=int, help="Total batch size for eval.")
    parser.add_argument('--seed', type=int, default=12345,
                        help="random seed for initialization")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    set_seed(args)

    MODEL_CLASSES = {
        'bert': (BertConfig, BertTokenizer, BertModel)
    }
    ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]
    intent_list = [
        "restaurant_reviews",
        "nutrition_info",
        "account_blocked",
        "oil_change_how",
        "time",
        "weather",
        "redeem_rewards",
        "interest_rate",
        "gas_type",
        "accept_reservations",
        "smart_home",
        "user_name",
        "report_lost_card",
        "repeat",
        "whisper_mode",
        "what_are_your_hobbies",
        "order",
        "jump_start",
        "schedule_meeting",
        "meeting_schedule",
        "freeze_account",
        "what_song",
        "meaning_of_life",
        "restaurant_reservation",
        "traffic",
        "make_call",
        "text",
        "bill_balance",
        "improve_credit_score",
        "change_language",
        "no",
        "measurement_conversion",
        "timer",
        "flip_coin",
        "do_you_have_pets",
        "balance",
        "tell_joke",
        "last_maintenance",
        "exchange_rate",
        "uber",
        "car_rental",
        "credit_limit",
        "oos",
        "shopping_list",
        "expiration_date",
        "routing",
        "meal_suggestion",
        "tire_change",
        "todo_list",
        "card_declined",
        "rewards_balance",
        "change_accent",
        "vaccines",
        "reminder_update",
        "food_last",
        "change_ai_name",
        "bill_due",
        "who_do_you_work_for",
        "share_location",
        "international_visa",
        "calendar",
        "translate",
        "carry_on",
        "book_flight",
        "insurance_change",
        "todo_list_update",
        "timezone",
        "cancel_reservation",
        "transactions",
        "credit_score",
        "report_fraud",
        "spending_history",
        "directions",
        "spelling",
        "insurance",
        "what_is_your_name",
        "reminder",
        "where_are_you_from",
        "distance",
        "payday",
        "flight_status",
        "find_phone",
        "greeting",
        "alarm",
        "order_status",
        "confirm_reservation",
        "cook_time",
        "damaged_card",
        "reset_settings",
        "pin_change",
        "replacement_card_duration",
        "new_card",
        "roll_dice",
        "income",
        "taxes",
        "date",
        "who_made_you",
        "pto_request",
        "tire_pressure",
        "how_old_are_you",
        "rollover_401k",
        "pto_request_status",
        "how_busy",
        "application_status",
        "recipe",
        "calendar_update",
        "play_music",
        "yes",
        "direct_deposit",
        "credit_limit_change",
        "gas",
        "pay_bill",
        "ingredients_list",
        "lost_luggage",
        "goodbye",
        "what_can_i_ask_you",
        "book_hotel",
        "are_you_a_bot",
        "next_song",
        "change_speed",
        "plug_type",
        "maybe",
        "w2",
        "oil_change_when",
        "thank_you",
        "shopping_list_update",
        "pto_balance",
        "order_checks",
        "travel_alert",
        "fun_fact",
        "sync_device",
        "schedule_maintenance",
        "apr",
        "transfer",
        "ingredient_substitution",
        "calories",
        "current_location",
        "international_fees",
        "calculator",
        "definition",
        "next_holiday",
        "update_playlist",
        "mpg",
        "min_payment",
        "change_user_name",
        "restaurant_suggestion",
        "travel_notification",
        "cancel",
        "pto_used",
        "travel_suggestion",
        "change_volume"
    ]

    test_examples = []
    labels = []
    with open("test_clinc.txt", "r", encoding="utf") as f:
        for line in f:
            split = line.strip().split('\t')
            lbl, context, response = int(split[0]), split[1:-1], split[-1]
            test_examples.append(context[0])
            labels.append(response)

    # init dataset and bert model
    tokenizer = TokenizerClass.from_pretrained(os.path.join(
        args.bert_model, "bert/vocab.txt"), do_lower_case=True, clean_text=False)
    context_transform = SelectionJoinTransform(
        tokenizer=tokenizer, max_len=args.max_contexts_length)
    response_transform = SelectionSequentialTransform(
        tokenizer=tokenizer, max_len=args.max_response_length)
    # val_dataset = SelectionDataset(os.path.join('dev_snips_small.txt'),
    #                                context_transform, response_transform, sample_cnt=None, mode=args.architecture)

    # val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size,
    #                             collate_fn=val_dataset.batchify_join_str, shuffle=False, num_workers=0)
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
    else:
        bert = BertModelClass(bert_config)
        resp_bert = BertModelClass(bert_config)

    if args.architecture == 'bi':
        model = BiEncoder(bert_config, bert=bert, resp_bert=resp_bert)
    else:
        raise Exception('Unknown architecture.')
    model.to(device)
    import time
    start = time.time()
    # cand_result = eval_running_model(val_dataloader, model, mode="candidate")
    # print(cand_result)
    with open("test_clinc.txt", "r", encoding="utf") as f:
        for line in f:
            split = line.strip().split('\t')
            lbl, context, response = int(split[0]), split[1:-1], split[-1]
            test_examples.append(context[0])
            labels.append(response)
    cont_embed = []
    for example in test_examples:
        cont_embed.append(get_embeddings(
            model=model, tokenizer=tokenizer, text_data=example, embed_type="context"))
    print(len(cont_embed))
    #cand_embed = []
    #for intent in intent_list:
    #    cand_embed.append(get_embeddings(
    #        model=model, tokenizer=tokenizer, text_data=intent, embed_type="candidate"))
    # print(cand_embed)
    #index = 0
    #hit = 0
    #for context_embedding in cont_embed:
    #    result = []
    #    for response_embedding in cand_embed:
    #        result.append(np.dot(context_embedding, response_embedding) / (np.linalg.norm(context_embedding) *
    #                                                                       np.linalg.norm(response_embedding)))
    #    if np.argmax(result) == intent_list.index(labels[index]):
    #        hit += 1
    #    index += 1
    #print(f"time taken : {time.time() - start}")
    #print(
    #    f"Total Test Data : {len(cont_embed)}\nhit : {hit}\nAccuracy : {hit/len(cont_embed)}")
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
