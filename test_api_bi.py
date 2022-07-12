# flake8: noqa
import json
import requests
import numpy as np
import os

for td_name in [
    "snips_full_training_pos_for_neg.json",
    # "bltd_datasets/bltd_banking_train.json",
    "snips_full_training_pos.json",
    # "banking_half_train.json",
]:
    with open(td_name, "r") as f:
        train_data = json.load(f)
    if td_name in [
        "snips_full_training_pos.json",
        "snips_full_training_pos_for_neg.json",
    ]:
        with open("snips_full_testing_pos.json", "r") as f:
            test_data = json.load(f)
        print("loaded snips testing data")
    else:
        with open("bltd_datasets/bltd_sgd_test.json", "r") as f:
            test_data = json.load(f)
    for loss_type in ["dot"]:
        for shared in [True]:
            for cand_model_name in ["prajjwal1/bert-tiny"]:
                for cont_model_name in ["prajjwal1/bert-tiny"]:
                    # if cont_model_name == "prajjwal1/bert-tiny":
                    #     cand_model_name = "bert-base-uncased"
                    #     cont_model_name = "prajjwal1/bert-tiny"
                    # else:
                    #     cand_model_name = "prajjwal1/bert-tiny"
                    #     cont_model_name = "bert-base-uncased"
                    config_data = {
                        "model_parameters": {
                            "cand_model_name": cand_model_name,
                            "cont_model_name": cont_model_name,
                            "shared": shared,
                            "loss_type": loss_type,
                            "loss_function": "mse",
                        }
                    }
                    for lr in [0.01, 0.001]:
                        r = requests.post(
                            "http://127.0.0.1:8000/set_model_config", json=config_data
                        )
                        if td_name in ["snips_full_training_pos_for_neg.json"]:
                            train_data["training_parameters"] = {
                                "num_train_epochs": 50,
                                "learning_rate": lr,
                                "train_with_neg": True,
                            }
                        else:
                            train_data["training_parameters"] = {
                                "num_train_epochs": 50,
                                "learning_rate": lr,
                                "train_with_neg": False,
                            }
                        r = requests.post(
                            "http://127.0.0.1:8000/train", json=train_data
                        )
                        if cont_model_name == "prajjwal1/bert-tiny":
                            model2 = "tiny"
                        else:
                            model2 = "bert_base"
                        if cand_model_name == "prajjwal1/bert-tiny":
                            model1 = "tiny"
                        else:
                            model1 = "bert_base"
                        if td_name in [
                            "snips_full_training_pos_for_neg.json",
                            "snips_full_training_pos.json",
                        ]:
                            if td_name == "snips_full_training_pos_for_neg.json":
                                log_name_new = os.path.join(
                                    "logoutput",
                                    model1
                                    + "_"
                                    + model2
                                    + "_"
                                    + str(lr)
                                    + "_"
                                    + loss_type
                                    + "_"
                                    + str(shared)
                                    + "_snips_neg"
                                    + "_log.txt",
                                )
                                p_file = os.path.join(
                                    "logoutput",
                                    model1
                                    + "_"
                                    + model2
                                    + "_"
                                    + str(lr)
                                    + "_"
                                    + loss_type
                                    + "_"
                                    + str(shared)
                                    + "_snips_neg"
                                    + "_pred.json",
                                )
                            else:
                                log_name_new = os.path.join(
                                    "logoutput",
                                    model1
                                    + "_"
                                    + model2
                                    + "_"
                                    + str(lr)
                                    + "_"
                                    + loss_type
                                    + "_"
                                    + str(shared)
                                    + "_snips_pos"
                                    + "_log.txt",
                                )
                                p_file = os.path.join(
                                    "logoutput",
                                    model1
                                    + "_"
                                    + model2
                                    + "_"
                                    + str(lr)
                                    + "_"
                                    + loss_type
                                    + "_"
                                    + str(shared)
                                    + "_snips_pos"
                                    + "_pred.json",
                                )
                        else:
                            if loss_type == "dot":
                                log_name_new = os.path.join(
                                    "logoutput",
                                    model1
                                    + "_"
                                    + model2
                                    + "_"
                                    + str(lr)
                                    + "_"
                                    + str(shared)
                                    + "_"
                                    + str(loss_type)
                                    + "_bltd_sgd_pos"
                                    + "_log.txt",
                                )
                                p_file = os.path.join(
                                    "logoutput",
                                    model1
                                    + "_"
                                    + model2
                                    + "_"
                                    + str(lr)
                                    + "_"
                                    + str(shared)
                                    + "_"
                                    + str(loss_type)
                                    + "_bltd_sgd_pos"
                                    + "_pred.json",
                                )
                            else:
                                log_name_new = os.path.join(
                                    "logoutput",
                                    model1
                                    + "_"
                                    + model2
                                    + "_"
                                    + str(lr)
                                    + "_"
                                    + str(shared)
                                    + "_"
                                    + str(loss_type)
                                    + "_bltd_sgd_neg"
                                    + "_log.txt",
                                )
                                p_file = os.path.join(
                                    "logoutput",
                                    model1
                                    + "_"
                                    + model2
                                    + "_"
                                    + str(lr)
                                    + "_"
                                    + str(shared)
                                    + "_"
                                    + str(loss_type)
                                    + "_bltd_sgd_neg"
                                    + "_pred.json",
                                )
                        print(
                            f"train complete for  : {model1+ '_'+ model2+'_'+str(shared)+'_'+str(td_name)}\n"
                        )
                        log_file_old = os.path.join("logoutput", "log.txt")
                        os.rename(log_file_old, log_name_new)
                        r = requests.post("http://127.0.0.1:8000/infer", json=test_data)
                        p_cand = []
                        for d in r.json():
                            p_cand.append(d["candidate"][np.argmax(d["score"])])
                        # print(p_cand)
                        with open(p_file, "w") as f:
                            f.write(json.dumps(p_cand))
                        print(
                            f"test complete for  : {model1+ '_'+ model2+'_'+str(shared)+'_'+str(td_name)}\n"
                        )
