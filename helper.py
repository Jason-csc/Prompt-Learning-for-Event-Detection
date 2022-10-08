import json
import re
import tqdm
import sys
import numpy as np
import torch

from openprompt.data_utils import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from openprompt.plms import load_plm


def save_div(a, b):
    if b != 0:
        return a / b 
    else:
        return 0.0


def evaluation(gold_labels, pred_labels, vocab):
    inv_vocab = {v:k for k,v in vocab.items()}
    result = {}
    for label, idx in vocab.items():
        if idx != 0:
            result[label] = {"prec": 0.0, "rec": 0.0, "f1": 0.0}
    
    total_pred_num, total_gold_num, total_correct_num = 0.0, 0.0, 0.0

    for i in range(len(gold_labels)):
        pred_labels_i = pred_labels[i]
        gold_labels_i = gold_labels[i]

        for idx in gold_labels_i:
            if idx != 0:
                total_gold_num += 1
                result[inv_vocab[idx]]["rec"] += 1
        
        for idx in pred_labels_i:
            if idx != 0:
                total_pred_num += 1
                result[inv_vocab[idx]]["prec"] += 1
        
                if idx in gold_labels_i:
                    total_correct_num += 1
                    result[inv_vocab[idx]]["f1"] += 1
    
    for label in result:
        counts = result[label]
        counts["prec"] = save_div(counts["f1"], counts["prec"])
        counts["rec"] = save_div(counts["f1"], counts["rec"])
        counts["f1"] = save_div(2*counts["prec"]*counts["rec"], counts["prec"]+counts["rec"])
    
    prec = save_div(total_correct_num, total_pred_num)
    rec = save_div(total_correct_num, total_gold_num)
    f1 = save_div(2*prec*rec, prec+rec)

    return prec, rec, f1, result


def load_data(input_dir):
    data_items = []

    with open(input_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        item = json.loads(line)
        data_items.append(item)
    return data_items


def get_vocab(train_dir, valid_dir):
    train_data = load_data(train_dir)
    valid_data = load_data(valid_dir)
    vocab = {"None": 0}

    for item in train_data:
        for event in item["events"]:
            if event[-1] not in vocab:
                vocab[event[-1]] = len(vocab)
    
    for item in valid_data:
        for event in item["events"]:
            if event[-1] not in vocab:
                vocab[event[-1]] = len(vocab)
    
    return vocab 


def process_data(input_dir, vocab):
    items = load_data(input_dir)
    output_items = []

    for i,item in enumerate(items):
        labels = []

        for event in item["events"]:
            if vocab[event[-1]] not in labels:
                labels.append(vocab[event[-1]])

        if len(labels) == 0:
            labels.append(0)

        input_example = InputExample(text_a=" ".join(item["tokens"]), label=labels, guid=i)
        output_items.append(input_example)

    return output_items


def my_collate_fn(batch):
    elem = batch[0]
    return_dict = {}
    for key in elem:
        if key == "encoded_tgt_text":
            return_dict[key] = [d[key] for d in batch]
        else:
            try:
                return_dict[key] = default_collate([d[key] for d in batch])
            except:
                return_dict[key] = [d[key] for d in batch]

    return InputFeatures(**return_dict)


def convert_labels_to_list(labels):
    label_list = []
    for label in labels:
        label_list.append(label.tolist().copy())
    return label_list
    

def to_device(data, device):
    for key in ["input_ids", "attention_mask", "decoder_input_ids", "loss_ids"]:
        if key in data:
            data[key] = data[key].to(device)
    return data


def get_plm(model_name):
    # You may change the PLM you'd like to use. Currently it's t5-base.
    # return load_plm("t5", "google/t5-v1_1-base")
    return load_plm(model_name[0],model_name[1])


def get_template():
    # You may design your own template here. 
    return '{"placeholder":"text_a"} The previous text describes the {"mask"} event.'


def get_verbalizer(vocab):
    # Input: a dictionary for event types to indices: e.g.: {'None': 0, 'Catastrophe': 1, 'Causation': 2, 'Motion': 3, 'Hostile_encounter': 4, 'Process_start': 5, 'Attack': 6, 'Killing': 7, 'Conquering': 8, 'Social_event': 9, 'Competition': 10}

    # Output: A 2-dim list. Verbalizers for each event type. Currently this function directly returns the lowercase for each event type name (and the performance is low). You may want to design your own verbalizers to improve the performance.

    return [[label.lower().replace('_',' ')] for label in vocab]


def loss_func(logits, labels):
    
    # INPUT: 
    ##  logits: a torch.Tensor of (batch_size, number_of_event_types) which is the output logits for each event type (none types are included).
    ##  labels: a 2-dim List which denotes the ground-truth labels for each sentence in the batch. Note that there cound be multiple events, a single event, or no events for each sentence. 
    ##  For example, if labels == [[0], [2,3,4]] then the batch size is 2 and the first sentence has no events, and the second sentence has three events of indices 2,3 and 4.

    ##  INSTRUCTIONS: In general, we want to maximize the logits of correct labels and minimize the logits with incorrect labels. You can implement your own loss function here or you can refer to what loss function is used in https://arxiv.org/pdf/2202.07615.pdf

    ## OUTPUT:
    ##   The output should be a pytorch scalar --- the loss.

    ###  YOU NEED TO WRITE YOUR CODE HERE.  ###

    device = logits.get_device()
    pos_loss = torch.tensor(0.0).to(device)
    neg_loss = torch.tensor(0.0).to(device)
    batch_size, num_event = logits.shape
    for i in range(batch_size):
        label = labels[i]
        logit = logits[i]
        pos_tmp = torch.tensor(0.0).to(device)
        neg_tmp = torch.tensor(0.0).to(device)
        if 0 not in label:
            for l in label:
                pos_tmp += torch.log(torch.exp(logit[l])/(torch.exp(logit[l])+torch.exp(logit[0])))
            pos_tmp /= len(label)
        neg_index = [ind for ind in range(0,num_event) if not ind in label or ind == 0]
        neg_logit = torch.exp(logit[neg_index])
        neg_tmp += torch.log(neg_logit[0]/torch.sum(neg_logit))
        pos_loss += pos_tmp
        neg_loss += neg_tmp
    loss = (pos_loss+neg_loss)/batch_size    
    return -loss


def predict(logits):
    # INPUT: 
    ##  logits: a torch.Tensor of (batch_size, number_of_event_types) which is the output logits for each event type (none types are included).
    # OUTPUT:
    ##  a 2-dim list which has the same format with the "labels" in "loss_func" --- the predictions for all the sentences in the batch.
    ##  For example, if predictions == [[0], [2,3,4]] then the batch size is 2, and we predict no events for the first sentence and three events (2,3,and 4) for the second sentence.

    ##  INSTRUCTIONS: The most straight-forward way for prediction is to select out the indices with maximum of logits. Note that this is a multi-label classification problem, so each sentence could have multiple predicted event indices. Using what threshold for prediction is important here. You can also use the None event (index 0) as the threshold as what https://arxiv.org/pdf/2202.07615.pdf does.

    ###  YOU NEED TO WRITE YOUR CODE HERE.  ###
    predictions = []
    batch_size, num_event = logits.shape
    for logit in logits:
        pred = [i for i in range(1,num_event) if logit[i] > logit[0]]
        pred = pred if len(pred) > 0 else [0]
        predictions.append(pred)
    return predictions
