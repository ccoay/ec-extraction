import sys, os, warnings, time
sys.path.append('..')
warnings.filterwarnings("ignore")
import numpy as np
import torch
from config import *
from data_loader import *
from networks.rank_cp import *
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *


def main(configs, fold_id):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    train_loader = build_train_data(configs, fold_id=fold_id)
    if configs.split == 'split20':
        valid_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    model = Network(configs).to(DEVICE)

    params = model.parameters()
    params_bert = model.bert.parameters()
    params_rest = list(model.seq2mat.parameters()) + list(model.mdrnn.parameters()) + list(model.pred.parameters()) + list(model.tab_pred.parameters()) \
    + list(model.reduce.parameters())
    assert sum([param.nelement() for param in params]) == \
           sum([param.nelement() for param in params_bert]) + sum([param.nelement() for param in params_rest])

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': configs.l2_bert, 'eps': configs.adam_epsilon, 'lr': configs.lr_bert},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'eps': configs.adam_epsilon, 'lr': configs.lr_bert},
        {'params': params_rest,
         'weight_decay': configs.l2}
    ]
    optimizer = AdamW(params, lr=configs.lr)

    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)

    model.zero_grad()
    max_ec, max_e, max_c = (-1, -1, -1), None, None
    metric_ec, metric_e, metric_c = (-1, -1, -1), None, None
    early_stop_flag = None
    
    for epoch in range(1, configs.epochs+1):
        epoch_loss = 0
        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            y_table, doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch
            
            couples_pred, pred_e, pred_c = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                              bert_clause_b, doc_len_b, adj_b, y_mask_b)
            
            loss_t, loss_e, loss_c = model.loss_pre(couples_pred, pred_e, pred_c, y_emotions_b, y_causes_b, y_table, y_mask_b)
            
            loss = loss_t + loss_e + loss_c
#             loss = loss_t
            
            loss = loss / configs.gradient_accumulation_steps
            epoch_loss += loss
            loss.backward()
            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
        print(epoch_loss)
        with torch.no_grad():
            model.eval()

            if configs.split == 'split10':
                test_ec, test_e, test_c, _, _, _ = inference_one_epoch(configs, test_loader, model)
                print("F1 is {}, P is {}, R is {}.".format(test_ec[2],test_ec[0],test_ec[1]))
                if test_ec[2] > metric_ec[2]:
                    early_stop_flag = 1
                    metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                else:
                    early_stop_flag += 1

            if configs.split == 'split20':
                valid_ec, valid_e, valid_c, _, _, _ = inference_one_epoch(configs, valid_loader, model)
                test_ec, test_e, test_c, _, _, _ = inference_one_epoch(configs, test_loader, model)
                if valid_ec[2] > max_ec[2]:
                    early_stop_flag = 1
                    max_ec, max_e, max_c = valid_ec, valid_e, valid_c
                    metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                else:
                    early_stop_flag += 1

        if epoch > configs.epochs / 2 and early_stop_flag >= 5:
            break
    return metric_ec, metric_e, metric_c


def inference_one_batch(configs, batch, model):
    y_table, doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b = batch

    couples_pred, pred_e, pred_c = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                      bert_clause_b, doc_len_b, adj_b, y_mask_b)
    _, pred = torch.max(couples_pred, 3)
    
    pred_couples = get_pred_doc_couples(pred, doc_len_b)
    
    loss_t, loss_e, loss_c = model.loss_pre(couples_pred, pred_e, pred_c, y_emotions_b, y_causes_b, y_table, y_mask_b)

    return to_np(loss_t), to_np(loss_e), to_np(loss_c), \
           doc_couples_b, pred_couples, doc_id_b


def inference_one_epoch(configs, batches, model):
    doc_id_all, doc_couples_all, doc_couples_pred_all = [], [], []
    for batch in batches:
        
        _, _, _, doc_couples, doc_couples_pred, doc_id_b = inference_one_batch(configs, batch, model)
        doc_id_all.extend(doc_id_b)
        doc_couples_all.extend(doc_couples)
        doc_couples_pred_all.extend(doc_couples_pred)

    
    metric_ec, metric_e, metric_c = eval_func(doc_couples_all, doc_couples_pred_all)
    return metric_ec, metric_e, metric_c, doc_id_all, doc_couples_all, doc_couples_pred_all


def get_pred_doc_couples(pred, doc_len_b):
    res = []
    for index in range(len(doc_len_b)):
        b = pred[index,:,:]
        t_res = []
        for i in range(doc_len_b[index]):
            for j in range(doc_len_b[index]):
                if int(b[i,j].item()) == 1:
                       t_res.append([i+1,j+1])
                        
        res.append(t_res)
    return res

if __name__ == '__main__':
    configs = Config()

    if configs.split == 'split10':
        n_folds = 10
        configs.epochs = 20
    elif configs.split == 'split20':
        n_folds = 20
        configs.epochs = 15
    else:
        print('Unknown data split.')
        exit()

    metric_folds = {'ecp': [], 'emo': [], 'cau': []}
    for fold_id in range(1, n_folds+1):
        print('===== fold {} ====='.format(fold_id))
        metric_ec, metric_e, metric_c = main(configs, fold_id)
        print('F_ecp: {}'.format(float_n(metric_ec[2])))

        metric_folds['ecp'].append(metric_ec)
        metric_folds['emo'].append(metric_e)
        metric_folds['cau'].append(metric_c)

    metric_ec = np.mean(np.array(metric_folds['ecp']), axis=0).tolist()
    metric_e = np.mean(np.array(metric_folds['emo']), axis=0).tolist()
    metric_c = np.mean(np.array(metric_folds['cau']), axis=0).tolist()
    print('===== Average =====')
    print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_ec[2]), float_n(metric_ec[0]), float_n(metric_ec[1])))
    print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
    print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))
    write_b({'ecp': metric_ec, 'emo': metric_e, 'cau': metric_c}, '{}_{}_metrics.pkl'.format(time.time(), configs.split))

