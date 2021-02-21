import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from transformers import BertModel
from networks.seq2mat import CatReduce
from networks.mdrnns import *


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.ratio = configs.ratio
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.reduce = nn.Linear(configs.feat_dim, configs.reduced_dim)
        self.seq2mat = CatReduce(n_in=configs.reduced_dim, n_out=configs.reduced_dim)
        self.mdrnn=get_mdrnn_layer(configs, emb_dim = configs.reduced_dim, direction=configs.direction, norm='')
        
        self.pred = Pre_Predictions(configs)
        self.tab_pred = Tab_Predictions(configs)

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, doc_len, adj, y_mask_b):
        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),
                                attention_mask=bert_masks_b.to(DEVICE),
                                token_type_ids=bert_segment_b.to(DEVICE))
        doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE))
        reduced_doc_sents_h = self.reduce(doc_sents_h)
        X = self.seq2mat(reduced_doc_sents_h, reduced_doc_sents_h) # (B, N, N, H)
        X = F.relu(X, inplace=True)
        masks = torch.IntTensor(y_mask_b).to(DEVICE)
        masks = masks.unsqueeze(1) & masks.unsqueeze(2)
        T, _ = self.mdrnn(X, states=None, masks=masks)
        T_diagonal = torch.diagonal(T, offset=0, dim1=1, dim2=2)
        diagonal = T_diagonal.permute(0, 2, 1)
        
        pred_e, pred_c = self.pred(diagonal)
        couples_pred = self.tab_pred(T)
        
        return couples_pred, pred_e, pred_c

    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        return doc_sents_h

    
    def output_util(self, couples_pred, emo_cau_pos, doc_couples, y_mask, test=False):
        """
        TODO: combine this function to data_loader
        """
        batch, n_couple = couples_pred.size()

        couples_true, couples_mask = [], []
        doc_couples_pred = []
        for i in range(batch):
            y_mask_i = y_mask[i]
            max_doc_idx = sum(y_mask_i)

            doc_couples_i = doc_couples[i]
            couples_true_i = []
            couples_mask_i = []
            for couple_idx, emo_cau in enumerate(emo_cau_pos):
                if emo_cau[0] > max_doc_idx or emo_cau[1] > max_doc_idx:
                    couples_mask_i.append(0)
                    couples_true_i.append(0)
                else:
                    couples_mask_i.append(1)
                    couples_true_i.append(1 if emo_cau in doc_couples_i else 0)

            couples_pred_i = couples_pred[i]
            doc_couples_pred_i = []
            if test:
                if torch.sum(torch.isnan(couples_pred_i)) > 0:
                    k_idx = [0] * 3
                else:
                    _, k_idx = torch.topk(couples_pred_i, k=3, dim=0)
                doc_couples_pred_i = [(emo_cau_pos[idx], couples_pred_i[idx].tolist()) for idx in k_idx]

            couples_true.append(couples_true_i)
            couples_mask.append(couples_mask_i)
            doc_couples_pred.append(doc_couples_pred_i)
        return couples_true, couples_mask, doc_couples_pred

    def loss_pre(self, couples_pred, pred_e, pred_c, y_emotions, y_causes, y_table, y_mask):
        
        y_mask = torch.FloatTensor(y_mask).to(DEVICE)
        matrix_mask_float = y_mask[:, None] * y_mask[:, :, None] # B, N, N
        matrix_mask_float = matrix_mask_float.to(DEVICE)
        
        y_emotions = torch.LongTensor(y_emotions).to(DEVICE)
        y_causes = torch.LongTensor(y_causes).to(DEVICE)
        y_table = torch.LongTensor(y_table).to(DEVICE)
        
        weighted_matrix_mask_float = matrix_mask_float + self.ratio * y_table
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        loss_c = criterion(pred_c.permute(0, 2, 1), y_causes)
        loss_c = (loss_c * y_mask).sum()
        
        loss_e = criterion(pred_e.permute(0, 2, 1), y_emotions)
        loss_e = (loss_e * y_mask).sum()
        
        loss_t = criterion(couples_pred.permute(0, -1, 1, 2), y_table)
        loss_t = (loss_t * weighted_matrix_mask_float).sum()
        
        return loss_t, loss_e, loss_c




class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        
        self.out_e = nn.Linear(configs.hidden_dim, 2)
        self.out_c = nn.Linear(configs.hidden_dim, 2)

    def forward(self, h):
        pred_e = self.out_e(h)
        pred_c = self.out_c(h)
        return pred_e, pred_c
    
class Tab_Predictions(nn.Module):
    def __init__(self, configs):
        super(Tab_Predictions, self).__init__()
        self.out = nn.Linear(configs.hidden_dim, 2)

    def forward(self, h):
        pred = self.out(h)
        return pred