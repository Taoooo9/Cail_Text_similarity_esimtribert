import torch
import torch.nn as nn

from transformers import BertForSequenceClassification, BertConfig
from esim.layers import SoftmaxAttention
from esim.utils import replace_masked


class MyBertModel(nn.Module):

    def __init__(self, config):
        super(MyBertModel, self).__init__()
        self.config = config
        self.bert_config = BertConfig.from_pretrained('Data/ms/.')
        self.bert = BertForSequenceClassification.from_pretrained('Data/ms/.')
        self._attention = SoftmaxAttention()
        self._projection = nn.Sequential(nn.Linear(4*self.bert_config.hidden_size,
                                                   self.bert_config.hidden_size),
                                         nn.ReLU())
        self._cat_projection = nn.Sequential(nn.Linear(3 * 4 * self.bert_config.hidden_size,
                                                       self.config.hidden_size * 2),
                                             nn.ReLU())
        self._classification = nn.Sequential(nn.Dropout(p=self.config.dropout),
                                             nn.Linear(2 * 4 * self.bert_config.hidden_size,
                                                       self.bert_config.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.config.dropout),
                                             nn.Linear(self.bert_config.hidden_size,
                                                       self.config.class_num))
        self.sfm = nn.LogSoftmax(dim=1)
        print(self.bert_config)

    def forward(self, src_premise_matrix, src_hypothesis_matrix, p_mask, h_mask):
        _, p_hidden_states = self.bert(src_premise_matrix, attention_mask=p_mask)
        _, h_hidden_states = self.bert(src_hypothesis_matrix, attention_mask=h_mask)
        if self.bert_config.num_hidden_layers in [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]:
            p_last_hidden = p_hidden_states[-1]
            h_last_hidden = h_hidden_states[-1]
        else:
            if self.bert_config.num_hidden_layers == 3:
                p_last_hidden1 = p_hidden_states[0]
                p_last_hidden2 = p_hidden_states[1]
                p_last_hidden3 = p_hidden_states[2]
                h_last_hidden1 = h_hidden_states[0]
                h_last_hidden2 = h_hidden_states[1]
                h_last_hidden3 = h_hidden_states[2]
            elif self.bert_config.num_hidden_layers == 9:
                p_last_hidden1 = p_hidden_states[2]
                p_last_hidden2 = p_hidden_states[5]
                p_last_hidden3 = p_hidden_states[8]
                h_last_hidden1 = h_hidden_states[2]
                h_last_hidden2 = h_hidden_states[5]
                h_last_hidden3 = h_hidden_states[8]
            else:
                p_last_hidden1 = p_hidden_states[2]
                p_last_hidden2 = p_hidden_states[7]
                p_last_hidden3 = p_hidden_states[11]
                h_last_hidden1 = h_hidden_states[2]
                h_last_hidden2 = h_hidden_states[7]
                h_last_hidden3 = h_hidden_states[12]

            if self.config.hidden_style == 'mean':
                p_last_hidden = p_last_hidden1 + p_last_hidden2 + p_last_hidden3
                p_last_hidden = torch.div(p_last_hidden, 3)
                h_last_hidden = h_last_hidden1 + h_last_hidden2 + h_last_hidden3
                h_last_hidden = torch.div(h_last_hidden, 3)

            elif self.config.hidden_style == 'add':
                p_last_hidden = p_last_hidden1 + p_last_hidden2 + p_last_hidden3
                h_last_hidden = h_last_hidden1 + h_last_hidden2 + h_last_hidden3
            elif self.config.hidden_style == 'single':
                p_last_hidden = p_last_hidden3
                h_last_hidden = h_last_hidden3
            else:
                p_last_hidden = torch.cat([p_last_hidden1, p_last_hidden2, p_last_hidden3], dim=-1)
                h_last_hidden = torch.cat([h_last_hidden1, h_last_hidden2, h_last_hidden3], dim=-1)

        attended_premises, attended_hypotheses = self._attention(p_last_hidden, p_mask,
                                                                 h_last_hidden, h_mask)
        enhanced_premises = torch.cat([p_last_hidden,
                                       attended_premises,
                                       p_last_hidden - attended_premises,
                                       p_last_hidden * attended_premises],
                                       dim=-1)

        enhanced_hypotheses = torch.cat([h_last_hidden,
                                       attended_hypotheses,
                                       h_last_hidden - attended_hypotheses,
                                       h_last_hidden * attended_hypotheses],
                                       dim=-1)

        if self.config.hidden_style == 'cat':
            enhanced_premises = self._cat_projection(enhanced_premises)
            enhanced_hypotheses = self._cat_projection(enhanced_hypotheses)
        else:
            enhanced_premises = self._projection(enhanced_premises)
            enhanced_hypotheses = self._projection(enhanced_hypotheses)

        v_p_avg = torch.sum(enhanced_premises * p_mask.unsqueeze(1).transpose(2, 1), dim=1) \
                  / torch.sum(p_mask, dim=1, keepdim=True)
        v_h_avg = torch.sum(enhanced_hypotheses * h_mask.unsqueeze(1).transpose(2, 1), dim=1) \
                  / torch.sum(h_mask, dim=1, keepdim=True)
        v_p_max, _ = replace_masked(enhanced_premises, p_mask, -1e7).max(dim=1)
        v_h_max, _ = replace_masked(enhanced_hypotheses, h_mask, -1e7).max(dim=1)

        v_p = torch.cat((v_p_avg, v_p_max), dim=1)
        v_h = torch.cat((v_h_avg, v_h_max), dim=1)

        return v_p, v_h