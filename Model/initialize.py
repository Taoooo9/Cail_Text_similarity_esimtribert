import torch.nn as nn
import numpy as np


def init_embedding(input_embedding):
    """
    初始化embedding层权重
    """
    scope = np.sqrt(3.0 / input_embedding.weight.size(1))
    nn.init.uniform_(input_embedding.weight, -scope, scope)


def init_lstm_weight(lstm, num_layer=1):
    """
    初始化lstm权重
    """
    for i in range(num_layer):
        weight_h = getattr(lstm, 'weight_hh_l{0}'.format(i))
        scope = np.sqrt(6.0 / (weight_h.size(0) / 4. + weight_h.size(1)))
        nn.init.uniform_(getattr(lstm, 'weight_hh_l{0}'.format(i)), -scope, scope)

        weight_i = getattr(lstm, 'weight_ih_l{0}'.format(i))
        scope = np.sqrt(6.0 / (weight_i.size(0) / 4. + weight_i.size(1)))
        nn.init.uniform_(getattr(lstm, 'weight_ih_l{0}'.format(i)), -scope, scope)

    if lstm.bias:
        for i in range(num_layer):
            weight_i = getattr(lstm, 'bias_ih_l{0}'.format(i))
            weight_i.data.zero_()
            weight_i.data[lstm.hidden_size:2 * lstm.hidden_size] = 1

            weight_h = getattr(lstm, 'bias_hh_l{0}'.format(i))
            weight_h.data.zero_()
            weight_h.data[lstm.hidden_size:2 * lstm.hidden_size] = 1


def init_linear(input_linear):
    """
    初始化全连接层权重
    """
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -scope, scope)

    if input_linear.bias is not None:
        scope = np.sqrt(6.0 / (input_linear.bias.size(0) + 1))
        input_linear.bias.data.uniform_(-scope, scope)
