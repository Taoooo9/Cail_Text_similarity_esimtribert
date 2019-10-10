import torch.nn as nn
import torch
import torch.nn.functional as F


def tri_loss(logit_p, logit_h, config):
    p = config.p
    margin = config.margin
    count = 0
    loss_sum = 0
    logit_p = torch.split(logit_p, 2, 0)
    logit_h = torch.split(logit_h, 2, 0)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=p)
    for i in range(len(logit_p)):
        anchor = torch.add(logit_p[i][0], logit_p[i][1])
        anchor = torch.div(anchor, 2)
        positive = logit_h[i][0]
        negative = logit_h[i][1]
        anchor = torch.unsqueeze(anchor, 0)
        positive = torch.unsqueeze(positive, 0)
        negative = torch.unsqueeze(negative, 0)
        loss = triplet_loss(anchor, positive, negative)
        d1 = l_norm(p, anchor, positive)
        d2 = l_norm(p, anchor, negative)
        if d1 < d2:
            count += 1
        loss_sum += loss
    last_loss = loss_sum / len(logit_p)
    return last_loss, count


def l_norm(p, x, y):
    return torch.norm_except_dim(x - y, p)


def ln_activation_p(x, n, epsilon):
    return -torch.log(-torch.div(x, n) + 1 + epsilon)


def ln_activation_n(x, n, epsilon):
    return -torch.log(-torch.div(n - x, n + 0.01) + 1 + epsilon)


def M_Loss(d1, d2, margin):
    return torch.max(d1 - d2 + margin, 0)


def class_loss(logit, gold):
    m = nn.NLLLoss()
    loss = m(logit, gold)
    correct = (torch.max(logit, 1)[1].view(gold.size()).data == gold.data).sum()
    return loss, correct


def less_loss(d1, d2, n, epsilon):
    pos_loss = ln_activation_p(d1, n, epsilon)
    neg_loss = ln_activation_n(d2, n, epsilon)
    loss = pos_loss + neg_loss
    return loss


def less_triplet_loss(logit_p, logit_h, config):
    p = config.p
    count = 0
    loss_sum = 0
    epsilon = config.epsilon
    logit_p = torch.split(logit_p, 2, 0)
    logit_h = torch.split(logit_h, 2, 0)
    n = 1536
    for i in range(len(logit_p)):
        anchor = torch.add(logit_p[i][0], logit_p[i][1])
        anchor = torch.div(anchor, 2)
        positive = logit_h[i][0]
        negative = logit_h[i][1]
        anchor = torch.unsqueeze(anchor, 0)
        positive = torch.unsqueeze(positive, 0)
        negative = torch.unsqueeze(negative, 0)
        d1 = l_norm(p, anchor, positive)
        d2 = l_norm(p, anchor, negative)
        loss = less_loss(d1, d2, n, epsilon)
        loss_sum += loss
        if d1 < d2:
            count += 1
    avg_loss = loss_sum / len(logit_p)
    return avg_loss, count





