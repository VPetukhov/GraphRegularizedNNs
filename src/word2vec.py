from tqdm.notebook import tqdm, trange
from collections import Counter, defaultdict
import random, math
import numpy as np
import pandas as pd

import torch
from torch import nn, autograd
import torch.optim as optim
import torch.nn.functional as F


def subsample_frequent_words(corpus, word_counts=None):
    filtered_corpus = []
    if word_counts is None:
        word_counts = dict(Counter(list(itertools.chain.from_iterable(corpus))))
    sum_word_counts = sum(list(word_counts.values()))
    word_counts = {w: cnt / float(sum_word_counts) for w,cnt in word_counts.items()}
    for text in tqdm(corpus):
        filtered_corpus.append([])
        for word in text:
            if random.random() < (1 + math.sqrt(word_counts[word] * 1e3)) * 1e-3 / float(word_counts[word]):
                filtered_corpus[-1].append(word)
    return filtered_corpus


def get_batches(context_tuple_list, batch_size=100): # DEPRECATED?
    random.shuffle(context_tuple_list)
    batches = []
    batch_target, batch_context, batch_negative = [], [], []
    for i,ctup in enumerate(context_tuple_list):
        batch_target.append(word_to_index[ctup[0]])
        batch_context.append(word_to_index[ctup[1]])
        if len(ctup) > 2:
            batch_negative.append([word_to_index[w] for w in ctup[2]])
        if (i+1) % batch_size == 0 or i == len(context_tuple_list)-1:
            tensor_target = autograd.Variable(torch.from_numpy(np.array(batch_target)).long())
            tensor_context = autograd.Variable(torch.from_numpy(np.array(batch_context)).long())
            if len(ctup) > 2:
                tensor_negative = autograd.Variable(torch.from_numpy(np.array(batch_negative)).long())
                batches.append((tensor_target, tensor_context, tensor_negative))
            else:
                batches.append((tensor_target, tensor_context))
            
            batch_target, batch_context, batch_negative = [], [], []
    return batches


def get_next_batch(sample_probability: pd.Series, targ_lst, cont_lst, batch_start: int, batch_size: int):
    neg_words = np.random.choice(sample_probability.index.values, size=(min(len(targ_lst) - batch_start, batch_size), 5), p=sample_probability.values)
    targ_lst = targ_lst[batch_start:(batch_start + batch_size)]
    cont_lst = cont_lst[batch_start:(batch_start + batch_size)]
    return [torch.from_numpy(x).long() for x in (targ_lst, cont_lst, neg_words)]


class Word2Vec(nn.Module):

    def __init__(self, embedding_size, vocab_size):
        super(Word2Vec, self).__init__()
        self.embeddings_target = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)

    def forward(self, target_word, context_word, negative_example):
        emb_target = self.embeddings_target(target_word)
        emb_context = self.embeddings_context(context_word)
        emb_product = torch.mul(emb_target, emb_context)
        emb_product = torch.sum(emb_product, dim=1)
        out = torch.sum(F.logsigmoid(emb_product))
        emb_negative = self.embeddings_context(negative_example)
        emb_product = torch.bmm(emb_negative, emb_target.unsqueeze(2))
        emb_product = torch.sum(emb_product, dim=1)
        out += torch.sum(F.logsigmoid(-emb_product))
        return -out