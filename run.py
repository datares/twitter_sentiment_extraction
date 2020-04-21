from preprocess import batchify, TEXT
import torch.nn as nn
import torchtext
from model import TransformerModel
from train import train_model, evaluate
import torch
import time
import math
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_sentiment_words():
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)

    ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
    batch_size = 20
    eval_batch_size = 10
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    criterion = nn.BCELoss()
    lr = 0.001 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 3
    best_model = None
    
    ds = pd.read_csv("train.csv", encoding = "ISO-8859-1", header=None)
    x, y = create_trainset(ds)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_model(x, y, model, criterion, optimizer, scheduler, epoch)
        # val_loss = evaluate(model,criterion, val_data)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #     'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                 val_loss, math.exp(val_loss)))
        # print('-' * 89)

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model = model

        # scheduler.step()

    # test_loss = evaluate(best_model, criterion, test_data)

    # print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    #     test_loss, math.exp(test_loss)))
    # print('=' * 89)

def create_trainset(ds):
    dic = {0: -1, 2: 0, 4: +1}

    tweets = ds.iloc[:,5].values
    sentiments = ds.iloc[:,0].values
    x = []
    y = []
    for i in range(len(tweets)):
        tweet_embed = TEXT.numericalize([tweets[i]])
        sentiment = torch.FloatTensor([[dic[sentiments[i]]]])
        x.append(tweet_embed)
        y.append(sentiment)
    return pad_sequence(x), y

def batch_it(x, bs):
    set_len = len(x) // bs
    result = []
    start = 0
    end = set_len
    for i in range(0, set_len):
        result.append(x[start:end])
        start += set_len
        end += set_len
    return result

if __name__ == "__main__":
    extract_sentiment_words()



