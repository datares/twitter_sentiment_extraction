# main modules
import time
import math
import pandas as pd
import numpy as np
import torch.nn as nn
# torch modules
import torchtext
import torch
from torch.nn.utils.rnn import pad_sequence

# local modules
from preprocess import TEXT
from model import TransformerModel
from train import train_model, evaluate

# set up cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess():
    train_ds = pd.read_csv("train.csv", encoding = "ISO-8859-1", header=None)
    test_ds = pd.read_csv("test.csv", encoding = "ISO-8859-1", header=None)
    
    r_train = train_ds.sample(frac=1)
    r_test = test_ds.sample(frac=1)
    x_train, y_train = create_trainset(r_train)
    x_val, y_val = create_trainset(r_test)

    return x_train, y_train, x_val, y_val, r_train, r_test

def create_trainset(ds):
    dic = {0: 0,2:0.5, 4: 1}

    tweets = ds.iloc[:,5].values
    sentiments = ds.iloc[:,0].values
    x, y = [], []
    for i in range(len(tweets)):
    #for i in range(len(tweets)):
        # create tensors
        if len(tweets[i]) <= 120:

            tweet_embed = TEXT.numericalize([tweets[i]])
            sentiment = torch.LongTensor([[dic[sentiments[i]]]])
            
            # append to list of tensors
            x.append(tweet_embed)
            y.append(sentiment)
    # return a padded version of the sequence to allow for same size tensors
    return pad_sequence(x).to(device), pad_sequence(y).to(device)

def extract_sentiment_words():
    # create vocabulary using wikitext2
    train_txt, _, _ = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)

    start = time.time()
    x_train, y_train, x_val, y_val, rtrain, rtest = preprocess()
    end = time.time()

    print("PREPROCESSING TIME: {}".format(end - start))
    
    ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
    
    # FIXME set up batched examples for better generality
    # batch_size = 20
    # eval_batch_size = 10

    # configs
    emsize = 200 # embedding dimension
    nhid = 200 # feedforward dimension
    nlayers = 2 # n encoders
    nhead = 2 # multiattention heads
    dropout = 0.2 # the dropout value

    # initialize main torch vars
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    lr = 0.05 # learning rate

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 50
    best_model = None
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_model(x_train, y_train, model, criterion, optimizer, scheduler, epoch)
        val_loss = evaluate(x_val, y_val,rtest, model,criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()
    
    # test_loss = evaluate(best_model, criterion, test_data)

    # print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    #     test_loss, math.exp(test_loss)))
    # print('=' * 89)
    return best_model

if __name__ == "__main__":
    best = extract_sentiment_words()
    torch.save(best.state_dict(), "./best")



