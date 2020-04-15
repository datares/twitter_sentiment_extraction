from preprocess import batchify, TEXT
import torch.nn as nn
import torchtext
from model import TransformerModel
from train import train_model, evaluate
import torch
import time
import math

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
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    train_data = batchify(train_txt, batch_size)
    train_data = train_data.to(device)
    val_data = batchify(val_txt, eval_batch_size)
    val_data = val_data.to(device)

    test_data = batchify(test_txt, eval_batch_size)
    test_data = test_data.to(device)


    best_val_loss = float("inf")
    epochs = 3
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_model(model, criterion, optimizer, train_data, scheduler, epoch)
        val_loss = evaluate(model,criterion, val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    test_loss = evaluate(best_model, criterion, test_data)

    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

if __name__ == "__main__":
    extract_sentiment_words()



