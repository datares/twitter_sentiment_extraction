import torch
import time
import math
from preprocess import TEXT
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(x, y, model, criterion, optimizer, scheduler, current_epoch):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for i in range(0,1000):        
        optimizer.zero_grad()
        p = random.randrange(0, x.size(1) - 16)
        ins = x[:,i:i+200,0].t().to(device)
        output = model(ins)
        loss = criterion(output, y[0,i:i+200,0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 500
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    current_epoch, i, x.size(1)//10000, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(x, y,real, model, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, 200):
            print('-' * 89)
            print(real.iloc[i,5])
            ins = x[:,i:i+1,0].t()
            output = model(ins)
            loss = criterion(output, y[0,i:i+1,0])
            print("pred: {}, truth: {}, loss: {}".format(output, y[0,i:i+1,0], loss))
            losses.append(loss.cpu().numpy())
            print('-' * 89)

    losses = np.array(losses)
    return losses.mean()