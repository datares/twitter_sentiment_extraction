import torch
import time
import math
from preprocess import TEXT
import random
def train_model(x, y, model, criterion, optimizer, scheduler, current_epoch):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    print(x.shape, len(y))
    for i in range(0, 30):        
        optimizer.zero_grad()
        pick = random.randrange(0, x.size(1))
        output = model(x[:,pick,0])
        #print(output)
        mean_sig = torch.mean(output)
        #print(mean_sig)
        loss = criterion(mean_sig, y[i])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 1
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

def evaluate(x, y, model, criterion):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, 50):
            output = model(x[:,i,0])
            mean_sig = torch.mean(output)
            loss = criterion(mean_sig, y[i])
            total_loss += loss
    return total_loss / ((x.size(1)) - 1)