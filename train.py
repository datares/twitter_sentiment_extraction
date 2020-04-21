import torch
import time
import math
from preprocess import get_batch, TEXT, bptt
import random
def train_model(x, y, model, criterion, optimizer, scheduler, current_epoch):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    print(x.shape, len(y))
    for i in range(0, 20):        
        optimizer.zero_grad()
        pick = random.randrange(0, x.size(1))
        output = model(x[:,pick,0])

        mean_sig = torch.mean(output)
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
                    current_epoch, i, x.size(0) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, criterion, data_source):
    eval_model.eval()
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)