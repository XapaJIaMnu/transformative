#!/usr/bin/python
import torch
import numpy as np

def get_data(train_path, max_len=100):
    print('Loading data...')

    # read sents and get vocab                                                                                     
    sents = []
    vocab = {'<pad>': 0, '<UNK>': 1, '<bos>': 2, '<eos>': 3}
    i = len(vocab)
    with open(train_path) as fp:
        for line in fp:
            if line.strip() == '' or len(line.split()) > max_len:
                continue
            sent = ['<bos>'] + line.strip().split() + ['<eos>']
            sents.append(sent)
            for word in sent:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
    return sents, vocab
    

class batch_iterator:

    def __init__(self, train, train_dict, batch_size=5, maxlen=20):
        self.train = train
        self.train_dict = train_dict
        self.vocab_len = len(train_dict)
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.train_buffer = []
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        return self

    def __next__(self):
        batch = []
        
        # if source buffer empty, fill it up
        if len(self.train_buffer) == 0:
            for sent in self.train:
                self.train_buffer.append(sent)
        # otherwise add to batch

        #@TODO mend this later
        maxlen = 0
        try:
            while len(batch) < self.batch_size:
                newsent = self.train_buffer.pop()
                if len(newsent) > maxlen:
                    maxlen = len(newsent)
                batch.append(newsent)
        except IndexError:
            self.end_of_data = True
            

        # now transform into tensors
        one_hot_batch = []
        ys = []
        for sent in batch:
            sent_one_hot = np.zeros((maxlen, self.vocab_len)) #torch.zeros((maxlen, self.vocab_len), dtype=torch.long)
            y = np.zeros(maxlen, dtype=np.long) #torch.zeros(len(sent), dtype=torch.long)
            for i, tok in enumerate(sent):
                sent_one_hot[i][self.train_dict.get(tok, 0)] = 1
                y[i] = self.train_dict.get(tok, 0)
            # pad out
            for j in range(i+1, maxlen):
                sent_one_hot[j][self.train_dict.get('<pad>')] = 1
                y[i] = self.train_dict.get('<pad>')
            one_hot_batch.append(sent_one_hot)
            ys.append(y)
            
        return torch.Tensor(one_hot_batch), torch.tensor(ys)
