import math
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 dropout_rate,
                 pretrained_embeddings=None):

        super(LSTMModel, self).__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        dropout_out = self.dropout(lstm_out)
        output = self.fc(dropout_out[:, -1, :])
        return output

'''
A remark on the nn.LSTM arguments (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
input_size: the size of each 'point' entering an LSTM unit
            in our task a point is a vector representating a word, asx extracted from the embedding layer
hidden_size: the number of hidden states to calculate in the LSTM layer
             so this is the same as the number of lstm units/cell in the LSTM layer
             (same as the 'units' argument in Keras)
num_layers: the number of LSTM layers
'''
