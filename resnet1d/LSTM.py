import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=1,
                           hidden_size=128,
                           num_layers=4,
                           bidirectional=True,
                           batch_first=True)

        # self.maxpool = nn.MaxPool1d()
        self.dense = nn.Linear(128,4)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        out = x
        # (batch, channel ,length) -> (batch, length, channel)
        out = out.permute(0,2,1)
        _, (out, _) = self.rnn(out)
        # (length, batch, channel) -> (batch, channel ,length)
        out = out.permute(1,2,0)
        out = out.max(-1).values
        out = self.dense(out)
        out = self.softmax(out)
        return out