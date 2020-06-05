import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class GRU_Layer(nn.Module):
  def __init__(self, word_embeddings, args):
    super(GRU_Layer, self).__init__()
    print('-'*50)
    print('num_layers:', args.num_layers)
    print('bidirection:', args.bidirection)
    print('-'*50)
    self.embed = nn.Embedding.from_pretrained(embeddings=word_embeddings, freeze=True)
    self.h0 = torch.zeros(1, args.batch_size, args.hidden_size)
    self.gru = nn.GRU(args.input_size, args.hidden_size, num_layers=args.num_layers, batch_first=True, bidirectional=args.bidirection)
    self.fc = nn.Linear(args.hidden_size, args.output_size)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=args.dropout)
    init.xavier_uniform_(self.fc.weight)

  def forward(self, x):
    x = self.embed(x)
    out, h = self.gru(x, self.h0)
    h = h.squeeze(dim=0)
    # print('out: {}, h: {}'.format(out.size(), h.size()))
    h = self.fc(h)
    # h = self.relu(h)
    # h = self.dropout(h)
    return h