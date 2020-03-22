import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, input_dim, w2v_dim):
        super().__init__()
        self.input_dim = input_dim
        self.V = nn.Embedding(input_dim, w2v_dim, sparse=False)
        self.U = nn.Embedding(input_dim, w2v_dim, sparse=False)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.zeros_(self.U.weight)
        
    def forward(self, x, target, device):
        double_ws = target.shape[1]
        bs = x.shape[0]
        
        x = x.to(device)
        target = target.to(device)
        v_c = self.V(x)
        u_j = self.U(target)
        u_V = self.U(torch.arange(self.input_dim, dtype=torch.long).to(device))
        
        pos_loss = -torch.sum(torch.bmm(v_c, u_j.transpose(1, 2)))
        neg_value = torch.matmul(v_c, u_V.T)
        max_neg =  torch.max(neg_value)
        neg_loss = double_ws * max_neg * bs + double_ws * torch.sum(torch.log( torch.sum(torch.exp(neg_value - max_neg), dim=(1, 2)) ))
        
        return pos_loss / bs, neg_loss / bs


class SkipGramNS(SkipGram):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x, target, neg_samples, device):
        double_ws = target.shape[1]
        bs = x.shape[0]
        
        x = x.to(device)
        target = target.to(device)
        neg_samples = neg_samples.to(device)
        v_c = self.V(x)
        u_j = self.U(target)
        u_k = self.U(neg_samples)
        
        pos_loss = torch.sum( -F.logsigmoid(torch.bmm(v_c, u_j.transpose(1, 2))) )
        neg_loss = torch.sum( torch.sum( -F.logsigmoid(-torch.bmm(v_c, u_k.transpose(1, 2))), dim=(1, 2) ))
        
        return pos_loss / bs, neg_loss / bs


if __name__ == '__main__':
    model = SkipGram(100, 10)
    print('Model SkipGram initiated')
    model = SkipGramNS(100, 10)
    print('Model SkipGramNS initiated')