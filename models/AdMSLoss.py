import torch
import torch.nn as nn
import torch.nn.functional as F

class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, ml=0.4, ms=0.1, num_live=6):
        '''
        Asymmetric AM-Softmax loss (PatchNet).

        Args:
            in_features:  embedding dimension (ResNet-18 -> 512).
            out_features: number of fine-grained classes (OULU-NPU P1 -> 30:
                          6 live + 6 cameras x 4 attacks = 24 spoof).
            s:            scale applied to the cosine logits.
            ml:           additive margin for LIVE classes (larger -> more compact).
            ms:           additive margin for SPOOF classes (smaller -> looser).
            num_live:     number of live classes; labels [0, num_live) are live,
                          labels [num_live, out_features) are spoof.
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = torch.tensor(s).cuda()
        self.ml = ml
        self.ms = ms
        self.num_live = num_live
        self.in_features = in_features
        self.out_features = out_features
        # self.fc = nn.Linear(in_features, out_features, bias=False)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        # for W in self.fc.parameters():
        #     W.data = F.normalize(W, dim=1)
        # wf = self.fc(x)
        wf = F.linear(F.normalize(x, dim=1), F.normalize(self.weight))
        wf = wf.clamp(-1, 1)
        # asymmetric margin: live classes get the larger margin ml, spoof get ms
        m = torch.zeros(int(labels.shape[0]))
        real = labels < self.num_live
        fake = labels >= self.num_live
        real_indices = real.nonzero().squeeze(1)
        fake_indices = fake.nonzero().squeeze(1)
        m[real_indices] = self.ml
        m[fake_indices] = self.ms
        # 計算loss
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - m.cuda())
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        # print(numerator)
        # print(excl)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
    
    def _predict(self, x):
        # for W in self.fc.parameters():
        #     W.data = F.normalize(W, dim=1)
        # x = F.normalize(x, dim=1)
        y = self.s * F.linear(F.normalize(x, dim=1), F.normalize(self.weight))
        prob = F.softmax(y, dim=1)
        # live probability = sum of softmax over the live classes [0, num_live)
        live_prob = torch.sum(prob[:, 0:self.num_live], dim=1)
        return live_prob