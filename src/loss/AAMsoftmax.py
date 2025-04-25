import torch, math
import torch.nn as nn
import torch.nn.functional as F


class AAMsoftmax(nn.Module):
    def __init__(self, n_class, margin, scale, embedding_size=192): 
        super(AAMsoftmax, self).__init__()
        self.m = margin
        self.s = scale
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, embedding_size), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embeddings, labels, **kwargs):
        x = embeddings    
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, labels)

        return {"loss": loss, "logits": output}