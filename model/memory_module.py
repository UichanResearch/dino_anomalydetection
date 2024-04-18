import math
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 추후 구현
# 그냥 메모리를 patch num까지 구현해서, 패치 넘버 넣으면 알아서 space를 assign해주도록 구성

class Memory(nn.Module):
    def __init__(self,num_slots, slot_dim, patch_size = 16, top = 20):
        super(Memory, self).__init__()
        self.top = top
        self.patch_size = patch_size
        self.patch_num = patch_size**2
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.patch_table = np.array([i for i in range(self.patch_num)]).reshape(patch_size,-1)

        self.memMatrix = nn.Parameter(torch.empty(self.patch_num,num_slots, slot_dim))  # M,C
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)

    def get_patch_adress(self,num):
        patch_loc = (num//self.patch_size,num%self.patch_size)
        x1 = max(0,patch_loc[0]-1)
        x2 = min(self.patch_size,patch_loc[0]+2)
        y1 = max(0,patch_loc[1]-1)
        y2 = min(self.patch_size,patch_loc[1]+2)
        address = self.patch_table[x1:x2,y1:y2]

        return address.reshape(-1)

    def forward(self, x, num):
        address = self.get_patch_adress(num)
        mem = self.memMatrix[address]
        mem = mem.reshape([-1,mem.shape[-1]])
        weight = F.linear(input=x, weight=mem)

        hs_weights = weight.clone()
        hs_weights = F.softmax(hs_weights, dim=1)
        thres, _ = torch.topk(hs_weights, self.top, sorted=True)
        thres = thres[:,[-1]]
        hs_weights = (F.relu(hs_weights - thres) * hs_weights) / (torch.abs(hs_weights - thres) + 1e-12)
        hs_weights = F.normalize(hs_weights, p=1, dim=1)
        final_weight = (hs_weights - weight).detach() + weight
        out = F.linear(final_weight, mem.permute(1, 0))

        return out
        
if __name__ == "__main__":
    feature = torch.randn(3,700)

    M1 = Memory(10,700)
    M2 = Memory(20,700)


