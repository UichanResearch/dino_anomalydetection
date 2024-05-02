from model.encoder import DinoEncoder
from model.decoder import ViTDecoder
from model.discriminator import Discriminator
import torch.nn as nn
import torch
import random
    
class DinoAE(nn.Module): #constrtive learn
    def __init__(self,device = "cuda"):
        super().__init__()
        self.device = device
        sample = torch.ones([1,3,224,224]).to(device)
        self.En = DinoEncoder().to(device)
        self.De = ViTDecoder(device = device).to(device)
        temp_feature = self.En(sample)
        self.patch_row = int((temp_feature.shape[1] - 1)**0.5)

        self.mask1,self.mask2 = self.gen_img_mask()
        self.mask1 = self.mask1.to(device)
        self.mask2 = self.mask2.to(device)

        self.feature_mask1 = torch.zeros([1,temp_feature.shape[1],1]).to(device)
        for i in range(self.patch_row):
            for j in range(self.patch_row):
                if i % 2 == j % 2:
                    self.feature_mask1[0,i*self.patch_row + j+1,0] = 1
        self.feature_mask2 = torch.ones_like(self.feature_mask1) - self.feature_mask1
        self.feature_mask1[0,0,0] = 1 #cls tocken
        self.feature_mask2[0,0,0] = 1 #cls tocken

        self.feature_mask1.to(device)
        self.feature_mask2.to(device)
    
    def forward(self,x,random_mask = False):
        feature = self.En(x)

        feature1 = feature * self.feature_mask1
        feature2 = feature * self.feature_mask2
        cls_tocken = feature[:,1,:]

        x1,x2 = self.De(feature1,feature2)

        recon_i = x1 * self.mask1 + x2 * self.mask2
        recon_m = x1 * self.mask2 + x2 * self.mask1
        
        #A_mask = self.Dis(recon_m.detach())
        #A_img = self.Dis(x[:,0:1,...])
        
        return {"feature":feature,
                "recon_with_img":recon_i,
                "recon_with_mask":recon_m,
                }
    
    @staticmethod
    def gen_img_mask(random_mask = False):
        patch_num = 16*16
        patch_row = int(patch_num**0.5)
        img1 = torch.zeros(224,224)
        window = torch.ones(14,14)
        if random_mask:
            random_numbers = random.sample(range(patch_num), patch_num//2)
        else:
            random_numbers = []
            count = 0
            for i in range(16):
                for j in range(16):
                    if i%2 == j%2:
                        random_numbers.append(i*16 + j)

        for n in random_numbers:
            i = n//patch_row
            j = n% patch_row
            img1[14*i:14*(i+1),14*j:14*(j+1)] = window.clone()
        
        img2 = torch.ones_like(img1)
        img2 -= img1
        img1 = img1.reshape(1,1,224,224)
        img2 = img2.reshape(1,1,224,224)
        return img1, img2
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    a = DinoAE()
    a(torch.zeros([1,3,244,244]))

    