from model.encoder import DinoEncoder
from model.decoder import ViTDecoder
import torch.nn as nn
import torch

class DinoAE(nn.Module): #basic
    def __init__(self,device = "cuda"):
        # img = torch.zeros(16,3,224,224).to("cuda")
        super().__init__()
        self.En = DinoEncoder().to(device)
        self.De = ViTDecoder().to(device)
    
    def forward(self,x):
        x = self.En(x)
        x = self.De(x)
        return x
    
class DinoAE2(nn.Module): #constrtive learn
    def __init__(self,device = "cuda"):
        # img = torch.zeros(16,3,224,224).to("cuda")
        super().__init__()
        self.En = DinoEncoder().to(device)
        self.De = ViTDecoder().to(device)

        self.mask1,self.mask2 = self.gen_img_mask()
        self.mask1 = self.mask1.to(device)
        self.mask2 = self.mask2.to(device)
    
    def forward(self,x):
        x1 = x.clone()
        x2 = x.clone()

        x1 *= self.mask1
        x2 *= self.mask2

        feature1 = self.En(x1)
        feature2 = self.En(x2)

        x1 = self.De(feature1)
        x2 = self.De(feature2)

        recon_i = x1 * self.mask1 + x2 * self.mask2
        recon_m = x1 * self.mask2 + x2 * self.mask1
        
        return {"feature1":feature1,"feature2":feature2,"recon_with_img":recon_i,"recon_with_mask":recon_m}
    
    @staticmethod
    def gen_img_mask():
        img1 = torch.zeros(224,224)
        window = torch.ones(14,14)
        for i in range(16):
            for j in range(16):
                if i % 2 == j % 2:
                    img1[14*i:14*(i+1),14*j:14*(j+1)] = window
        img2 = torch.ones_like(img1)
        img2 -= img1
        img1 = img1.reshape(1,1,224,224)
        img2 = img2.reshape(1,1,224,224)
        return img1, img2
    
if __name__ == "__main__":
    pass

    

