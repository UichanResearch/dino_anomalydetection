from model.encoder import DinoEncoder
from model.decoder import ViTDecoder
import torch.nn as nn
import torch
import random
    
class DinoAE(nn.Module): #constrtive learn
    def __init__(self,device = "cuda"):
        super().__init__()
        self.device = device
        self.En = DinoEncoder().to(device)
        self.De = ViTDecoder(device = device).to(device)
    
    def forward(self,x,random_mask = False):

        feature = self.En(x)
        out = self.De(feature)
        
        return {"feature":feature,"out":out}
    
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
    plt.imsave("a.jpg",a.mask1())

    

