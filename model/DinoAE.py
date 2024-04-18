from model.encoder import DinoEncoder
from model.decoder import ViTDecoder
from model.memory_module import Memory
import torch.nn as nn
import torch
import random
    
class DinoAE(nn.Module): #constrtive learn
    def __init__(self,device = "cuda"):
        # img = torch.zeros(16,3,224,224).to("cuda")
        super().__init__()
        self.device = device
        self.En = DinoEncoder().to(device)
        self.De = ViTDecoder(device = device).to(device)
        self.Mem = Memory(512,768)

        self.mask1,self.mask2 = self.gen_img_mask()
        self.mask1 = self.mask1.to(device)
        self.mask2 = self.mask2.to(device)

        self.feature_mask1 = torch.tensor([int(i % 2 == 0)for i in range(16*16)]+[0]).to(device)
        self.feature_mask1 = self.feature_mask1.view([1,-1,1])
        self.feature_mask2 = torch.tensor([int(i % 2 == 1)for i in range(16*16)]+[0]).to(device)
        self.feature_mask2 = self.feature_mask2.view([1,-1,1])
        self.cls_mask = torch.zeros([1,16*16+1,1]).to(device)
        self.cls_mask[0,-1,0] = 1
    
    def forward(self,x,random_mask = False):
        x1 = x.clone()
        x2 = x.clone()

        if random_mask:
            self.gen_img_mask(random_mask=True)

        x1 *= self.mask1
        x2 *= self.mask2

        feature1 = self.En(x1)
        feature2 = self.En(x2)

        img_feature = feature1 * self.feature_mask1 + feature2 * self.feature_mask2
        mask_feature = feature1 * self.feature_mask2 + feature2 * self.feature_mask1

        new_mask_feature = []
        for i in range(16*16):
            result = self.Mem(mask_feature[:,i,:],i)
            result = result.reshape([result.shape[0],1,-1])
            new_mask_feature.append(result)
        new_mask_feature.append(torch.zeros_like(new_mask_feature[0]).to(self.device))
        mask_feature = torch.cat(new_mask_feature, dim = 1)

        new_feature1 = img_feature * self.feature_mask1 + mask_feature * self.feature_mask2 + feature1 * self.cls_mask
        new_feature2 = img_feature * self.feature_mask2 + mask_feature * self.feature_mask1 + feature2 * self.cls_mask

        x1,x2,f_loss = self.De(new_feature1,new_feature2)
        
        recon_i = x1 * self.mask1 + x2 * self.mask2
        recon_m = x1 * self.mask2 + x2 * self.mask1
        
        return {"feature1":feature1,"feature2":feature2,"recon_with_img":recon_i,"recon_with_mask":recon_m, "decoder_feature_loss":f_loss, "new_feature1":new_feature1,"new_feature2":new_feature2}
    
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

    

