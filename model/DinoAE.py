from model.encoder import DinoEncoder
from model.decoder import ViTDecoder
from model.discriminator import Discriminator
import torch.nn as nn
import torch
import random
from tqdm import tqdm
from torch import autograd
    
class DinoAE(nn.Module): #constrtive learn
    def __init__(self,device = "cuda"):
        # img = torch.zeros(16,3,224,224).to("cuda")
        super().__init__()
        self.device = device
        self.En = DinoEncoder().to(device)
        self.De = ViTDecoder(device = device).to(device)
        self.DIS = Discriminator().to(device)

        self.mask1,self.mask2 = self.gen_img_mask()
        self.mask1 = self.mask1.to(device)
        self.mask2 = self.mask2.to(device)

        # train param
        # Encoder
        self.lr_En = 0.0001
        self.En_optimizer = torch.optim.Adam(self.En.parameters(), lr=self.lr_En, eps=1e-7, betas=(0.5, 0.999), weight_decay=0.00001)
        self.En_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.En_optimizer, T_max= 300, eta_min= self.lr_En * 0.2)

        #Decoder
        self.lr_De = 0.0001
        self.De_optimizer = torch.optim.Adam(self.De.parameters(), lr=self.lr_De, eps=1e-7, betas=(0.5, 0.999), weight_decay=0.00001)
        self.De_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.De_optimizer, T_max= 300, eta_min= self.lr_De * 0.2)

        # Discriminator
        self.lr_Dis = 0.0001
        self.generator_iters = 10
        self.critic_iter = 5
        self.lambda_term = 10
        self.Dis_optimizer = torch.optim.Adam(self.DIS.parameters(), lr=self.lr_Dis, eps=1e-7, betas=(0.5, 0.999), weight_decay=0.00001)

        #loss
        self.recon_img_criterion = torch.nn.MSELoss(reduction='mean').to(device)  
        self.recon_mask_criterion = torch.nn.MSELoss(reduction='mean').to(device) 

    def train(self, train_loader):
        data = self.get_infinite_batches(train_loader)
        postive = torch.tensor(1.0).to(self.device)
        negative = torch.tensor(-1.0).to(self.device)

        self.En.train()
        self.De.train()
        self.DIS.train()

        total_dis_fake_loss = 0
        total_dis_real_loss = 0
        
        total_gen_recon_i_loss = 0
        total_gen_recon_m_loss = 0
        total_gen_dis_loss = 0
        total_gen_recon_loss = 0

        for ae_iter in tqdm(range(self.generator_iters)):
            #discreminator update
            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0

            self.DIS.train()
            for dis_iter in range(self.critic_iter):
                self.Dis_optimizer.zero_grad()

                x = next(data).to(self.device)
                img = x[:,0:1]
                batch_size = x.shape[0]
                
                # train real
                dis_real_loss = self.DIS(img)
                dis_real_loss = dis_real_loss.mean()
                dis_real_loss.backward(negative)

                # train fake
                result = self.forward(x)
                recon_m = result["recon_with_mask"]
                dis_fake_loss = self.DIS(recon_m)
                dis_fake_loss = dis_fake_loss.mean()
                dis_fake_loss.backward(postive)

                # gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(x[:,0:1].data, recon_m.data)
                gradient_penalty.backward()
                self.Dis_optimizer.step()

                Wasserstein_D += dis_real_loss - dis_fake_loss
                d_loss_real += dis_real_loss
                d_loss_fake += dis_fake_loss
            
            #AE update
            self.En.zero_grad()
            self.De.zero_grad()

            x = next(data).to(self.device)
            img = x[:,0:1]
            batch_size = x.shape[0]

            result = self.forward(x)
            recon_i = result["recon_with_img"]
            recon_m = result["recon_with_mask"]

            recon_img_loss = self.recon_img_criterion(recon_i,img)
            recon_img_loss.backward(retain_graph=True)
            recon_mask_loss = self.recon_mask_criterion(recon_m,img)
            recon_mask_loss.backward(retain_graph=True)

            dis_loss = self.DIS(recon_m)
            dis_loss = dis_loss.mean()
            dis_loss.backward(negative)

            self.En_optimizer.step()
            self.De_optimizer.step()
            self.En_scheduler.step()
            self.De_scheduler.step()

            # log loss
            total_dis_fake_loss += d_loss_fake / batch_size
            total_dis_real_loss += d_loss_real / batch_size
            total_dis_loss = total_dis_fake_loss + total_dis_real_loss

            total_gen_recon_i_loss += recon_img_loss / batch_size
            total_gen_recon_m_loss += recon_mask_loss / batch_size
            total_gen_dis_loss += dis_loss / batch_size
            total_gen_recon_loss = total_gen_recon_i_loss + total_gen_recon_m_loss

            total_loss = total_dis_fake_loss + total_dis_real_loss + total_gen_recon_i_loss + total_gen_recon_m_loss

            loss_dict = {"train_dis_fake_loss":total_dis_fake_loss,
                         "train_dis_real_loss":total_dis_real_loss,
                         "train_dis_loss":total_dis_loss,
                         "train_gen_recon_i_loss":total_gen_recon_i_loss,
                         "train_gen_recon_m_loss":total_gen_recon_m_loss,
                         "train_gen_dis_loss":total_gen_dis_loss,
                         "train_gen_loss":total_gen_recon_loss,
                         "train_loss":total_loss}

        return loss_dict
    
    def val(self,val_loader,label = ""):
        self.En.eval()
        self.De.eval()
        self.DIS.eval()

        DIS_features = []

        total_recon_img_loss = 0
        total_recon_mask_loss = 0
        total_dis_loss = 0
        total_recon_loss = 0

        for x, _ in val_loader:
            x = x.to(self.device)
            img = x[:,0:1]
            batch_size = x.shape[0]
            result = self.forward(x)
            recon_i = result["recon_with_img"]
            recon_m = result["recon_with_mask"]

            recon_img_loss = self.recon_img_criterion(recon_i,img)
            recon_mask_loss = self.recon_mask_criterion(recon_m,img)

            dis_loss = self.DIS(recon_m)
            DIS_features.append(dis_loss)
            dis_loss = dis_loss.mean()

            total_recon_img_loss += recon_img_loss / batch_size
            total_recon_mask_loss += recon_mask_loss / batch_size
            total_dis_loss += dis_loss / batch_size
            total_recon_loss = total_recon_img_loss + total_recon_mask_loss

        img_dict = {"recon_i":recon_i[0],"recon_m":recon_m[0],"x":x[0]}
        DIS_features = torch.vstack(DIS_features) #주의 모든 val의 feature를 반환하는게 아님
        loss_dict = {label + "_recon_img_loss":total_recon_img_loss,
                     label + "_recon_mask_loss":total_recon_mask_loss,
                     label + "_dis_loss":total_dis_loss,
                     label + "_recon_loss":total_recon_loss}
        
        return loss_dict, DIS_features, img_dict
        
            
    def forward(self,x,random_mask = False):
        x1 = x.clone()
        x2 = x.clone()
        if random_mask:
            self.gen_img_mask(random_mask=True)
        x1 *= self.mask1
        x2 *= self.mask2

        feature1 = self.En(x1)
        feature2 = self.En(x2)

        x1,x2,f_loss = self.De(feature1,feature2)
        
        recon_i = x1 * self.mask1 + x2 * self.mask2
        recon_m = x1 * self.mask2 + x2 * self.mask1

        return {"recon_with_img":recon_i,"recon_with_mask":recon_m}
    
    def calculate_gradient_penalty(self, real_images, fake_images):
        batch_size = real_images.shape[0]
        eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3)).requires_grad_(True)
        eta = eta.to(self.device)
        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # calculate probability of interpolated examples
        prob_interpolated = self.DIS(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated,
                               inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True,
                               retain_graph=True)[0]

        # flatten the gradients to it calculates norm batchwise
        gradients = gradients.view(gradients.size(0), -1)
        
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty
    
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
    
    @staticmethod
    def get_infinite_batches(data_loader):
        while True:
            for images, _ in data_loader:
                yield images

    
