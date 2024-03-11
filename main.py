from model.DinoAE import *
from tools.dataloaders import zhanglab
from torch.utils.data import DataLoader

from tools.make_dir import make_dir

import os
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

#-----------------------wandb option------------------------
entity = 'uichan980202'

project = 'DinoAE with zhang'

config={
"architecture": "basic_Dino",
"dataset": "zhang",
}

wandb.init(project = project, entity = entity, config = config)
wandb.run.name = 'run2'
#-------------------------------------------------------------

NAME = "try2"
DEVICE = "cuda:0"
BATCH = 2
EPOCH = 90
lr = 1e-4

make_dir(NAME)

train_data = DataLoader(zhanglab(mode = "train"), batch_size=BATCH, shuffle=True, num_workers=2)
val_data = DataLoader(zhanglab(mode = "val"), batch_size=BATCH, shuffle=False, num_workers=2)

model = DinoAE2().to(DEVICE)
optimizer = opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7, betas=(0.5, 0.999), weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max= 300, eta_min= lr * 0.2)

recon_img_criterion = torch.nn.MSELoss(reduction='mean').to(DEVICE)
recon_mask_criterion = torch.nn.MSELoss(reduction='mean').to(DEVICE)
feature_criterion = torch.nn.MSELoss(reduction='mean').to(DEVICE)


best_val_loss = 100
for e in range(EPOCH):
    model.train()
    train_loss = 0
    for data, label in tqdm(train_data):
        optimizer.zero_grad()
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        img = data[:,0:1,...]

        result = model(data)

        recon_img_loss = recon_img_criterion(result["recon_with_img"],img)
        recon_img_loss.backward(retain_graph=True)
        recon_img_loss = float(recon_img_loss)/BATCH

        recon_mask_loss = recon_mask_criterion(result["recon_with_mask"],img)
        recon_mask_loss.backward(retain_graph=True)
        recon_mask_loss = float(recon_mask_loss)/BATCH

        feature_loss = feature_criterion(result["feature1"],result["feature2"])
        feature_loss.backward(retain_graph=True)
        feature_loss = float(feature_loss)/BATCH
        
        train_loss += recon_img_loss + recon_mask_loss + feature_loss
        optimizer.step()

    wandb.log({"train_total":train_loss,"recon_img":recon_img_loss,"recon_mask":recon_mask_loss,"feature":feature_loss})

    model.eval()
    val_loss = 0
    for data, label in tqdm(val_data):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        img = data[:,0:1,...]

        result = model(data)

        recon_loss = recon_img_criterion(result["recon_with_mask"],img)
        val_loss += float(recon_loss)/BATCH

    
    print(f"epoch : {e}")
    print(f"train loss : {train_loss}")
    print(f"val loss : {val_loss}")

    num = str(1000+e)[1:]
    target_img = img[0][0].cpu().detach().numpy()
    recon_img = result["recon_with_mask"][0][0].cpu().detach().numpy()
    residual_img = np.abs(target_img - recon_img)
    plt.imsave(os.path.join("result",NAME,"imgs",num+".png"),np.hstack([target_img,recon_img,residual_img]))
    if val_loss <= best_val_loss:
        torch.save(model.state_dict(), os.path.join("result",NAME,"model","best.pth"))
        best_val_loss = val_loss
    wandb.log({"val_total":val_loss})



        
        
        



