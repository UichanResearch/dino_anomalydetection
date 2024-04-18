from model.DinoAE import *
from tools.dataloaders import load_data
from torch.utils.data import DataLoader

from tools.make_dir import make_dir
from tools.set_random_seed import set_random_seed
set_random_seed(42)

import os
import yaml
import argparse
import math

import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

#arg
parser = argparse.ArgumentParser()
parser.add_argument('config',nargs='?', default='config.yaml' ,type=str, help='config yaml')
args = parser.parse_args()

# load args
with open(os.path.join("config",args.config), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# wandb option
wandb.init(project = config["project"],
           entity = config["entity"],
           config = config)
wandb.run.name = config["file name"]

# model option
DATA = config["dataset"]
NAME = config["file name"]
DEVICE = config["device"]
BATCH = config["batch"]
EPOCH = config["epoch"]
lr = config["learning rate"]

#init
make_dir(NAME)

# load data
train_data = DataLoader(load_data(DATA,"train"), batch_size=BATCH, shuffle=True, num_workers=2)
val_normal_data = DataLoader(load_data(DATA,"val_normal"), batch_size=BATCH, shuffle=False, num_workers=2)
val_abnormal_data = DataLoader(load_data(DATA,"val_abnormal"), batch_size=BATCH, shuffle=False, num_workers=2)

# model
model = DinoAE(device=DEVICE).to(DEVICE)

# optimizer
optimizer = opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7, betas=(0.5, 0.999), weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max= 300, eta_min= lr * 0.2)

#loss
recon = torch.nn.MSELoss(reduction='mean').to(DEVICE)

best_val_loss = 100
for e in range(EPOCH):
    print(e)
    #train
    model.train()
    train_loss = 0

    for data, label in tqdm(train_data):
        optimizer.zero_grad()
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        img = data[:,0:1,...]

        result = model(data)

        recon_loss = recon(result["out"],img)

        loss = recon_loss
        loss.backward()
        optimizer.step()

        train_loss += float(loss)/len(label)
    
    train_img = img[0][0].cpu().detach().numpy()
    recon_img = result["out"][0][0].cpu().detach().numpy()
    sub_img = np.abs(train_img - recon_img)
    train_result = np.hstack([train_img,recon_img,sub_img])

    # val
    model.eval()
    val_loss = 0
    val_normal = 0
    val_abnormal = 0

    for data, label in tqdm(val_normal_data): #normal
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        img = data[:,0:1,...]

        result = model(data)

        recon_loss_v = recon(result["out"],img)
        val_normal += float(recon_loss_v)/len(label)
        val_loss += float(recon_loss_v)/len(label)

    normal_img = img[0][0].cpu().detach().numpy()
    normal_recon_mask = result["out"][0][0].cpu().detach().numpy()
    sub_img = np.abs(normal_img - normal_recon_mask)
    normal_result = np.hstack([normal_img,normal_recon_mask,sub_img])
    
    for data, label in tqdm(val_abnormal_data): # abnormal
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        img = data[:,0:1,...]

        result = model(data)

        recon_loss_v = recon(result["out"],img)
        val_abnormal += float(recon_loss_v)/len(label)
        val_loss += float(recon_loss_v)/len(label)

    abnormal_img = img[0][0].cpu().detach().numpy()
    abnormal_recon_mask = result["out"][0][0].cpu().detach().numpy()
    sub_img = np.abs(abnormal_img - abnormal_recon_mask)
    abnormal_result = np.hstack([abnormal_img,abnormal_recon_mask,sub_img])

    #save result
    num = str(1000+e+1)[1:]
    plt.imsave(os.path.join("result",NAME,"imgs",num+".png"),np.vstack([train_result,normal_result,abnormal_result]))

    #save model
    torch.save(model.state_dict(), os.path.join("result",NAME,"model","last.pth"))
    if val_loss <= best_val_loss:
        torch.save(model.state_dict(), os.path.join("result",NAME,"model","best.pth"))
        best_val_loss = val_loss

    wandb.log({
        #train loss
        "train_total":train_loss,

        #val loss
        "val_total":val_loss,
        "val_normal":val_normal,
        "val_abnormal":val_abnormal
        })
    
    if math.isnan(train_loss):
        break



        
        
        



