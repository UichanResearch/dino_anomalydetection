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

SAVE_RESULT = True
WAND_LOG = True

# load args
with open(os.path.join("config",args.config), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# wandb option
if WAND_LOG:
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
if SAVE_RESULT:
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
recon_img_criterion = torch.nn.MSELoss(reduction='mean').to(DEVICE)
recon_mask_criterion = torch.nn.MSELoss(reduction='mean').to(DEVICE)
feature_criterion = torch.nn.MSELoss(reduction='mean').to(DEVICE)
new_feature_criterion = torch.nn.MSELoss(reduction='mean').to(DEVICE)

best_val_loss = 100
for e in range(EPOCH):
    print(e)
    #train
    model.train()
    train_loss = 0
    total_recon_img = 0
    total_recon_mask = 0
    total_feature = 0

    for data, label in tqdm(train_data):
        optimizer.zero_grad()
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        img = data[:,0:1,...]

        result = model(data)

        recon_img_loss = recon_img_criterion(result["recon_with_img"],img)
        recon_mask_loss = recon_mask_criterion(result["recon_with_mask"],img)
        feature_loss = feature_criterion(result["feature1"],result["feature2"])
        new_feature_loss = new_feature_criterion(result["new_feature1"],result["new_feature2"])
        decoder_feature_loss = result["decoder_feature_loss"]

        #loss = recon_img_loss + recon_mask_loss + 0.5*feature_loss + new_feature_loss + decoder_feature_loss
        loss = recon_img_loss + 0.5*feature_loss + decoder_feature_loss + new_feature_loss
        loss.backward()
        optimizer.step()

        train_loss += float(loss)/len(label)
        total_recon_img += float(recon_img_loss)/len(label) 
        total_recon_mask += float(recon_mask_loss)/len(label)
        total_feature += float(feature_loss)/len(label)
    
    train_img = img[0][0].cpu().detach().numpy()
    train_recon_img = result["recon_with_img"][0][0].cpu().detach().numpy()
    train_recon_mask = result["recon_with_mask"][0][0].cpu().detach().numpy()
    train_result = np.hstack([train_img,train_recon_img,train_recon_mask])

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

        recon_loss_v = recon_img_criterion(result["recon_with_mask"],img)
        val_normal += float(recon_loss_v)/len(label)
        val_loss += float(recon_loss_v)/len(label)

    normal_img = img[0][0].cpu().detach().numpy()
    normal_recon_mask = result["recon_with_mask"][0][0].cpu().detach().numpy()
    residual_img = np.abs(normal_img - normal_recon_mask)
    normal_result = np.hstack([normal_img,normal_recon_mask,residual_img])
    
    for data, label in tqdm(val_abnormal_data): # abnormal
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        img = data[:,0:1,...]

        result = model(data)

        recon_loss_v = recon_img_criterion(result["recon_with_mask"],img)
        val_abnormal += float(recon_loss_v)/len(label)
        val_loss += float(recon_loss_v)/len(label)

    abnormal_img = img[0][0].cpu().detach().numpy()
    abnormal_recon_mask = result["recon_with_mask"][0][0].cpu().detach().numpy()
    residual_img = np.abs(abnormal_img - abnormal_recon_mask)
    abnormal_result = np.hstack([abnormal_img,abnormal_recon_mask,residual_img])

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
        "recon_img":total_recon_img,
        "recon_mask":total_recon_mask,
        "feature":total_feature,

        #val loss
        "val_total":val_loss,
        "val_normal":val_normal,
        "val_abnormal":val_abnormal
        })
    
    if math.isnan(train_loss):
        break



        
        
        



