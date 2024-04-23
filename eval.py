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

DATA = "digit_local" #zhang chexperts
PATH = "result/digit_local6"
BATCH = 8
DEVICE = "cuda:0"

#dir
img_dir = os.path.join(PATH,"test")
os.mkdir(img_dir)

#data
normal_data = DataLoader(load_data(DATA,"test_normal"), batch_size=BATCH, shuffle=False, num_workers=2)
abnormal_data = DataLoader(load_data(DATA,"test_abnormal"), batch_size=BATCH, shuffle=False, num_workers=2)

# model
model = DinoAE(device=DEVICE).to(DEVICE).eval()
state_dict = torch.load(os.path.join(PATH,"model","last.pth"))
model.load_state_dict(state_dict)

#loss
recon_img_criterion = torch.nn.MSELoss(reduction='mean').to(DEVICE)
recon_mask_criterion = torch.nn.MSELoss(reduction='mean').to(DEVICE)

# test
model.eval()

total_loss = 0
total_normal = 0
total_abnormal = 0

num = 0
for data, label in tqdm(normal_data): #normal
    num+=1
    data = data.to(DEVICE)
    label = label.to(DEVICE)
    img = data[:,0:1,...]

    result = model(data)

    recon_loss_v = recon_img_criterion(result["recon_with_mask"],img)
    total_normal += float(recon_loss_v)/len(label)
    total_loss += float(recon_loss_v)/len(label)

    normal_img = img[0][0].cpu().detach().numpy()
    normal_recon_mask = result["recon_with_mask"][0][0].cpu().detach().numpy()
    normal_recon_img = result["recon_with_img"][0][0].cpu().detach().numpy()

    residual_img1 = np.abs(normal_img - normal_recon_mask)
    residual_img2 = np.abs(normal_img - normal_recon_img)

    normal_result = np.hstack([normal_img,normal_recon_mask,residual_img1,normal_recon_img,residual_img2])
    num_str = str(num + 1000)[1:]
    plt.imsave(os.path.join(img_dir,"normal"+num_str+".png"),normal_result)

num = 0
for data, label in tqdm(abnormal_data): # abnormal
    num+=1
    data = data.to(DEVICE)
    label = label.to(DEVICE)
    img = data[:,0:1,...]

    result = model(data)

    recon_loss_v = recon_img_criterion(result["recon_with_mask"],img)
    total_abnormal += float(recon_loss_v)/len(label)
    total_loss += float(recon_loss_v)/len(label) 

    abnormal_img = img[0][0].cpu().detach().numpy()
    abnormal_recon_mask = result["recon_with_mask"][0][0].cpu().detach().numpy()
    abnormal_recon_img = result["recon_with_img"][0][0].cpu().detach().numpy()

    residual_img1 = np.abs(abnormal_img - abnormal_recon_mask)
    residual_img2 = np.abs(abnormal_img - abnormal_recon_img)

    abnormal_result = np.hstack([abnormal_img,abnormal_recon_mask,residual_img1,abnormal_recon_img,residual_img2])
    num_str = str(num + 1000)[1:]
    plt.imsave(os.path.join(img_dir,"abnormal"+num_str+".png"),abnormal_result)






        
        
        



