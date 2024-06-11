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
from sklearn import datasets
from sklearn.manifold import TSNE

from tqdm import tqdm

DATA = "chexpert" #zhang chexpert
PATH = "result/chexpert15"
BATCH = 1
DEVICE = "cuda:2"

#dir
img_dir = os.path.join(PATH,"test")
try:
    os.mkdir(img_dir)
except:
    pass

#data
normal_data = DataLoader(load_data(DATA,"test_normal"), batch_size=BATCH, shuffle=False, num_workers=2)
abnormal_data = DataLoader(load_data(DATA,"test_abnormal"), batch_size=BATCH, shuffle=False, num_workers=2)

# model
model = DinoAE(device=DEVICE).to(DEVICE).eval()
state_dict = torch.load(os.path.join(PATH,"model","best.pth"))
model.load_state_dict(state_dict)

#loss
recon_img_criterion = torch.nn.MSELoss(reduction='mean').to(DEVICE)
recon_mask_criterion = torch.nn.MSELoss(reduction='mean').to(DEVICE)

# test
model.eval()

total_loss = 0
total_normal = 0
total_abnormal = 0

normal_feature_i = []
normal_feature_m = []
abnormal_feature_i = []
abnormal_feature_m = []

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

    normal_feature_i.append(result["feature_i"][0].cpu().detach().numpy())
    normal_feature_m.append(result["feature_m"][0].cpu().detach().numpy())
    

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

    abnormal_feature_i.append(result["feature_i"][0].cpu().detach().numpy())
    abnormal_feature_m.append(result["feature_m"][0].cpu().detach().numpy())

#tsne
data = np.vstack([normal_feature_i,normal_feature_m,abnormal_feature_i,abnormal_feature_m])
label = np.array([0]*len(normal_feature_i) + [1]*len(normal_feature_m) + [2]*len(abnormal_feature_i) + [3]*len(abnormal_feature_m))
tsne = TSNE(n_components=2, random_state=0)
data = tsne.fit_transform(data)

plt.scatter(data[label==0,0],data[label==0,1],label="normal_i")
plt.scatter(data[label==1,0],data[label==1,1],label="normal_m")
plt.scatter(data[label==2,0],data[label==2,1],label="abnormal_i")
plt.scatter(data[label==3,0],data[label==3,1],label="abnormal_m")
plt.legend()
plt.savefig(os.path.join(img_dir,"tsne.png"))
plt.close()







        
        
        



