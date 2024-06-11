from model.DinoAE import *
from tools.dataloaders import load_data
from torch.utils.data import DataLoader

from tools.make_dir import make_dir
from tools.set_random_seed import set_random_seed
set_random_seed(42)

import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

wandb.init(project = "new DinoAE with chexpert",
           entity = "uichan980202")
wandb.run.name = "chexpert1"

try:
    make_dir("chexpert1")
except:
    pass

DATA = "chexpert"
BATCH = 4
DEVICE = "cuda:7"

# load data
train_data = DataLoader(load_data(DATA,"train"), batch_size=BATCH, shuffle=True, num_workers=2)
val_normal_data = DataLoader(load_data(DATA,"val_normal"), batch_size=BATCH, shuffle=False, num_workers=2)
val_abnormal_data = DataLoader(load_data(DATA,"val_abnormal"), batch_size=BATCH, shuffle=False, num_workers=2)

model = DinoAE(device=DEVICE).to(DEVICE)
for i in range(1000):
    train_log = model.train(train_data)
    wandb.log(train_log)
    val_log,feat_n,img_n = model.val(val_normal_data,label = "normal")
    wandb.log(val_log)
    val_log,feat_a,img_a = model.val(val_abnormal_data,label = "abnormal")
    wandb.log(val_log)

    normal_img = img_n["x"][0].detach().to("cpu").numpy()
    normal_i = img_n["recon_i"][0].detach().to("cpu").numpy()
    normal_m = img_n["recon_m"][0].detach().to("cpu").numpy()
    normal_img = np.concatenate([normal_img,normal_i,normal_m],axis = 1)
    
    abnormal_img = img_a["x"][0].detach().to("cpu").numpy()
    abnormal_i = img_a["recon_i"][0].detach().to("cpu").numpy()
    abnormal_m = img_a["recon_m"][0].detach().to("cpu").numpy()
    abnormal_img = np.concatenate([abnormal_img,abnormal_i,abnormal_m],axis = 1)

    result_img = np.concatenate([normal_img,abnormal_img],axis = 0)
    plt.imsave(f"result/chexpert1/imgs/{str(1000+i)[1:]}.png",result_img)

    #tsne
    # feat_n = feat_n.detach().to("cpu").numpy()
    # feat_a = feat_a.detach().to("cpu").numpy()
    # data = np.vstack([feat_n,feat_a])
    # labels = np.vstack([np.zeros((feat_n.shape[0],1)),np.ones((feat_a.shape[0],1))])
    # tsne = TSNE(n_components=2, perplexity=20)
    # tsne_results = tsne.fit_transform(data)
    # plt.scatter(tsne.embedding_[:,0],tsne.embedding_[:,1],c = labels)
    # plt.savefig(f"result/chexpert1/tsne/{str(1000+i)[1:]}.png")



    




