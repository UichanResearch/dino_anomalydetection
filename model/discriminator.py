import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self,size=7,inplace=True):
        super().__init__()

        keep_stats = True

        self.conv_model = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2, bias=True),
            nn.BatchNorm2d(16, track_running_stats=keep_stats),
            nn.LeakyReLU(0.2, inplace=inplace),
            # group1
            nn.Conv2d(16, 32, 5, 2, 2, bias=True),
            nn.BatchNorm2d(32, track_running_stats=keep_stats),
            nn.LeakyReLU(0.2, inplace=inplace),
            # group3
            nn.Conv2d(32, 64, 5, 2, 2, bias=True),
            nn.BatchNorm2d(64, track_running_stats=keep_stats),
            nn.LeakyReLU(0.2, inplace=inplace),
            # group3
            nn.Conv2d(64, 128, 5, 2, 2, bias=True),
            nn.BatchNorm2d(128, track_running_stats=keep_stats),
            nn.LeakyReLU(0.2, inplace=inplace),
            # group4
            nn.Conv2d(128, 128, 5, 2, 2, bias=True),
            nn.BatchNorm2d(128, track_running_stats=keep_stats),
            nn.LeakyReLU(0.2, inplace=inplace),
        )
        

    def forward(self, img):
        B = img.size(0)
        x = self.conv_model(img) # B, 128, W/16, H/16
        x = x.view(B, -1)
        return x

if __name__ == "__main__":
    pass