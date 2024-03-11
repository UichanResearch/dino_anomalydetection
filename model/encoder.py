import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoEncoder(nn.Module):

    def __init__(self,Encoder_backbone_size = 'base'):
        super().__init__()

        # encoder (dinov2)
        backbone_archs = {"small": "vits14", "base": "vitb14", "large": "vitl14", "giant": "vitg14"}
        backbone_arch = backbone_archs[Encoder_backbone_size]
        self.encoder_backbone_name = f"dinov2_{backbone_arch}"
        self.encoder = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model= self.encoder_backbone_name).cuda()

    def forward(self,x):    
        """
            input:
                14 배수 크기의 이미지

            output:
            batch,14,14,feature크기 vector
        """
        
        result = self.encoder.forward_features(x)
        x_patch = result['x_norm_patchtokens']
        x_cls = result['x_norm_clstoken'].reshape(result['x_norm_clstoken'].shape[0],1,-1)
        x = torch.concat((x_cls,x_patch),dim = 1)
        return x
    
if __name__ == "__main__":
    En = DinoEncoder()
    img = torch.zeros(1,3,224,224).to("cuda")
    result = En(img)
    print(result.shape)