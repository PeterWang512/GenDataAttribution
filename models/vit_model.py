import torch
from torchvision import transforms
from timm.models.vision_transformer import vit_base_patch16_384


class ViTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_base_patch16_384(pretrained=True)
        self.model.head = torch.nn.Identity()
        self.model.eval()
        self.feat_dim = 768
        self.preprocess = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
        self.input_size = 384
        self.norm_mean = 0.5
        self.norm_std = 0.5

    def forward(self, images):
        return self.model(images)
