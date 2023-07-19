import torch
from torchvision import transforms


class DINOModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
        self.model.eval()
        self.feat_dim = self.model.num_features
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.input_size = 224
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    def forward(self, images):
        return self.model(images)
