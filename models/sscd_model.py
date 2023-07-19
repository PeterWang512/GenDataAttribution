import torch
from torchvision import transforms


class SSCDModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.jit.load("weights/pretrained/sscd_disc_mixup.torchscript.pt")
        self.model.eval()
        self.feat_dim = 512
        self.preprocess = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.input_size = 288
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    def forward(self, images):
        return self.model(images)
