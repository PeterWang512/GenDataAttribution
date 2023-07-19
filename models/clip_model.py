import torch
import clip


class CLIPModel(torch.nn.Module):
    """
    Wrapper class to run CLIP on the clean-fid framework.
    """
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32")
        self.model.eval()
        self.feat_dim = 512
        self.input_size = 224
        self.norm_mean = [0.48145466, 0.4578275, 0.40821073]
        self.norm_std = [0.26862954, 0.26130258, 0.27577711]

    def forward(self, images):
        return self.model.encode_image(images)
