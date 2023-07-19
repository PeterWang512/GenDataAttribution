import torch


def create_encoders(model, tune_type, **kwargs):
    if tune_type == 'linear':
        real_encoder = LinearProbe(model)
        synth_encoder = LinearProbe(model)
    elif tune_type == 'mlp':
        real_encoder = MLPProbe(model, **kwargs)
        synth_encoder = MLPProbe(model, **kwargs)
    elif tune_type == 'channel':
        real_encoder = ChannelUpdate(model)
        synth_encoder = ChannelUpdate(model)
    else:
        raise NotImplementedError

    return real_encoder, synth_encoder


class ElementWiseLayer(torch.nn.Module):
  def __init__(self, dim=512):
    super(ElementWiseLayer, self).__init__()
    self.weights = torch.nn.Parameter(torch.ones(1, dim))

  def forward(self, x):
    return x * self.weights


class ChannelUpdate(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        self.mapper = ElementWiseLayer(self.model.feat_dim)

    def forward(self, images):
        x = self.model(images).float().detach()
        x = self.mapper(x)
        x = torch.nn.functional.normalize(x, dim=-1)
        return x


class LinearProbe(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        self.mapper = torch.nn.Linear(self.model.feat_dim, self.model.feat_dim)
        self.mapper.weight.data.copy_(torch.eye(self.model.feat_dim))
        self.mapper.bias.data.fill_(0)

    def forward(self, images):
        x = self.model(images).float().detach()
        x = self.mapper(x)
        x = torch.nn.functional.normalize(x, dim=-1)
        return x


class MLPProbe(torch.nn.Module):
    def __init__(self, model, mlp_dim, output_dim, drop_out=0, num_layers=2):
        super().__init__()
        self.model = model

        for param in self.model.parameters():
            param.requires_grad = False

        self.mapper = self._build_mlp(num_layers=num_layers, input_dim=self.model.feat_dim, mlp_dim=mlp_dim, output_dim=output_dim, drop_out=drop_out)

    def forward(self, images):
        x = self.model(images).float().detach()
        x = self.mapper(x)
        x = torch.nn.functional.normalize(x, dim=-1)
        return x

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, drop_out=0, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(torch.nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(torch.nn.BatchNorm1d(dim2))
                mlp.append(torch.nn.ReLU(inplace=True))
                mlp.append(torch.nn.Dropout(p=drop_out))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(torch.nn.BatchNorm1d(dim2, affine=False))

        return torch.nn.Sequential(*mlp)


class Calibrator(torch.nn.Module):
    def __init__(self, bias_gain=0.001):
        super().__init__()
        self.inv_tau = torch.nn.Parameter(torch.tensor(1.0))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.bias_gain = bias_gain
        self.threshold = torch.nn.Softplus(beta=100)

    def get_bias(self):
        return self.bias * self.bias_gain

    def forward(self, scores):
        # Assume scores is a 2D tensor of size (batch_size, num_samples)
        score_scaled = scores * self.inv_tau
        exp_t = torch.exp(score_scaled - torch.max(score_scaled, dim=1, keepdim=True).values)
        trunc = torch.nn.functional.relu(exp_t + self.get_bias())

        assignment = trunc / torch.sum(trunc, dim=1, keepdim=True)
        return assignment

    def loss_forward(self, count_gt, selected_score, rest_count, rest_dist, return_batch=False):
        B, L = selected_score.shape
        max_score = selected_score.max(dim=1, keepdim=True).values
        selected_score = selected_score - max_score
        exp_score = (selected_score * self.inv_tau).exp_()

        trunc_score = self.threshold(exp_score + self.get_bias())

        denominator = torch.log(trunc_score.sum(dim=1))
        indices = torch.arange(L).expand(B, -1).to(selected_score.device)
        mask = (indices < count_gt[:, None]).float()
        masked_score = trunc_score * mask + torch.ones_like(trunc_score) * (1 - mask)
        nominator = torch.log(masked_score).sum(dim=1) / count_gt

        if return_batch:
            loss = denominator - nominator
            return loss

        loss = (denominator - nominator).mean()
        return loss
