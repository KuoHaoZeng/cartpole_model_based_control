import torch
import torch.nn as nn
from network.mobilenet import mobilenet_v2
from network.resnet import resnet18

class basic_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, activation_func=nn.Tanh):
        super(basic_MLP, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_func(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_func(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.backbone(x)

class model_CNN(nn.Module):
    def __init__(self, cfg):
        super(model_CNN, self).__init__()
        # some hypeparameters
        self.cfg = cfg
        self.batch_size = cfg.data.num_datapoints_per_epoch
        self.input_dim = cfg.data.input_dim

        # allocate model parameters
        self.backbone = backbone[cfg.model.backbone](num_classes=cfg.model.hidden_sim)
        self.fc = nn.Linear(cfg.model.hidden_sim * 2, cfg.model.output_sim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # resort the input dimension
        x = x.view(self.batch_size, self.input_dim[0], self.input_dim[1], self.input_dim[2])

        # do forward through CNN
        x = self.tanh(self.backbone(x))

        # final linear layer
        x = self.fc(x)

        return x

class policy_state_basic(nn.Module):
    def __init__(self, cfg):
        super(policy_state_basic, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.data.num_datapoints_per_epoch
        self.horizon = cfg.data.horizon
        self.input_dim = cfg.data.input_dim
        self.output_dim = cfg.data.output_dim
        self.hidden_dim = cfg.policy.hidden_dim

        self.backbone = backbone[cfg.policy.backbone](self.input_dim, self.output_dim, self.hidden_dim)

    def forward(self, x):
        x = x.view(self.batch_size * self.horizon, self.input_dim)
        return self.backbone(x)

backbone = {'resnet18': resnet18, 'mobilenet': mobilenet_v2, 'fc': basic_MLP}
