import torch
import torch.nn as nn
from network.mobilenet import mobilenet_v2
from network.resnet import resnet18

protocol = {'resnet18': resnet18, 'mobilenet': mobilenet_v2}

class model_CNN(nn.Module):
    def __init__(self, cfg):
        super(model_CNN, self).__init__()
        # some hypeparameters
        self.cfg = cfg
        self.batch_size = cfg.model.batch_size
        self.input_dim = cfg.model.input_dim

        # allocate model parameters
        self.backbone = protocol[cfg.model.protocol](num_classes=cfg.model.hidden_size)
        self.fc = nn.Linear(cfg.model.hidden_size * 2, cfg.model.output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # resort the input dimension
        x = x.view(self.batch_size, self.input_dim[0], self.input_dim[1], self.input_dim[2])

        # do forward through CNN
        x = self.tanh(self.backbone(x))

        # final linear layer
        x = self.fc(x)

        return x