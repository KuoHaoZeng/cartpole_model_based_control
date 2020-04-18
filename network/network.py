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

        # allocate model parameters
        self.backbone = protocol[cfg.model.protocol](num_classes=cfg.model.hidden_size)
        self.fc = nn.Linear(cfg.hidden_size * 2, cfg.output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # resort the input dimension
        x = x.view(self.batch_size, self.input_num_frame, self.input_size, self.input_size)

        # do forward through CNN
        x = self.tanh(self.backbone(x))

        # final linear layer
        x = self.fc(x)

        return x