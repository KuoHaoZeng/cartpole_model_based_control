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


class basic_GRU(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, activation_func=nn.Tanh, num_layers=1
    ):
        super(basic_GRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.activation = activation_func()

    def forward(self, x):
        x = self.activation(self.linear_in(x))
        h0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).to(x.device)
        x, hn = self.gru(x, h0)
        x = self.linear_out(self.activation(x))
        return x


class basic_LSTM(nn.Module):
    def __init__(
            self, input_dim, output_dim, hidden_dim, activation_func=nn.Tanh, num_layers=1
    ):
        super(basic_LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.activation = activation_func()

    def forward(self, x):
        x = self.activation(self.linear_in(x))
        h0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).to(x.device)
        c0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).to(x.device)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.linear_out(self.activation(x))
        return x


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
        x = x.view(
            self.batch_size, self.input_dim[0], self.input_dim[1], self.input_dim[2]
        )

        # do forward through CNN
        x = self.tanh(self.backbone(x))

        # final linear layer
        x = self.fc(x)

        return x


class model_state(nn.Module):
    def __init__(self, cfg):
        super(model_state, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.data.num_datapoints_per_epoch
        self.horizon = cfg.data.horizon
        self.input_dim = cfg.data.input_dim
        self.output_dim = cfg.data.output_dim
        self.hidden_dim = cfg.model.hidden_dim

        self.backbone = backbone[cfg.model.backbone](
            self.input_dim, self.output_dim, self.hidden_dim
        )

    def forward(self, x):
        return self.backbone(x)


backbone = {
    "resnet18": resnet18,
    "mobilenet": mobilenet_v2,
    "fc": basic_MLP,
    "gru": basic_GRU,
    "lstm": basic_LSTM,
}
