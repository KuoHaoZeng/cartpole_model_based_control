import torch
import torch.nn as nn
from network.mobilenet import mobilenet_v2
from network.resnet import resnet18


class basic_model(nn.Module):
    def __init__(self, config):
        super(basic_model, self).__init__()

        self.hidden_dim = config.model.hidden_dim
        self.input_dim = config.data.input_dim
        self.output_dim = config.data.output_dim
        self.p = config.model.dropout_p
        self.activation_func = activation[config.model.activation_func]
        self.num_layers = config.model.num_layers

    def forward(self):
        raise NotImplementedError


class basic_MLP(basic_model):
    def __init__(self, config):
        super(basic_MLP, self).__init__(config)

        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            self.activation_func(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation_func(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x, m):
        return self.backbone(x), m


class dropout_MLP(basic_model):
    def __init__(self, config):
        super(dropout_MLP, self).__init__(config)

        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            self.activation_func(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation_func(),
            nn.Dropout(self.p),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x, m):
        return self.backbone(x), m


class basic_GRU(basic_model):
    def __init__(self, config):
        super(basic_GRU, self).__init__(config)

        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)

    def forward(self, x, h0):
        x = self.activation(self.linear_in(x))
        if type(h0) == type(None):
            h0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).to(
                x.device
            )
        x, hn = self.gru(x, h0)
        x = self.linear_out(self.activation(x))
        return x, hn


class dropout_GRU(basic_model):
    def __init__(self, config):
        super(dropout_GRU, self).__init__(config)

        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(self.p)

    def forward(self, x, h0):
        x = self.activation(self.linear_in(x))
        if type(h0) == type(None):
            h0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).to(
                x.device
            )
        x, hn = self.gru(x, h0)
        x = self.dropout(self.activation(x))
        x = self.linear_out(x)
        return x, hn


class basic_LSTM(basic_model):
    def __init__(self, config):
        super(basic_LSTM, self).__init__(config)

        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)

    def forward(self, x, m):
        x = self.activation(self.linear_in(x))
        if isinstance(m, tuple):
            h0 = m[0]
            c0 = m[1]
        else:
            h0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).to(
                x.device
            )
            c0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).to(
                x.device
            )
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.linear_out(self.activation(x))
        return x, (hn, cn)


class dropout_LSTM(basic_model):
    def __init__(self, config):
        super(dropout_LSTM, self).__init__(config)

        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(self.p)

    def forward(self, x, m):
        x = self.activation(self.linear_in(x))
        if isinstance(m, tuple):
            h0 = m[0]
            c0 = m[1]
        else:
            h0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).to(
                x.device
            )
            c0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim)).to(
                x.device
            )
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.dropout(self.activation(x))
        x = self.linear_out(x)
        return x, (hn, cn)


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
            self.cfg
        )

    def forward(self, x, m=None):
        return self.backbone(x, m)


backbone = {
    "resnet18": resnet18,
    "mobilenet": mobilenet_v2,
    "fc": basic_MLP,
    "gru": basic_GRU,
    "lstm": basic_LSTM,
    "dfc": dropout_MLP,
    "dgru": dropout_GRU,
    "dlstm": dropout_LSTM,
}

activation = {
    "tanh": nn.Tanh,
    "sigm": nn.Sigmoid,
    "relu": nn.ReLU,
}
