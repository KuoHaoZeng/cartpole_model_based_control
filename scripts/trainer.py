import torch, progressbar, sys, os
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from network import network
from data import dataset

model_protocol = {"state": network.model_state, "image": network.model_CNN}
dataset_protocol = {"state": dataset.state_dataset, "image": dataset.image_dataset}


class Trainer:
    def __init__(self, config):

        ### somethings
        self.cfg = config
        self.dataset = dataset_protocol[config.data.protocol](config)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=config.data.batch_size, num_workers=4,
        )
        widgets = [
            "Training phase [",
            progressbar.SimpleProgress(),
            "] [",
            progressbar.Percentage(),
            "] ",
            progressbar.Bar(marker="█"),
            " (",
            progressbar.Timer(),
            " ",
            progressbar.ETA(),
            ") ",
        ]
        self.bar = progressbar.ProgressBar(
            max_value=config.train.num_epoch, widgets=widgets, term_width=100
        )
        self.best_loss = sys.maxsize

        ### logging
        self.logger = SummaryWriter("{}/runs_{}".format(config.base_dir, config.mode))

        ### model
        self.model = model_protocol[config.model.protocol](config)
        if config.framework.num_gpu > 0:
            self.model.to(device=0)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.train.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.train.lr_ms, gamma=0.1
        )

    def save_checkpoints(self, losses, avg_period=5):
        if not os.path.isdir(self.cfg.checkpoint_dir):
            os.makedirs(self.cfg.checkpoint_dir)

        sd = {"parameters": self.model.state_dict(), "epoch": len(losses)}
        checkpoint_dir = "{}/{:05d}.pt".format(self.cfg.checkpoint_dir, len(losses))
        torch.save(sd, checkpoint_dir)

        loss = np.mean(losses[-avg_period:])
        if loss < self.best_loss:
            checkpoint_dir = "{}/best_model.pt".format(self.cfg.checkpoint_dir)
            torch.save(sd, checkpoint_dir)
            self.best_loss = loss

    def load_checkpoints(self):
        sd = torch.load(
            "{}/{}.pt".format(self.cfg.checkpoint_dir, self.cfg.checkpoint_file),
            map_location=torch.device("cpu"),
        )
        self.model.load_state_dict(sd["parameters"])

    def run(self):
        raise NotImplementedError


class Trainer_policy(Trainer):
    def __init__(self, configs):
        super(Trainer_policy, self).__init__(configs)

        ### define loss functions
        self.criterion = nn.L1Loss()

    def run(self):
        losses = []
        for epoch in range(1, self.cfg.train.num_epoch + 1):
            for idx, (imgs, s, x, y) in enumerate(self.dataloader):
                if self.cfg.framework.num_gpu > 0:
                    s, x, y = s.to(device=0), x.to(device=0), y.to(device=0)
                y_action = x[:, :, -1]
                x = x[:, :, :-1]

                # forward
                p, _ = self.model(x)
                p = p.view_as(y_action)

                # loss
                l = self.criterion(p, y_action)

                # backprop
                self.model.zero_grad()
                l.backward()
                self.optimizer.step()
                self.scheduler.step()

                # log
                self.logger.add_scalar(
                    "{}/loss".format(self.cfg.mode),
                    l.data,
                    idx + self.cfg.train.num_epoch * epoch,
                )
                losses.append(l.detach().cpu().numpy())

            if epoch % self.cfg.train.save_iter == 0:
                self.save_checkpoints(losses)
            self.bar.update(epoch)
        print("finish!")


class Trainer_dynamic_model(Trainer):
    def __init__(self, configs):
        super(Trainer_dynamic_model, self).__init__(configs)

        ### define loss functions
        self.criterion = nn.L1Loss()

    def run(self):
        losses = []
        for epoch in range(1, self.cfg.train.num_epoch + 1):
            for idx, (imgs, s, x, y) in enumerate(self.dataloader):
                if self.cfg.framework.num_gpu > 0:
                    s, x, y = s.to(device=0), x.to(device=0), y.to(device=0)

                # forward
                p, _ = self.model(x)
                p = p.view_as(y)

                # loss
                l = self.criterion(p, y)

                # backprop
                self.model.zero_grad()
                l.backward()
                self.optimizer.step()
                self.scheduler.step()

                # log
                self.logger.add_scalar(
                    "{}/loss".format(self.cfg.mode),
                    l.data,
                    idx + self.cfg.train.num_epoch * epoch,
                )
                losses.append(l.detach().cpu().numpy())

            if epoch % self.cfg.train.save_iter == 0:
                self.save_checkpoints(losses)
            self.bar.update(epoch)
        print("finish!")
