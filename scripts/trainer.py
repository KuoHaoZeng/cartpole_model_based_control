import torch, progressbar, sys, os
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from network import network
from data import dataset

model_protocol = {"state": network.model_state, "image": network.model_image}
dataset_protocol = {"state": dataset.state_dataset, "image": dataset.image_dataset}


class Trainer:
    def __init__(self, config):

        ### somethings
        self.cfg = config
        self.dataset = dataset_protocol[config.data.protocol](config)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.data.batch_size,
            num_workers=config.framework.num_thread,
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

    def load_checkpoints(self, config, model):
        sd = torch.load(
            "{}/{}.pt".format(config.checkpoint_dir, config.checkpoint_file),
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(sd["parameters"])

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
                    imgs, s, x, y = (
                        imgs.to(device=0),
                        s.to(device=0),
                        x.to(device=0),
                        y.to(device=0),
                    )
                    if isinstance(self.dataset, dataset.image_dataset):
                        imgs = imgs.to(device=0)
                y_action = x[:, :, -1]
                x = x[:, :, :-1]

                # forward
                if isinstance(self.dataset, dataset.image_dataset):
                    p, _ = self.model(imgs)
                else:
                    p, _ = self.model(x)
                p = p.view_as(y_action)

                # loss
                l = self.criterion(p, y_action)

                # backprop
                self.model.zero_grad()
                l.backward()
                self.optimizer.step()

                # log
                self.logger.add_scalar(
                    "{}/loss".format(self.cfg.mode),
                    l.data,
                    idx
                    + (
                        self.cfg.data.num_datapoints_per_epoch
                        / self.cfg.data.batch_size
                    )
                    * epoch,
                )
                losses.append(l.detach().cpu().numpy())

            if epoch % self.cfg.train.save_iter == 0:
                self.save_checkpoints(losses)
            self.scheduler.step()
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
                    if isinstance(self.dataset, dataset.image_dataset):
                        imgs = imgs.to(device=0)
                y_action = x[:, :, -1]

                # forward
                if isinstance(self.dataset, dataset.image_dataset):
                    p, _ = self.model(imgs, y_action)
                else:
                    p, _ = self.model(x)
                p = p.view_as(y)

                # loss
                l = self.criterion(p, y)

                # backprop
                self.model.zero_grad()
                l.backward()
                self.optimizer.step()

                # log
                self.logger.add_scalar(
                    "{}/loss".format(self.cfg.mode),
                    l.data,
                    idx
                    + (
                        self.cfg.data.num_datapoints_per_epoch
                        / self.cfg.data.batch_size
                    )
                    * epoch,
                )
                losses.append(l.detach().cpu().numpy())
            if epoch % self.cfg.train.save_iter == 0:
                self.save_checkpoints(losses)
            self.scheduler.step()
            self.bar.update(epoch)
        print("finish!")


class Trainer_model_predictive_policy_learning(Trainer):
    def __init__(self, configs):
        super(Trainer_model_predictive_policy_learning, self).__init__(configs)

        self.dm_model = model_protocol[configs.dm_model.model.protocol](
            configs.dm_model
        )
        self.load_checkpoints(configs.dm_model, self.dm_model)
        if configs.framework.num_gpu > 0:
            self.dm_model.to(device=0)
        # self.dm_model.eval()
        # if configs.dm_model.model.backbone.startswith("d"):
        #    self.turn_on_dropout(self.dm_model)

        self.criterion = nn.L1Loss()

    def turn_on_dropout(self, model):
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def augmented_state(self, state):
        """
        :param state: cartpole state
        :param action: action applied to state
        :return: an augmented state for training GP dynamics
        """
        dtheta, dx, theta, x = (
            state[:, :, 0],
            state[:, :, 1],
            state[:, :, 2],
            state[:, :, 3],
        )
        return torch.cat(
            [
                x.unsqueeze(2),
                dx.unsqueeze(2),
                dtheta.unsqueeze(2),
                torch.sin(theta).unsqueeze(2),
                torch.cos(theta).unsqueeze(2),
            ],
            dim=2,
        )

    def cov(self, m):
        mean = torch.mean(m, dim=0)
        m = m - mean
        cov = m.transpose(0, 1).mm(m)
        return cov

    def run(self):
        losses = []
        for epoch in range(1, self.cfg.train.num_epoch + 1):
            for idx, (imgs, s, x, y) in enumerate(self.dataloader):
                if self.cfg.framework.num_gpu > 0:
                    s, x, y = s.to(device=0), x.to(device=0), y.to(device=0)

                delta_states = []
                state = s[:, 0, :].unsqueeze(1)
                policy_memory, dm_memory = None, None
                loss_policy = 0
                for t in range(self.cfg.data.horizon):
                    y_action = []
                    for b in range(self.cfg.data.batch_size):
                        gt_a = self.dataset.policy.predict(
                            state[b, 0, :].detach().cpu().numpy()
                        )
                        y_action.append(gt_a)
                    y_action = torch.Tensor(y_action)
                    if self.cfg.framework.num_gpu > 0:
                        y_action = y_action.to(device=0)

                    policy_state = self.augmented_state(state)

                    pred_a, policy_memory = self.model(policy_state, policy_memory)
                    p = pred_a.view_as(y_action)
                    loss_policy += self.criterion(p, y_action)

                    delta_state_n = []
                    for n in range(self.cfg.dm_model.data.num_traj_samples):
                        dm_state = torch.cat([policy_state, pred_a], dim=2)
                        delta_state, dm_memory_n = self.dm_model(dm_state, dm_memory)
                        delta_state_n.append(delta_state.unsqueeze(0))
                    delta_state_n = torch.cat(delta_state_n, dim=0)
                    delta_states.append(delta_state_n)
                    dm_memory = dm_memory_n

                    # state += delta_state.detach()
                    next_state = []
                    for b in range(self.cfg.data.batch_size):
                        next_s = self.dataset.sim.step(
                            state[b, 0, :].detach().cpu().numpy(),
                            [pred_a[b, 0, 0].detach().cpu().numpy()],
                            noisy=True,
                        )
                        next_state.append(next_s)
                    state = torch.Tensor(np.array(next_state)).unsqueeze(1)
                    if self.cfg.framework.num_gpu > 0:
                        state = state.to(device=0)
                delta_states = torch.cat(delta_states, dim=2)
                delta_states = delta_states.view(
                    self.cfg.dm_model.data.num_traj_samples, -1
                )
                cov = self.cov(delta_states)
                loss_uncertainty = cov.trace() / (
                    self.cfg.data.horizon
                    * self.cfg.data.batch_size
                    * self.cfg.dm_model.data.output_dim
                    * self.cfg.dm_model.data.num_traj_samples
                )
                loss_policy = loss_policy / (self.cfg.data.horizon)
                l = loss_policy + self.cfg.train.LAMBDA * loss_uncertainty

                # backprop
                self.model.zero_grad()
                l.backward()
                self.optimizer.step()

                # log
                self.logger.add_scalar(
                    "{}/loss".format(self.cfg.mode),
                    l.data,
                    idx
                    + (
                        self.cfg.data.num_datapoints_per_epoch
                        / self.cfg.data.batch_size
                    )
                    * epoch,
                )
                self.logger.add_scalar(
                    "{}/loss_policy".format(self.cfg.mode),
                    loss_policy.data,
                    idx
                    + (
                        self.cfg.data.num_datapoints_per_epoch
                        / self.cfg.data.batch_size
                    )
                    * epoch,
                )
                self.logger.add_scalar(
                    "{}/loss_uncertainty".format(self.cfg.mode),
                    loss_uncertainty.data,
                    idx
                    + (
                        self.cfg.data.num_datapoints_per_epoch
                        / self.cfg.data.batch_size
                    )
                    * epoch,
                )
                losses.append(l.detach().cpu().numpy())
            if epoch % self.cfg.train.save_iter == 0:
                self.save_checkpoints(losses)
            self.scheduler.step()
            self.bar.update(epoch)
        print("finish!")
