import torch
import numpy as np
from torch.utils.data import Dataset
from data.cartpole_sim import CartpoleSim
from data.policy import SwingUpAndBalancePolicy, RandomPolicy
from data.cartpole_test import sim_rollout, make_training_data
from data.visualization import CartpoleVisualizer

protocol = {"random": RandomPolicy, "swing_up": SwingUpAndBalancePolicy}


class state_dataset(Dataset):
    def __init__(self, cfg):

        # store useful things ...
        self.cfg = cfg
        self.length = cfg.data.num_datapoints_per_epoch
        self.horizon = cfg.data.horizon
        self.delta_t = cfg.data.delta_t
        self.rng = np.random.RandomState(cfg.framework.seed)

        # allocate the default initial state
        self.default_init_state = np.array(cfg.data.default_init_state)
        self.default_init_state[2] *= np.pi

        # allocate default parameters
        self.state_dim = 4
        self.aug_state_dim = 6

        # allocate the rollout policy
        # the seed used in HW1 is 12831
        self.policy = protocol[cfg.data.expert_policy](
            cfg.framework.seed, cfg.data.policy_dir
        )

        # allocate the simulator
        self.sim = CartpoleSim(dt=self.delta_t)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        s, x, y = self.generate_data()
        return torch.Tensor(s), torch.Tensor(x), torch.Tensor(y)

    def generate_data(self):
        # normal state: [dtheta, dp, theta, p]
        # augmented state: [dtheta, dp, sin(theta), cos(theta), p, u]
        # delta state: [Δdtheta, Δp, Δtheta, Δp]
        # s: a tensor of size [horizon x 4], which indicates current state (dim=4) before augmentation
        # x: a tensor of size [horizon x 6], which indicates input augmented state (dim=5) + action (dim=1)
        # y: a tensor of size [horizon x 4], which indicates output delta state (dim=4)

        init_state = self.default_init_state * self.rng.randn(self.state_dim)
        ts, state_traj, action_traj = sim_rollout(
            self.sim, self.policy, self.horizon, self.delta_t, init_state
        )
        delta_state_traj = state_traj[1:] - state_traj[:-1]

        x, y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)
        s = state_traj.copy()
        return s, x, y


class image_dataset(state_dataset):
    def __init__(self, cfg):
        super(image_dataset, self).__init__(cfg)

        self.alpha = cfg.data.image.alpha
        self.vis = CartpoleVisualizer(
            cfg.data.image.cart_width,
            cfg.data.image.cart_height,
            cfg.data.image.pole_length,
            cfg.data.image.pole_thickness,
            tuple(cfg.data.image.figsize),
        )

    def __getitem__(self, index):
        s, x, y = self.generate_data()
        imgs = []
        for h in range(self.horizon):
            img = self.vis.draw_cartpole("", s[h, 3], s[h, 2], self.alpha)
            imgs.append(img)
        return torch.Tensor(imgs), torch.Tensor(s), torch.Tensor(x), torch.Tensor(y)
