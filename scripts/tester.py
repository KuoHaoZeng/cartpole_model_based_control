import torch, progressbar
import torch.nn as nn
import numpy as np

from network import network
from data import dataset
from data.visualization import Visualizer
import cv2
import matplotlib.pyplot as plt

plt.style.use("ggplot")

model_protocol = {"state": network.model_state, "image": network.model_CNN}
dataset_protocol = {"state": dataset.state_dataset, "image": dataset.image_dataset}


class Tester:
    def __init__(self, config):
        ### somethings
        self.cfg = config
        self.dataset = dataset_protocol[config.data.protocol](config)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=config.data.batch_size, num_workers=4,
        )
        widgets = [
            "Testing phase [",
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
            max_value=config.data.batch_size, widgets=widgets, term_width=100
        )

        ### model
        self.model = model_protocol[config.model.protocol](config)
        self.load_checkpoints()
        if config.framework.num_gpu > 0:
            self.model.to(device=0)
        self.model.eval()

        ### visualization
        self.vis = Visualizer(
            cartpole_length=1.5,
            x_lim=(0.0, config.data.delta_t * config.data.num_datapoints_per_epoch),
            figsize=(6, 8),
            gt_title=config.test.gt_title,
            model_title=config.test.model_title,
        )

    def load_checkpoints(self):
        sd = torch.load(
            "{}/{}.pt".format(self.cfg.checkpoint_dir, self.cfg.checkpoint_file),
            map_location=torch.device("cpu"),
        )
        self.model.load_state_dict(sd["parameters"])

    def run(self):
        raise NotImplementedError


class Tester_policy(Tester):
    def __init__(self, configs):
        super(Tester_policy, self).__init__(configs)

        ### define loss functions
        self.criterion = nn.MSELoss()

    def sim_rollout(self, sim, policy, n_step, dt, init_state):
        states = []
        state = init_state
        actions = []
        for i in range(n_step):
            states.append(state)
            inp = self.augmented_state(state)
            inp = torch.Tensor(inp).unsqueeze(0).unsqueeze(0)
            if self.cfg.framework.num_gpu > 0:
                inp = inp.to(device=0)
            if i == 0:
                action, m = policy(inp)
            else:
                action, m = policy(inp, m)
            action = action.detach().cpu().numpy()
            actions.append(action)
            state = sim.step(state, [action], noisy=True)
        states.append(state)
        time = np.arange(n_step + 1) * dt
        return time, np.array(states), np.array(actions)

    def augmented_state(self, state):
        """
        :param state: cartpole state
        :param action: action applied to state
        :return: an augmented state for training GP dynamics
        """
        dtheta, dx, theta, x = state
        return np.array([x, dx, dtheta, np.sin(theta), np.cos(theta)])

    def run(self):
        losses, rollout_losses = [], []
        for idx, (imgs, s, x, y) in enumerate(self.dataloader):
            if self.cfg.framework.num_gpu > 0:
                s, x, y = s.to(device=0), x.to(device=0), y.to(device=0)
            y_action = x[:, :, -1]
            gt_action = x[:, :, -1].cpu().numpy()
            x = x[:, :, :-1]

            # forward
            p, _ = self.model(x)
            p = p.view_as(y_action)

            # loss
            l = self.criterion(p, y_action)

            losses.append(l.detach().cpu().numpy())

            # rollout
            for j in range(len(s)):
                # simulate with trained policy
                state_traj_rollout, action_traj_rollout = [], []
                for n in range(self.cfg.data.num_traj_samples):
                    # rollout #num_traj_samples trajectories
                    ts, state_traj, action_traj = self.sim_rollout(
                        self.dataset.sim,
                        self.model,
                        self.cfg.data.num_datapoints_per_epoch,
                        self.cfg.data.delta_t,
                        s[j][0].detach().cpu().numpy(),
                    )
                    state_traj_rollout.append(state_traj)
                    action_traj_rollout.append(action_traj)
                state_traj_rollout = np.array(state_traj_rollout)
                delta_state_traj_rollout = (
                    state_traj_rollout[:, 1:] - state_traj_rollout[:, :-1]
                )
                state_traj_mean = state_traj_rollout.mean(0)
                delta_state_traj_mean = delta_state_traj_rollout.mean(0)
                delta_delta_state_traj_var = delta_state_traj_rollout.var(0)
                action_traj_rollout = np.array(action_traj_rollout)[:, :, 0, 0, 0]

                # compute rollout loss w/ swing up policy
                action_rollout_loss = (
                    ((action_traj_rollout - gt_action[j]) ** 2) ** 0.5
                ).mean()
                rollout_losses.append(action_rollout_loss)

                # make a video
                if self.cfg.test.film:
                    self.vis.clear()
                    state_traj = s[j].cpu().numpy()
                    for i in range(len(s[j]) - 1):
                        delta_state = state_traj[1:] - state_traj[:-1]
                        self.vis.set_gt_cartpole_state(
                            state_traj[i][3], state_traj[i][2]
                        )
                        self.vis.set_gt_delta_state_trajectory(
                            ts[: i + 1], delta_state[: i + 1]
                        )

                        self.vis.set_gp_cartpole_state(
                            state_traj_mean[i][3], state_traj_mean[i][2]
                        )
                        self.vis.set_gp_cartpole_rollout_state(
                            state_traj_rollout[:, i, 3], state_traj_rollout[:, i, 2],
                        )

                        self.vis.set_gp_delta_state_trajectory(
                            ts[: i + 1],
                            delta_state_traj_mean[: i + 1],
                            delta_delta_state_traj_var[: i + 1],
                        )

                        self.vis.set_info_text(
                            "trajectory: {}\npolicy model: {}".format(
                                j, self.cfg.model.backbone
                            )
                        )
                        vis_img = self.vis.draw(redraw=(i == 0))
                        # cv2.imshow("vis", vis_img)

                        if idx == 0 and j == 0 and i == 0:
                            video_out = cv2.VideoWriter(
                                "{}/{}_policy.mp4".format(
                                    self.cfg.base_dir, self.cfg.checkpoint_file
                                ),
                                cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                int(1.0 / self.cfg.data.delta_t),
                                (vis_img.shape[1], vis_img.shape[0]),
                            )
                        video_out.write(vis_img)
                        cv2.waitKey(int(1000 * self.cfg.data.delta_t))

                self.bar.update(j + 1)
        print("finish!")
        print("L2 loss: {:.2f}±{:.2f}".format(np.mean(losses), np.std(losses)))
        print(
            "rollout L2 loss: {:.2f}±{:.2f}".format(
                np.mean(rollout_losses), np.std(rollout_losses)
            )
        )


class Tester_dynamic_model(Tester):
    def __init__(self, configs):
        super(Tester_dynamic_model, self).__init__(configs)

        ### define loss functions
        self.criterion = nn.MSELoss()

    def sim_rollout(self, dm, n_step, dt, init_state, actions):
        states = []
        state = init_state
        for i in range(n_step):
            states.append(state)
            inp = self.augmented_state(state, actions[i])
            inp = torch.Tensor(inp).unsqueeze(0).unsqueeze(0)
            if self.cfg.framework.num_gpu > 0:
                inp = inp.to(device=0)
            if i == 0:
                delta_state, h = dm(inp)
            else:
                delta_state, h = dm(inp, h)

            delta_state = delta_state.detach().cpu().numpy()[0, 0, :]
            state = state + delta_state
        states.append(state)
        time = np.arange(n_step + 1) * dt
        return time, np.array(states)

    def augmented_state(self, state, action):
        """
        :param state: cartpole state
        :param action: action applied to state
        :return: an augmented state for training GP dynamics
        """
        dtheta, dx, theta, x = state
        return x, dx, dtheta, np.sin(theta), np.cos(theta), action

    def run(self):
        losses, rollout_losses = [], []
        for idx, (imgs, s, x, y) in enumerate(self.dataloader):
            if self.cfg.framework.num_gpu > 0:
                s, x, y = s.to(device=0), x.to(device=0), y.to(device=0)

            gt_action = x[:, :, -1].cpu().numpy()
            # forward
            p, _ = self.model(x)
            p = p.view_as(y)

            # loss
            l = self.criterion(p, y)

            losses.append(l.detach().cpu().numpy())

            # rollout
            for j in range(len(s)):
                # simulate with trained policy
                state_traj_rollout = []
                for n in range(self.cfg.data.num_traj_samples):
                    # rollout #num_traj_samples trajectories
                    ts, state_traj = self.sim_rollout(
                        self.model,
                        self.cfg.data.num_datapoints_per_epoch,
                        self.cfg.data.delta_t,
                        s[j][0].detach().cpu().numpy(),
                        gt_action[j],
                    )
                    state_traj_rollout.append(state_traj)

                state_traj_rollout = np.array(state_traj_rollout)
                delta_state_traj_rollout = (
                    state_traj_rollout[:, 1:] - state_traj_rollout[:, :-1]
                )
                state_traj_mean = state_traj_rollout.mean(0)
                delta_state_traj_mean = delta_state_traj_rollout.mean(0)
                delta_delta_state_traj_var = delta_state_traj_rollout.var(0)

                # compute rollout loss w/ swing up policy
                # this loss is based on normal state. It's usually larger because we didn't optimize the model on it.
                # state_rollout_loss = (
                #    ((state_traj_rollout - s[j].cpu().numpy()) ** 2) ** 0.5
                # ).mean()
                # this loss is based on delta normal state and it's what the model was optimized for.
                state_rollout_loss = (
                    (
                        (
                            (state_traj_rollout[:, 1:] - state_traj_rollout[:, :-1])
                            - (s[j, 1:] - s[j, :-1]).cpu().numpy()
                        )
                        ** 2
                    )
                    ** 0.5
                ).mean()
                rollout_losses.append(state_rollout_loss)

                # make a video
                if self.cfg.test.film:
                    self.vis.clear()
                    state_traj = s[j].cpu().numpy()
                    for i in range(len(s[j]) - 1):
                        delta_state = state_traj[1:] - state_traj[:-1]
                        self.vis.set_gt_cartpole_state(
                            state_traj[i][3], state_traj[i][2]
                        )
                        self.vis.set_gt_delta_state_trajectory(
                            ts[: i + 1], delta_state[: i + 1]
                        )

                        self.vis.set_gp_cartpole_state(
                            state_traj_mean[i][3], state_traj_mean[i][2]
                        )
                        self.vis.set_gp_cartpole_rollout_state(
                            state_traj_rollout[:, i, 3], state_traj_rollout[:, i, 2],
                        )

                        self.vis.set_gp_delta_state_trajectory(
                            ts[: i + 1],
                            delta_state_traj_mean[: i + 1],
                            delta_delta_state_traj_var[: i + 1],
                        )

                        self.vis.set_info_text(
                            "trajectory: {}\npolicy model: {}".format(
                                j, self.cfg.model.backbone
                            )
                        )
                        vis_img = self.vis.draw(redraw=(i == 0))
                        # cv2.imshow("vis", vis_img)

                        if idx == 0 and j == 0 and i == 0:
                            video_out = cv2.VideoWriter(
                                "{}/{}_policy.mp4".format(
                                    self.cfg.base_dir, self.cfg.checkpoint_file
                                ),
                                cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                int(1.0 / self.cfg.data.delta_t),
                                (vis_img.shape[1], vis_img.shape[0]),
                            )
                        video_out.write(vis_img)
                        cv2.waitKey(int(1000 * self.cfg.data.delta_t))

                self.bar.update(j + 1)
        print("finish!")
        print("L2 loss: {:.2f}±{:.2f}".format(np.mean(losses), np.std(losses)))
        print(
            "rollout L2 loss: {:.2f}±{:.2f}".format(
                np.mean(rollout_losses), np.std(rollout_losses)
            )
        )
