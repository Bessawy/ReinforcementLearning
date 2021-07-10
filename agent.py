from typing import Dict
import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import Module
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from notsowise import Worker

class Policy(Module):
    def __init__(self):
        super().__init__()
        # 84x84 frame and produces a 20x20 frame
        self.conv1 = nn.Conv2d(in_channels= 4, out_channels=32, kernel_size=8, stride=4)
        # 20x20 frame and produces a 9x9 frame
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # 9x9 frame and produces a 7x7 frame
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # 512 features
        self.lin = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        # A fully connected layer to get policy
        self.pi_logits = nn.Linear(in_features=512, out_features=3)
        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=512, out_features=1)
        self.activation = nn.ReLU()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.conv1(obs))
        h = self.activation(self.conv2(h))
        h = self.activation(self.conv3(h))
        h = h.reshape((-1, 7 * 7 * 64))
        h = self.activation(self.lin(h))
        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)
        return pi, value

def obs_to_torch(obs: np.ndarray, device):
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.

class Agent:
    def __init__(self):
        # number of updates
        self.updates = 10000
        # number of epochs to train the model with sampled data
        self.epochs = 4
        # number of worker processes
        self.n_workers = 64
        # number of steps to run on each process for a single update
        self.worker_steps = 128
        # number of mini batches
        self.n_mini_batch = 4
        # total number of samples for a single update
        self.batch_size = self.n_workers * self.worker_steps
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        # tensorboard
        self.writer = SummaryWriter("runs/batch32_hid12_gamma98_Adam_Hubber_glie250")
        # win-history
        self.win_history = []
        #Gae parameters
        self.lambdas = 0.96
        self.gamma = 0.99
        # create workers
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]
        # initialize workers for observations
        self.obs = np.zeros((self.n_workers, 4, 84, 84), dtype=np.uint8)
        #self.obs2 = np.zeros((self.n_workers, 4, 84, 84), dtype=np.uint8)
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()
            #self.obs[i], self.obs2[i] = worker.child.recv()

        self.device = self.device()
        self.policy = Policy().to(self.device)
        #self.policy2 = Policy().to(device)
        # optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=2e-4)

        self.episode = 0
        self.obs_2 = np.zeros((2, 84, 84))
        self.obs_4 = np.zeros((4, 84, 84))
        self.obs_N = np.zeros((1, 4, 84, 84), dtype=np.uint8)

    def train(self):
        for update in range(self.updates):

            progress = update / self.updates
            learnrate = 2.5e-4 * (1 - progress)
            clip = 0.1 * (1 - progress)

            values = np.zeros((self.n_workers, self.worker_steps + 1), dtype=np.float32)
            log_pi = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
            done = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
            rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
            actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
            obs = np.zeros((self.n_workers, self.worker_steps, 4, 84, 84), dtype=np.uint8)

            with torch.no_grad():
                for t in range(self.worker_steps):
                    obs[:, t] = self.obs

                    # sample action
                    pi, v = self.policy.forward(obs_to_torch(self.obs, self.device))
                    # pi2, _ = self.policy2.forward(obs_to_torch(self.obs2))
                    values[:, t] = v.cpu().numpy()

                    a = pi.sample()
                    # a2 = pi2.sample()
                    # actions2[:, t] = a2.cpu().numpy()
                    # actions that is given to environment
                    actions[:, t] = a.cpu().numpy()
                    # log for loss computation
                    log_pi[:, t] = pi.log_prob(a).cpu().numpy()

                    # run sampled actions on each worker
                    for w, worker in enumerate(self.workers):
                        worker.child.send(("step", actions[w, t]))
                        # worker.child.send(("step", [actions[w, t], actions2[w, t]]))

                    # get results after executing the actions
                    for w, worker in enumerate(self.workers):
                        self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()
                        # self.obs[w], rewards[w, t], done[w, t], self.obs2[w] = worker.child.recv()

                        # assess the win_rate
                        if done[w, t]:
                            self.episode += 1
                            self.win_history.append(0 if rewards[w, t] == -10 else 1)
                            if self.episode % 100 == 0:
                                print("Winrate for the last 100 episode: ", sum(self.win_history), "%")
                                self.writer.add_scalar('Win_rate_simple_ai', sum(self.win_history), self.episode)
                                self.writer.flush()
                                self.win_history = []

                # Get value of after the final step
                _, v = self.policy.forward(obs_to_torch(self.obs, self.device))
                values[:, self.worker_steps] = v.cpu().numpy()

            # calculate advantages for all samples
            gae = 0
            adv = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)

            # value(t+1) for all workers
            value_step = values[:, -1]

            # we go in the reverse order with the number of worker step we have
            for t in reversed(range(self.worker_steps)):
            # mask determine the termination of episode, if done mask is equal zero and
            # thus next step is zero
                mask = 1.0 - done[:, t]
                 # delta
                delta = rewards[:, t] + self.gamma * value_step * mask - values[:, t]
                # gae(t) from gae(t+1)
                gae = delta + self.gamma * self.lambdas * gae * mask
                # save for each time step
                adv[:, t] = gae
                value_step = values[:, t]

            samples = {'advantages': adv, 'actions': actions,'log_pi_old': log_pi,'obs': obs, 'values': values[:, :-1]}

            # samples are currently in `[workers, time_step]` table,
            # we should flatten it for training
            samples_flat = {}
            for k, v in samples.items():
                v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
                if k == 'obs':
                    samples_flat[k] = obs_to_torch(v, self.device)
                else:
                    samples_flat[k] = torch.tensor(v, device=self.device)


            for i in range(self.epochs):
                # shuffle for each epoch
                indexes = torch.randperm(self.batch_size)
                # for each mini batch
                for start in range(0, self.batch_size, self.mini_batch_size):
                    # get mini batch
                    end = start + self.mini_batch_size
                    mini_batch_indexes = indexes[start: end]
                    mini_batch = {}
                    for k, v in samples_flat.items():
                        mini_batch[k] = v[mini_batch_indexes]

                    obs = mini_batch ['obs']
                    action = mini_batch ['actions']
                    adv = mini_batch['advantages']
                    values = mini_batch['values']
                    log_pi_old = mini_batch ['log_pi_old']

                    # commuted return
                    commuted_returns = adv + values
                    # normalize adv
                    adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-10)
                    # commute current policy and value
                    pi, value = self.policy.forward(obs)
                    # commute log policy
                    log_pi_new = pi.log_prob(action)

                    ratio = torch.exp(log_pi_new - log_pi_old)
                    p1 = ratio * adv_normalized
                    p2 = ratio.clamp(min=1.0 - clip, max=1.0 + clip) * adv_normalized
                    policy_loss = -torch.mean(torch.min(p1, p2))

                    # clipped value loss ppo2
                    v1 = (value - commuted_returns) ** 2
                    clipped = values + (value - values).clamp(min=-clip, max=clip)
                    v2 = (clipped - commuted_returns) ** 2
                    critic_loss = torch.mean(torch.max(v1, v2))

                    loss = policy_loss + 0.25 * critic_loss - 0.02 * (pi.entropy().mean())

                    # Set learning rate
                    for mod in self.optimizer.param_groups:
                        mod['lr'] = learnrate

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if update % 500 == 0:
                print("Update Model: ", update)
                self.store_model(update)
                self.optimizer.step()


    def get_name(self):
        return "NotsoWise"

    def get_action(self, observation):
        obs = self._process_obs(observation)
        self.obs_2 = np.roll(self.obs_2, shift=-1, axis=0)
        self.obs_2[-1] = obs
        self.diff = self.obs_2[-1] - self.obs_2[-2]
        self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
        self.obs_4[-1] = self.diff
        #x = torch.from_numpy(observation).float().to(device)
        self.obs_N[0] = self.obs_4
        pi, _ = self.policy.forward(obs_to_torch(self.obs_N, self.device))
        a = pi.sample()
        return a.cpu().numpy()

    def store_model(self, it):
        torch.save(self.policy.state_dict(), str(it) + 'modellast.mdl')

    def load_model(self):
        weights = torch.load("12000modelOpp.mdl")
        #weight2 = torch.load("13500modelfinal.mdl")
        self.policy.load_state_dict(weights, strict=False)
        #self.policy2.load_state_dict(weights, strict=False)
        return "model_loaded"

    def destroy(self):
        for worker in self.workers:
            worker.child.send(("close", None))

    def reset(self):
        self.obs_2 = np.zeros((2, 84, 84))
        self.obs_4 = np.zeros((4, 84, 84))

    @staticmethod
    def _process_obs(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs

    def device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        print(device)
        return device



def main():
    # Initialize the trainer
    m = Agent()
    m.load_model()
    # Run and monitor the experiment
    m.train()
    # Stop the workers
    m.destroy()


# ## Run it
if __name__ == "__main__":
    main()
