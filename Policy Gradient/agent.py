import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards
import math


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64

        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        #self.fc2_var = torch.nn.Linear(self.hidden, action_space)
        self.var = torch.nn.Parameter(torch.tensor([10.0]))
        self.sigma = torch.tensor(math.sqrt(5.0))  #variance(sigma**2)=5 # TODO: Implement accordingly (T1, T2)
        self.c = 5.0 * math.pow(10, -4)
        self.sigma0 = math.sqrt(10)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)



    def forward(self, x, episode_number):
        v = x
        x = self.fc1(x)
        x = F.relu(x)
        action_mean = self.fc2_mean(x)
        #v = self.fc2_var(x)
        #action_var = F.softplus(v + 10)
        #sigma = torch.sqrt(action_var)

        #sigma = torch.sqrt(self.var)
        #sigma = self.var

        variance = math.pow(self.sigma0, 2) * math.exp(-self.c * episode_number)
        sigma = math.sqrt(variance)
        #sigma = self.sigma  # TODO: Is it a good idea to leave it like this?
        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1)

        #Normal takes std
        action_dist = Normal(action_mean, sigma)

        return action_dist


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"  # ""cuda" if torch.cuda.is_available() else "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        self.states, self.action_probs, self.rewards = [], [], []
        # TODO: Compute discounted rewards (use the discount_rewards function)
        discounted = discount_rewards(rewards, self.gamma)
        discounted -= torch.mean(discounted)
        discounted /= torch.std(discounted)

        # TODO: Compute the optimization term (T1)
        weighted_probs = -action_probs * discounted
        #weighted_probs = -action_probs * (discounted - 20)
        loss = torch.mean(weighted_probs)
        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()



    def get_action(self, observation, episode_number , evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        aprob = self.policy.forward(x, episode_number)
        #print(aprob)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        #print('problem')
        if evaluation:
            action = torch.tenor(aprob.mean)
        else:
            action = torch.tensor(aprob.sample().item())
        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = aprob.log_prob(action)
        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

