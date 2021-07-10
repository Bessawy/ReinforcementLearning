
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from utils import discount_rewards
import numpy as np


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 16
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        # TODO: Add another linear layer for the critic
        self.fc2_value = torch.nn.Linear(self.hidden, action_space)
        self.var = torch.nn.Parameter(torch.tensor([10.0])) # TODO: Implement learned variance (or copy from Ex5)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Common part
        x = self.fc1(x)
        x = F.relu(x)

        # Actor part
        action_mean = self.fc2_mean(x)
        sigma = self.var  # TODO: Implement (or copy from Ex5)

        # Critic part
        # TODO: Implement
        value = self.fc2_value(x)
        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma
        # Implement or copy from Ex5
        action_dist = Normal(action_mean, sigma)
        # TODO: Return state value in addition to the distribution

        return action_dist, value


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.done = []

    def update_policy(self, episode_number):
        # Convert buffers to Torch tensors
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        # Clear state transition buffers
        self.states, self.action_probs, self.rewards = [], [], []
        self.next_states, self.done = [], []



        # TODO: Compute state values
        _, value = self.policy.forward(states)
        #discounted = discount_rewards(rewards, 1) #episodic
        value = torch.reshape(value, (-1,))
        _, value_next = self.policy.forward(next_states)
        value_next = torch.reshape(value_next, (-1,))
        value_next = torch.where(done.byte(), torch.tensor(0.0), value_next)
        y = rewards + self.gamma *value_next  #non-episodic
        #print(value_next)
        #print(done)
        #print(value.shape)
        #print(rewards.shape)

        # TODO: Compute critic loss (MSE)
        #E = value - discounted   # episodic
        E = value - y.detach() #non_epiosdic
        SE = torch.pow(E, 2)
        MSE = torch.mean(SE)
        #print(MSE)

        # Advantage estimates
        # TODO: Compute advantage estimates
        Adv = rewards + self.gamma * value_next - value
        #print(Adv.shape)

        # TODO: Calculate actor loss (very similar to PG)
        weighted_probs = -action_probs * Adv.detach()
        loss = torch.mean(weighted_probs)

        # TODO: Compute the gradients of loss w.r.t. network parameters
        # Or copy from Ex5
        t_loss = loss + MSE

        #MSE.backward()
        #loss.backward()
        t_loss.backward()

        # TODO: Update network parameters using self.optimizer and zero gradients
        # Or copy from Ex5
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network
        # Or copy from Ex5
        aprob, _ = self.policy.forward(x)
        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy
        # Or copy from Ex5
        if evaluation:
            action = torch.tensor(aprob.mean)
        else:
            action = torch.tensor(aprob.sample())
        # TODO: Calculate the log probability of the action
        # Or copy from Ex5
        act_log_prob = aprob.log_prob(action)
        return action, act_log_prob

    def store_outcome(self, state, next_state, action_prob, reward, done):
        # Now we need to store some more information than with PG
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
