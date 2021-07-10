import gym
import numpy as np
from matplotlib import pyplot as plt
from rbf_agent import Agent as RBFAgent  # Use for Tasks 1-3
from dqn_agent import Agent as DQNAgent  # Task 4
from itertools import count
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import plot_rewards
from matplotlib import cm

env_name = "CartPole-v0"
#env_name = "LunarLander-v2"
env = gym.make(env_name)
env.reset()

# Set hyperparameters
# Values for RBF (Tasks 1-3)
glie_a = 50
num_episodes = 1000

# Values for DQN  (Task 4)
'''
if "CartPole" in env_name:
     TARGET_UPDATE = 50
     glie_a = 500
     num_episodes = 2000
     hidden = 12
     gamma = 0.95
     replay_buffer_size = 500000
     batch_size = 256
elif "LunarLander" in env_name:
     TARGET_UPDATE = 20
     glie_a = 5000
     num_episodes = 15000
     hidden = 64
     gamma = 0.95
     replay_buffer_size = 50000
     batch_size = 128
'''
# else:
#     raise ValueError("Please provide hyperparameters for %s" % env_name)

# The output will be written to your folder ./runs/CURRENT_DATETIME_HOSTNAME,
# Where # is the consecutive number the script was run
writer = SummaryWriter("runs/batch32_hid12_gamma98_Adam_Hubber_glie250")

# Get number of actions from gym action space
n_actions = env.action_space.n
state_space_dim = env.observation_space.shape[0]

# Tasks 1-3 - RBF
agent = RBFAgent(n_actions)

# Task 4 - DQN
#agent = DQNAgent(env_name, state_space_dim, n_actions, replay_buffer_size, batch_size, hidden, gamma)

# Training loop
cumulative_rewards = []
for ep in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    done = False
    eps = glie_a/(glie_a+ep)
    cum_reward = 0
    while not done:
        # Select and perform an action
        action = agent.get_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward

        # Task 1: TODO: Update the Q-values
        #agent.single_update(state, action, next_state, reward, done)
        # Task 2: TODO: Store transition and batch-update Q-values
        agent.store_transition(state, action, next_state, reward, done)
        agent.update_estimator()
        # Task 4: Update the DQN
        #agent.store_transition(state, action, next_state, reward, done)
        #agent.update_network()

        # Move to the next state
        state = next_state
    cumulative_rewards.append(cum_reward)
    writer.add_scalar('Training ' + env_name, cum_reward, ep)
    writer.flush()
    # Update the target network, copying all weights and biases in DQN
    # Uncomment for Task 4
    '''
    if ep % TARGET_UPDATE == 0:
        agent.update_target_network()

    # Save the policy
    # Uncomment for Task 4
    if ep % 1000 == 0:
        torch.save(agent.policy_net.state_dict(),
        "weights_%s_%d.mdl" % (env_name, ep))
'''

plot_rewards(cumulative_rewards)
print('Complete')
plt.ioff()
plt.show()

# Task 3 - plot the policy

fig = plt.figure()
ax = fig.add_subplot(111)
#obs_limit = np.array([4.8, 5, 0.5, 5])
N = 100
x = np.linspace(-4.8, 4.8, N)
theta = np.linspace(-0.5, 0.5, N)
a, b = np.meshgrid(x, theta)

v = np.zeros((1, N*N))
theta_dot = np.zeros((1, N*N))
positions = np.array([a.flatten()])
thetas = np.array([b.flatten()])

States = np.concatenate((positions, v, thetas, theta_dot)).T
featurized = agent.featurize(States)


policy = []
for i in range(N*N):
    qs = [q.predict([featurized[i]])[0] for q in agent.q_functions]
    qs = np.array(qs)
    act = np.argmax(qs, axis=0)
    policy.append(act)

policy = np.array([policy])
policy = policy.reshape(N,N)

ax.contourf(a,b,policy)
plt.xlabel('Position')
plt.ylabel('angle')
plt.show()







