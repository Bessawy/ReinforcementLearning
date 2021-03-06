import gym
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
#a = 0  # TODO: Set the correct value.
#at target_eps = 0.1 and K = 20000
a = round((target_eps * 20000)/(1 - target_eps))
epsilon = 0.2
initial_q = 0# T3: Set to 50
K = 0 #kth episodes

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)


#hearmap drawing
def heatmap(q_grid):
    value= np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
    #value_function
    value = np.max(q_grid, axis=4)
    values_array = np.zeros(q_grid.shape[:-3])
    values_array = np.mean(np.mean(value, axis = 3), axis = 1)
    ax = sns.heatmap(values_array, xticklabels=np.around(x_grid,2), yticklabels=np.around(th_grid,2) )
    plt.xlabel("X")
    plt.ylabel("Theta")
    plt.show()

q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q
heatmap(q_grid)

def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av

def get_action(state, q_values, greedy=False):
    #TODO: Implement epsilon-greedy
    cell_index = get_cell_index(state)
    greedyaction = -1
    min = float('-inf')

    epsilon = a / (a + K)
    #epsilon = 0
    # Find greedy action
    for i in range(num_of_actions):
        if q_grid[cell_index[0], cell_index[1], cell_index[2], cell_index[3], i] > min:
            min = q_grid[cell_index[0], cell_index[1], cell_index[2], cell_index[3], i]
            greedyaction = i

    #select between all action with probability epsilon
    if np.random.random() < epsilon:
        action = np.random.randint(0, num_of_actions)
    else:
        action = greedyaction

    return int(action)
    #raise NotImplementedError("Implement epsilon-greedy")


#tested
def update_q_value(old_state, action, new_state, reward, done, q_array):
    # TODO: Implement Q-value update
    if done == False: #for non terninating next state
        old_cell_index = get_cell_index(old_state)
        new_cell_index = get_cell_index(new_state)
        q_old = q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action]
        q_max = np.max(q_array[new_cell_index[0], new_cell_index[1], new_cell_index[2], new_cell_index[3], :])
        q_old = q_old + alpha * (reward + (gamma * q_max) - q_old)
        q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = q_old
    else: #for terninating next state
        old_cell_index = get_cell_index(old_state)
        q_old = q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action]
        q_old = q_old + alpha * (reward - q_old)
        q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = q_old


# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    epsilon = 0.0  # T1: GLIE/constant, T3: Set to 0
    while not done:
        action = get_action(state, q_grid, greedy=test)
        new_state, reward, done, _ = env.step(action)
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid)
        else:
            env.render()
        state = new_state
        steps += 1
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))
    #increment GPIE
    K = K + 1
    if ep == 0 or ep == 10000:
        heatmap(q_grid)


# Save the Q-value array
np.save("q_values_task3_a.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
values = np.zeros(q_grid.shape[:-1])  #TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
#value_function
values = np.max(q_grid, axis=4)

np.save("value_func_task3_a.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY

# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib
heatmap(q_grid)


# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.xlabel("Episodes")
plt.ylabel("timesteps")
plt.show()

