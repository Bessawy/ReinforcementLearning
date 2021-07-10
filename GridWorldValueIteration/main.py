# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################


import numpy as np
from time import sleep
from sailing import SailingGridworld

epsilon = 10e-4  # TODO: Use this criteria for Task 3


# Set up the environment
env = SailingGridworld(rock_penalty=-2)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)


def bellman_optimality_update(env, V, w, h, gamma):
    min = float('-inf')
    Vnext = 0
    actions = [env.UP, env.DOWN, env.RIGHT, env.LEFT]

    for a in actions:
        transition = env.transitions[w, h, a]
        Vaction = 0
        if (transition[0].state != None):
            for t in range(len(transition)):
                Vnext = V[transition[t].state[0], transition[t].state[1]]
                Vaction = Vaction + transition[t].prob * (transition[t].reward + gamma * Vnext)
                #print(Vaction)
            if Vaction > min:
                min = Vaction
        else:
            min = 0

    return min


def q_greedify_policy(env, V, w, h, gamma):
    min = float('-inf')
    argmax = 0
    actions = [env.UP, env.DOWN, env.RIGHT, env.LEFT]
    for a in actions:
        transition = env.transitions[w, h, a]
        Vaction = 0
        if(transition[0].state != None):
            for t in range(len(transition)):
                    Vnext = V[transition[t].state[0], transition[t].state[1]]
                    Vaction = Vaction + transition[t].prob * (transition[t].reward + gamma * Vnext)
            if Vaction > min:
                min = Vaction
                argmax = a
        else:
            argamx = 0
    return argmax


def value_interation(env, gamma, theta):
    V = np.zeros((env.w, env.h))
    #use for loop only or (whilte true, and deltas)
    #for i in range(25):
    while True:
        delta = 0
        for w in range(env.w):
            for h in range(env.h):
                v = V[w, h]
                V[w, h] = bellman_optimality_update(env, V, w, h, gamma)

        #Deltas
                delta = max(delta, abs(v - V[w, h]))
        if delta < theta:
            break

        #env.clear_text()
        #env.draw_values(V)
        #env.render()
        #sleep(1)

    pi = np.zeros((env.w, env.h))
    for w in range(env.w):
        for h in range(env.h):
            pi[w, h] = q_greedify_policy(env, V, w, h, gamma)
    return V, pi




if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    #value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))

    gamma = 0.9
    theta = epsilon
    value_est, policy = value_interation(env, gamma, theta)


    #value_est[14, 9] = bellman_optimality_update(env, value_est, 14, 9, gamma)
    #print(value_est[14, 9])
    #Transition = env.transitions[13, 9, env.UP]
    #print(len(Transition))
    #print(Transition)


    # Show the values and the policy

    #env.draw_values(value_est)
    #env.draw_actions(policy)
    #env.render()
    #sleep(10)

    # Save the state values and the policy

    fnames = "values.npy", "policy.npy" 
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)


    cum_returns = np.zeros(1000, dtype=float)

    
    # Run a single episode
    print("ENTER")
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)
    for i in range(1000):
        state = env.reset()
        cum_return = 0
        power = 0
        done = False
        while not done:
            # Select a random action
            # TODO: Use the policy to take the optimal action (Task 2)
            #action = int(np.random.random()*4)
            action = int(policy[state[0], state[1]])
            # Step the environment
            state, reward, done, _ = env.step(action)
            cum_return = cum_return + pow(gamma, power)*reward
            power = power + 1
            # Render and sleep
           # env.render()
            #sleep(0.5)
        #print(cum_return)
        cum_returns[i] = cum_return


    #print(cum_returns)
    print("The mean is equal", np.mean(cum_returns))
    print("STD", np.std(cum_returns))
        #print(cum_returns)

