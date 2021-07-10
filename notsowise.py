import multiprocessing
import multiprocessing.connection
import cv2
import gym
import numpy as np
import wimblepong

class Pong:
    def __init__(self, seed: int):
        self.env = gym.make("WimblepongVisualMultiplayer-v0")
        self.env.seed(seed)
        self.player_id = 1
        self.opponent_id = 3 - self.player_id
        self.opponent = wimblepong.SimpleAi(self.env, self.opponent_id)
        self.env.set_names("Amr", "AI")
        self.rewards = []
        self.diff = np.zeros((84, 84))
        #self.obs_2_max = np.zeros((2, 84, 84))
        self.obs_2 = np.zeros((2, 84, 84))
        self.obs_4 = np.zeros((4, 84, 84))

    def step(self, action1):
        reward = 0.
        done = False
        #for i in range(2):
        action2 = self.opponent.get_action()
        (ob1, ob2), (rw1, rw2), done, info = self.env.step((action1, action2))
        obs = self._process_obs(ob1)
        self.rewards.append(reward)
        if done:
            episode_info = {"reward": rw1, "length": len(self.rewards)}
            self.reset()
        else:
            reward = 0
            episode_info = None
            self.obs_2 = np.roll(self.obs_2, shift=-1, axis=0)
            self.obs_2[-1] = obs
            self.diff = self.obs_2[-1] - self.obs_2[-2]
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
            self.obs_4[-1] = self.diff

        #self.env.render()
        return self.obs_4, rw1, done, episode_info

    def reset(self):
        ob1, ob2 = self.env.reset()
        obs = self._process_obs(ob1)
        self.obs_2 = np.zeros((2, 84, 84))
        self.obs_2[-1] = obs
        self.obs_4 = np.zeros((4, 84, 84))
        self.obs_4[-1] = self.obs_2[-1] - self.obs_2[-2]
        self.rewards = []
        return self.obs_4


    @staticmethod
    def _process_obs(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs

def worker_process(remote: multiprocessing.connection.Connection, seed: int):
    game = Pong(seed)
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class Worker:
    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed))
        self.process.start()
