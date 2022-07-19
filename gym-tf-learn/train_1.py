import gym
import numpy as np

def load_demonstration():
    sarsa_pairs = np.load("./data/sarsa_pairs.npy", allow_pickle=True)
    print(sarsa_pairs)

def main():

    env = gym.make("MountainCar-v0")
    for episode in range(100):
        env.reset()

        while True:
            #env.render()
            action = env.action_space.sample()
            #print(action)
            obs, reward, done, info = env.step(action)
            #print(obs)

if __name__=="__main__":
    #main()
    