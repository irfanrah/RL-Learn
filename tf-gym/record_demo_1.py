import gym
import readchar
import numpy as np


def main():
    SHARD_SIZE = 2000

    env = gym.make("MountainCar-v0")
    env = gym.wrappers.Monitor(env, './data', force=True)
    _last_obs = env.reset()
    sarsa_pairs = []
    
    while True:
        char_input = [readchar.readkey()]
        if char_input[0] == 'a':
            print("left")
            action = 1
        if char_input[0] == 'd':
            print("right") 
            action = 2
        env.render()
        #action = env.action_space.sample()
        print(action)
        obs, reward, done, info = env.step(action)
        sarsa = (_last_obs, action)
        _last_obs = obs
        sarsa_pairs.append(sarsa)
        #print(obs)
        if done:
            print("HAHHAAHHA")
        # Save out recording data.
            with open('./data/sarsa_pairs.npy', 'wb') as f:
                np.save(f, np.array(sarsa_pairs))
            print("SAVED")

            print("Done")
    

if __name__=="__main__":
    main()