import gym
import numpy as np
import torch as T

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 10
    model = T.load("./saved_model/best.pt")
    #model.eval()
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            env.render()
            action = model.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_
        print(f"Episode: {i}; Score: {score}; ")
