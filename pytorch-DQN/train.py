import gym
from model import Agent
from utils import plotLearning
import numpy as np
import torch as T

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
            epsilon_end=0.01, input_dims=[8], lr = 0.003)
    scores, eps_history = [], []
    n_games = 500
    max_avg_score = float('-inf')

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            #print(action)
            observation_, reward, done, info = env.step(action)
            score += reward 
            agent.store_transition(observation, action, reward, 
                                   observation_, done)
            agent.learn()
            observation = observation_
        

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(f"Episode: {i}; Score: {score}; Avg_score: {avg_score}; Epsilon: {agent.epsilon}")
        if avg_score > max_avg_score: 
            T.save(agent, "./saved_model/best.pt")
            print("model saved")
            max_avg_score = avg_score
    
    x = [i+1 for i in range(n_games)]
    filename = "lunar_lander_2022.png"
    plotLearning(x, scores, eps_history, filename)