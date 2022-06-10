## There is a problem in VecFrameStack ; when using VecFrameStack.learn, I got this error numpy.random._generator.Generator' object has no attribute 'randint' ; noops = self.unwrapped.np_random.randint(1, self.noop_max + 1); anyone can help me? thanks
from tabnanny import verbose
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import random



class AtariEnv():
    def __init__(self, environment_name):
        self.env_name = environment_name
        # self.dummy_env = VecFrameStack([lambda: self.env])
    
    def test_model_1(self):
        env = gym.make(self.env_name , render_mode="human")
        print("test_model_1")
        print(env.action_space)
        #print(env.observation_space)
        for episode in range(3):
            obs = env.reset()
            done = False
            score = 0
            
            while not done:
                #self.env.render() -> donesn't need this because we have used render_mode = human
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                score += reward
            print(f"Episode: {episode} Score:{score} ")
    def multiple_env_render(self, num_envs , mode ="render"):
        env = make_atari_env(self.env_name, n_envs=num_envs, seed=0)
        dummy_env = VecFrameStack(env, n_stack=num_envs)
        #print(dummy_env.action_space.sample())
        if mode == "render" :
            # while True:
            #     dummy_env.render()
            for episode in range(400):
                obs = dummy_env.reset()
                done = False
                score = 0
                while not done:
                    dummy_env.render()
                    action = dummy_env.action_space.sample()
                    obs, reward, done, info = dummy_env.step(action)
                    done = done.any()
                    score += reward
                print(f"Episode: {episode} Score:{score} ")
                
        elif mode == "learn":
            log_path = os.path.join("Training" , "Logs")
            model = A2C("CnnPolicy" , dummy_env, verbose=1, tensorboard_log=log_path)
            model.learn(total_timesteps = 25000)


if __name__ == "__main__":
    atari_env = AtariEnv("Breakout-v0")
    #atari_env.test_model_1()
    atari_env.multiple_env_render(num_envs = 4, mode = "render")