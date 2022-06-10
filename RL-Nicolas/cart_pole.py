import os
from pathlib import Path
import gym
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


environment_name = 'CartPole-v0'
EPISODES = 5
log_path = os.path.join("Training" , "Logs")
PPO_path = os.path.join("Training" , "Saved_Models" , "Model_Cartpole")

def create_dir():
	Path(log_path).mkdir(parents=True, exist_ok=True)
	Path(PPO_path).mkdir(parents=True, exist_ok=True)

class CartPoleEnv():
	def __init__(self, environment_name):
		self.env = gym.make(environment_name)
		self.dummy_env = DummyVecEnv([lambda: self.env])
		self.EPISODES = 5
		
	def train_model(self, save_dir , policy = "PPO"):
		if policy == "PPO":
			model = PPO("MlpPolicy" , self.dummy_env, verbose =1, tensorboard_log=log_path)
		elif policy == "PPO2":
			net_arch = [dict(pi=[128,128,128,128], vf = [128,128,128,128])]
			model = PPO("MlpPolicy" , self.dummy_env, verbose =1, tensorboard_log=log_path, policy_kwargs={'net_arch' : net_arch})
		elif policy == "DQN":
			model = PPO("MlpPolicy" , self.dummy_env, verbose =1, tensorboard_log=log_path)
		model.learn(total_timesteps = 20000)
		print(f"train {policy} finished")
		model.save(save_dir)

	def load_model(self, load_dir):
		self.loaded_model = PPO.load(load_dir + str(".zip"))
		#evaluate_policy(self.loaded_model, self.dummy_env, n_eval_episodes=3, render = True)
	
	def test_model(self):
		print("test model")
		for episode in range(self.EPISODES):
			obs = self.dummy_env.reset()
			done = False
			score = 0
			
			while not done:
				self.dummy_env.render()
				action, _ = self.loaded_model.predict(obs)
				obs, reward, done, info = self.dummy_env.step(action)
				score += reward
			print(f"Episode: {episode} Score: {score}")
		



if __name__ == "__main__":
	create_dir()
	myRL = CartPoleEnv(environment_name)
	myRL.train_model(PPO_path, policy ="DQN")
	myRL.load_model(PPO_path)
	myRL.test_model()