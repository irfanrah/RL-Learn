import gym

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
    main()