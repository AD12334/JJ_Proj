
import random

import gymnasium as gym
import ale_py
import pygame


gym.register_envs(ale_py)

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
observation, info = env.reset()

pygame.init()

action_meanings = env.unwrapped.get_action_meanings()
print("Available actions:", action_meanings)
print("Observation shape:", observation.shape)

def baseline(num_episodes):
    for i in range(1,num_episodes+1):
        state = env.reset()
        terminated = False
        score = 0   
        while not terminated:
            action = random.choice([0,1,2,3,4,5])
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
        print(f"Episode {i}, score {score}")

    env.close()
    pygame.quit()
    
if __name__ == "__main__":
    baseline(5)
