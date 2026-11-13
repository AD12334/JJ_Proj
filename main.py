
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

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            pygame.quit()
            exit()

    keys = pygame.key.get_pressed()

    action = 0

    if keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        action = action_meanings.index("LEFTFIRE")
    elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        action = action_meanings.index("RIGHTFIRE")
    elif keys[pygame.K_SPACE]:
        action = action_meanings.index("FIRE")
    elif keys[pygame.K_LEFT]:
        action = action_meanings.index("LEFT")
    elif keys[pygame.K_RIGHT]:
        action = action_meanings.index("RIGHT")

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
pygame.quit()
