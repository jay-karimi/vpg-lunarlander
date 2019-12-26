import os
from time import time

import gym
from gym import wrappers

from vanilla_policy_gradient import VanillaPolicyGradient

NUM_EPISODES = 5000
NUM_EVAL_EPISODES = 100
NUM_VIDEOS = 10


vpg = VanillaPolicyGradient(lr=0.001, reward_decay=0.99)
max_reward = -float('inf')

reward_history = []
max_reward_history = []
eval_reward_history = []
steps_history = []
eval_steps_history = []

time = time()

# collect initial videos (pre-training)
# for some reason, it only generates 3 videos instead of 10
env = gym.make('LunarLander-v2')
env.seed(42)
env = wrappers.Monitor(env, 'videos/' + str(time) + '_start/')
for episode in range(NUM_VIDEOS):
    observation = env.reset()

    while True:
        action = vpg.select_action(observation)
        observation, _, done, _ = env.step(action)

        if done:
            break
env.close()

# training loop
env = gym.make('LunarLander-v2')
env.seed(42)
for episode in range(NUM_EPISODES):
    observations, actions, rewards = [], [], []
    observation = env.reset()

    while True:
        observations.append(observation)

        action = vpg.select_action(observation)
        observation, reward, done, _ = env.step(action)

        max_reward = max(max_reward, sum(rewards))

        actions.append(action)
        rewards.append(reward)

        if done or sum(rewards) < -250:
            vpg.update_policy(observations, actions, rewards)
            reward_history.append(sum(rewards))
            steps_history.append(len(rewards))
            max_reward_history.append(max_reward)
            print(
                '#####################################################\n'
                f'Episode: {episode}\n'
                f'Reward: {sum(rewards)}\n'
                f'Steps: {len(rewards)}\n'
                f'Max Reward: {max_reward}\n',
            )
            break

    # training eval loop
    if episode % 250 == 0:
        total_reward = 0
        total_steps = 0

        for e in range(NUM_EVAL_EPISODES):
            observation = env.reset()

            while True:
                action = vpg.select_action(observation)
                _, reward, done, _ = env.step(action)
                total_reward += reward
                total_steps += 1

                if done:
                    break

    eval_reward_history.append(total_reward / NUM_EVAL_EPISODES)
    eval_steps_history.append(total_steps / NUM_EVAL_EPISODES)

env.close()

# collect videos post training
# for some reason, it only generates 3 videos instead of 10
env = gym.make('LunarLander-v2')
env.seed(42)
env = wrappers.Monitor(env, 'videos/' + str(time) + '_end/')
for episode in range(NUM_VIDEOS):
    observation = env.reset()

    while True:
        action = vpg.select_action(observation)
        observation, _, done, _ = env.step(action)

        if done:
            break

# save data
if not os.path.isdir('data/'):
    os.mkdir('data/')

with open(f'data/{time}_rewards_train.txt', 'w') as f:
    for el in reward_history:
        f.write(f'{el}\n')

with open(f'data/{time}_steps_train.txt', 'w') as f:
    for el in steps_history:
        f.write(f'{el}\n')

with open(f'data/{time}_rewards_test.txt', 'w') as f:
    for el in eval_reward_history:
        f.write(f'{el}\n')

with open(f'data/{time}_steps_test.txt', 'w') as f:
    for el in eval_steps_history:
        f.write(f'{el}\n')

with open(f'data/{time}_max_reward_train.txt', 'w') as f:
    for el in max_reward_history:
        f.write(f'{el}\n')



