import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.optim import Adam

from policy_network import PolicyNetwork


class VanillaPolicyGradient:
    def __init__(self, lr=0.01, reward_decay=0.95):
        self.policy_network = PolicyNetwork()
        self.optimizer = Adam(self.policy_network.parameters(), lr=lr)
        self.reward_decay = reward_decay

    def select_action(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        logits = self.policy_network(Variable(state, requires_grad=False))
        sampler = Categorical(logits=logits)
        return sampler.sample().item()

    def update_policy(self, observations, actions, rewards):
        rewards = torch.from_numpy(self.reward_to_go(rewards)).type(torch.FloatTensor)
        actions = torch.tensor(actions)
        observations = torch.from_numpy(np.asarray(observations)).type(torch.FloatTensor)

        log_probs = F.log_softmax(self.policy_network(observations), dim=1).gather(1, actions.view(-1, 1)).view(-1)
        loss = -(rewards * log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reward_to_go(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        cum_reward = 0
        for i in reversed(range(len(rewards))):
            cum_reward = cum_reward * self.reward_decay + rewards[i]
            discounted_rewards[i] = cum_reward

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards
