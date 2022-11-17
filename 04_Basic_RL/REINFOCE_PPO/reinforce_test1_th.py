# -*- coding: utf-8 -*-

import gym
gym.logger.set_level(40) # suppress warnings (please remove if gives error)
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from time import time
import itertools

import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from env_player import EnvPlayer

"""
Simple rewards avg steps:    1353.0
Norm disc rewards avg steps: 352.0

"""

class Agent(nn.Module):
  def __init__(self, device, s_size=4, h_size=16, a_size=2, name='test'):
    super(Agent, self).__init__()
    self.dev = device
    self.name = name
    self.fc1 = nn.Linear(s_size, h_size)
    self.fc2 = nn.Linear(h_size, a_size)
    if device.type == 'cuda':
      self.cuda(device)
    print("Agent init on device {}".format(self.dev))

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.softmax(x, dim=1)
  
  def act(self, state, return_proba=False):
    state = torch.from_numpy(state).float().unsqueeze(0).to(self.dev)
    probs = self.forward(state).cpu()
    m = Categorical(probs)
    action = m.sample()
    logp = m.log_prob(action)
    act = action.item()
    # return action and "error" according to the generated probs
    _ret = (act, logp) if return_proba else act #probs.max(1)[1].item
    return _ret 
  
  

def discounted_rewards(rewards, gamma, normalize=True):
  """
      Because we have a Markov process, the action at time-step tt can only affect 
      the future reward, so the past reward shouldn’t be contributing to the policy 
      gradient. So to properly assign credit to the action a_ta, we should ignore 
      the past reward. So a better policy gradient would simply have the future 
      reward as the coefficient .  
  """
  t_rewards = 0
  disc_rewards = np.zeros(len(rewards))
  for i in reversed(range(len(rewards))):
    t_rewards = rewards[i] + gamma * t_rewards
    disc_rewards[i] = t_rewards
  if normalize:
    disc_rewards -= disc_rewards.mean()
    disc_rewards /= disc_rewards.std()
  return disc_rewards
  

def grid_dict_to_values(params_grid):
    """
    method to convert a grid serach dict into a list of all combinations
    returns combinations and param names for each combination entry
    """
    params = []
    values = []
    for k in params_grid:
      params.append(k)
      assert type(params_grid[k]) is list, 'All grid-search params must be lists. Error: {}'.format(k)
      values.append(params_grid[k])
    combs = list(itertools.product(*values))
    return combs, params

def grid_pos_to_params(grid_data, params):
  """
  converts a grid search combination to a dict for callbacks that expect kwargs
  """
  func_kwargs = {}
  for j,k in enumerate(params):
    func_kwargs[k] = grid_data[j]  
  return func_kwargs

def reinforce(env, agent, n_episodes=2000, max_t=1000, 
              gamma=1.0, print_every=100, use_disc_rewards=True):
  solved = False
  optimizer = optim.Adam(agent.parameters(), lr=1e-2)
  scores_deque = deque(maxlen=100)
  scores = []
  timings = []
  print("Training with normed_disc_rewards={}".format(use_disc_rewards))
  for i_episode in range(1, n_episodes+1):
    t_0 = time()
    saved_log_probs = []
    rewards = []
    state = env.reset()
    for t in range(max_t):
      action, log_prob = agent.act(state, return_proba=True)
      saved_log_probs.append(log_prob)
      state, reward, done, _ = env.step(action)
      rewards.append(reward)
      if done:
        break 
    scores_deque.append(sum(rewards))
    scores.append(sum(rewards))
    
    if not use_disc_rewards:
      discounts = [gamma**i for i in range(len(rewards)+1)]
      R = sum([a*b for a,b in zip(discounts, rewards)])
      disc_rewards = [R] * len(saved_log_probs)
    else:
      disc_rewards = discounted_rewards(rewards, gamma)

    policy_loss = []
    """
        our goal is to optimize (max) sum(proba * disc_reward) for all steps
        example 1:
          gamma  = 1
          t = 0.5
          P(1 | state) = t  
          P(0 | state) = 1 - t
          action = 1 0 1
          reward = 0 0 1
          =>
          disc rewards = [0 + gamma * 1] [0 + gamma * 1] [1]
          grad = dlogP(1) * 1 + dlogP(0) * 1 + dlogP(0) * 1
          grad = 1 / P(1) * dP(1) * 1 + 1 / P(0) * dP(0) * 1+ 1 / P(0) * dP(0) * 1
          grad = (1/t) * 1 + (1/(1-t) * (-1)) * 1 + (1/(1-t) * (-1)) * 1
          
        example 2:
          actions: (0,1,0) rewards: (1,0,1)
          conclusions: 
            last two step-grads cancel each other and thus using total reward 
            will yield the same gradient results
          
    """
    for i,log_prob in enumerate(saved_log_probs):
      policy_loss.append(-log_prob * disc_rewards[i])
    policy_loss_batch = torch.cat(policy_loss)
    policy_loss = policy_loss_batch.sum()
    
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    t_1 = time()
    timings.append(t_1-t_0)
    
    if i_episode % print_every == 0:
      print('Episode {}\tAverage Score: {:.2f}\tAverage time/ep: {:.2f}s'.format(
            i_episode, np.mean(scores_deque), np.mean(timings)))
      timings = []

    if np.mean(scores_deque)>=195.0:
      print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
      solved = True
      break
      
  return solved, scores, i_episode


e = gym.make('CartPole-v0')

play_random = False

if play_random:
  p1 = EnvPlayer(env=e)
  p1.play()

e.seed(0)
print('observation space:', e.observation_space)
print('action space:', e.action_space)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

grid = {
    "NormDiscRewards" : [False,True]
    }

_combs, _params = grid_dict_to_values(grid)

results = []
best_agent = None
best_steps = np.inf

for grid_data in _combs:
  iter_params = grid_pos_to_params(grid_data, _params)
  NormDiscRewards = iter_params['NormDiscRewards']
  a = Agent(device=dev)
  
  solved, scores, n_ep = reinforce(env=e, agent=a, use_disc_rewards=NormDiscRewards)
  
  if solved:
    if n_ep < best_steps:
      best_steps = n_ep
      best_agent = a
    results.append((iter_params,n_ep))
    
results = sorted(results, key=lambda x:x[1])
for result in results:    
  print("Rrsult: {} avg nr of steps until completion for :  {}".format(
      result[1], result[0]))

p2 = EnvPlayer(env=e, agent=best_agent)
p2.play(cont=False, save_gif='cart_reinforce.gif')
