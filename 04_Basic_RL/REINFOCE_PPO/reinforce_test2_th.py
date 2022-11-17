# -*- coding: utf-8 -*-

import pong_utils

import gym
import time

import matplotlib
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)
class CNNPolicy(nn.Module):

  def __init__(self, size, nr_ch, seed=None, drop=None, device=None):
    """Initialize parameters and build model.
    Params
    ======
        size : height (width) of the input
        nr_ch (int): number of channels
        seed (int): Random seed
 
        implements automated conv output calc



    """
    super(CNNPolicy, self).__init__()
    if seed:
      self.seed = torch.manual_seed(seed)
    
    self.conv_blocks = [
                {
                  "channels": 8,
                  "kernel": 5,
                  "stride" : 2
                },
                {
                  "channels": 16,
                  "kernel": 5,
                  "stride" : 2
                },
                {
                  "channels": 32,
                  "kernel": 3,
                  "stride" : 2
                }
              ]
    self.dense_blocks = [256, 1]
    self.input_size = size
    self.size = size
    print("Initializing policy with the following architecture:")
    if len(self.conv_blocks) > 0:      
      self.convs = nn.ModuleList()
      prev_conv = nr_ch # start with initial number of channels
      bias = False
      for _conv in self.conv_blocks:
        _channels = _conv['channels']
        _kernel = _conv['kernel']
        _stride = _conv['stride']
        _pad = 0 if 'padding' not in  _conv.keys() else _conv['padding']
        self.convs.append(nn.Conv2d(prev_conv, _channels, 
                                    kernel_size=_kernel, 
                                    stride=_stride,
                                    padding=_pad,
                                    bias=bias))
        bias = True
        self.convs.append(nn.ReLU())
        prev_conv = _channels
        size  = (size - _kernel + _stride) // _stride
        conv_out_size = (size ** 2) * _channels
        if "CONV" in str(self.convs[-2]).upper():
          print(" {} output: {}={}x{}x{}".format(
              self.convs[-2], conv_out_size, size, size, _channels))
        else:
          print(" {}".format(self.convs[-2]))
        print(" {}".format(self.convs[-1]))

    
    self.conv_out_size = conv_out_size
    
    if len(self.dense_blocks) > 0:
      self.denses = nn.ModuleList()      
      prev_dense = self.conv_out_size
      for nrunits in self.dense_blocks[:-1]:
        self.denses.append(nn.Linear(in_features=prev_dense, 
                                     out_features=nrunits))
        self.denses.append(nn.ReLU())
        if drop:
          self.denses.append(nn.Dropout(p=0.5))
        prev_dense = nrunits
      self.denses.append(nn.Linear(in_features=prev_dense, 
                                   out_features=self.dense_blocks[-1]))
      self.denses.append(nn.Sigmoid())
      
      for _d in self.denses:
        print(" {}".format(_d))
      
    if device:
      self.cuda(device)
    return
   

  def forward(self, x):
    if len(self.conv_blocks) > 0:
      for _layer in self.convs:
        x = _layer(x)
      x = x.view(x.size(0), -1) # x = x.view(-1, self.conv_out_size)
    if len(self.dense_blocks) > 0:      
      for _layer in self.denses:
        x = _layer(x)        
              
    # final output
    # 'RIGHTFIRE' will be the sigmoid
    # 'LEFTFIRE' will be 1-sigmoid
    return x

def disc_rewards_1d(rewards, discount):
  disc_r = np.zeros(len(rewards))
  G = 0
  for i in reversed(range(len(rewards))):
    G = G * discount + rewards[i]
    disc_r[i] = G
  return disc_r

def disc_rewards_2d(rewards, discount):
  rewards = np.array(rewards)
  disc_r = np.zeros(rewards.shape)
  G = np.zeros(rewards.shape[1])
  for i in reversed(range(rewards.shape[0])):
    G = G * discount + rewards[i]
    disc_r[i] = G
  return disc_r

def bad(rewards, discount):
  discount = (discount**np.arange(len(rewards)))
  rewards = np.asarray(rewards)*discount  
  rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
  return rewards_future
  
  
def disc_rewards(rewards, discount):
  discount = (discount**np.arange(len(rewards))).reshape((-1,1))
  rewards = np.asarray(rewards)*discount
  
  # convert rewards to future rewards
  rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
  rewards_future = rewards_future/discount # adjust to corect formula
  
  # now we normaliza between workers !
  mean = rewards_future.mean(axis=1).reshape((-1,1))
  std = rewards_future.std(axis=1).reshape((-1,1)) + 1e-10
  normed_rewards = (rewards_future - mean) / std
  return normed_rewards  



def surrogate(policy, old_probs, states, actions, rewards,
              discount = 0.995, beta=0.01, epsilon=0.1, use_ppo_clip=False):

    norm_rewards = disc_rewards(rewards, discount)
    
    t_actions = torch.tensor(actions, dtype=torch.int8, device=device)
    t_old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)    
    t_norm_rewards = torch.tensor(norm_rewards, dtype=torch.float, device=device)
    
    
    
    # convert states to policy (or probability)
    t_new_probs = pong_utils.states_to_prob(policy, states)
    t_new_probs = torch.where(t_actions == pong_utils.RIGHT, t_new_probs, 1.0-t_new_probs)
    
    """
     now we can either calc log(t_new_probs) or 
     take directly t_new_probs/t_old_probs (old_probs is same same and fixed)
    """
    t_rap = t_new_probs / t_old_probs # only t_new_probs is diferentiable
    
    
    if use_ppo_clip:
      t_rap_clip = torch.clamp(t_rap, 1 - epsilon, 1 + epsilon)
      
      t_main_loss = torch.min(t_norm_rewards * t_rap, 
                              t_norm_rewards * t_rap_clip)
    else:
      t_main_loss = t_norm_rewards * t_rap

    # include a regularization term
    # this steers new_policy towards 0.5
    # which prevents policy to become exactly 0 or 1
    # this helps with exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(t_new_probs*torch.log(t_old_probs+1.e-10)+ \
        (1.0-t_new_probs)*torch.log(1.0-t_old_probs+1.e-10))

    return torch.mean(t_main_loss + beta*entropy)


if __name__ == '__main__':
  np.set_printoptions(precision=3)
  np.set_printoptions(linewidth=130)  
  # PongDeterministic does not contain random frameskip
  # so is faster to train than the vanilla Pong-v4 environment
  env = gym.make('PongDeterministic-v4')
  print("List of available actions: ", env.unwrapped.get_action_meanings())
  device = pong_utils.device
  print("using device: ",device)
  
  # show what a preprocessed image looks like
  env.reset()
  _, _, _, _ = env.step(0)
  # get a frame after 20 steps
  for _ in range(20):
      frame, _, _, _ = env.step(1)
  
  plt.subplot(1,2,1)
  plt.imshow(frame)
  plt.title('original image')
  
  plt.subplot(1,2,2)
  plt.title('preprocessed image')
  
  # 80 x 80 black and white image
  plt.imshow(pong_utils.preprocess_single(frame), cmap='Greys')
  plt.show()
  
  policy = CNNPolicy(80, 2, device=device)
  optimizer = optim.Adam(policy.parameters(), lr=1e-4)  
  
  if False:
    
    print("Playing random policy...")
    pong_utils.play(env, policy, time=100) 
    print("Done playing random policy.")
    
    print("Creating env with multiple workers...")
    envs = pong_utils.parallelEnv('PongDeterministic-v4', n=4, seed=12345)
    print("Collecting samples for each worker...")
    prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=100)  
    print("Done collecting samples.")
  
    Lsur= surrogate(policy, prob, state, action, reward)
    print("Surogate result: {}".format(Lsur))
  
  if True:
        
    
    from parallelEnv import parallelEnv
    # WARNING: running through all 800 episodes will take 30-45 minutes
    
    # training loop max iterations
    #episode = 500
    episode = 800
    nr_workers = 8
    
    ## widget bar to display progress
    #!pip install progressbar
    #import progressbar as pb
    #widget = ['training loop: ', pb.Percentage(), ' ', 
    #          pb.Bar(), ' ', pb.ETA() ]
    #timer = pb.ProgressBar(widgets=widget, maxval=episode).start()
    
    # initialize environment
    envs = parallelEnv('PongDeterministic-v4', n=nr_workers, seed=1234)
    
    discount_rate = .99
    beta = .01
    tmax = 320
    epsilon = 0.1
    use_ppo = False
    SGD_PPO_epochs = 4
    title = "PPO_PG" if use_ppo else "REINFORCE"
    
    title += "_{}_workers".format(nr_workers)
    
    print("Starting {} for {} episodes".format(title, episode))
    
    # keep track of progress
    mean_rewards = []
    timings = []
    
    for e in range(episode):
        
        t0 = time.time()
    
        # collect trajectories
        old_probs, states, actions, rewards = \
            pong_utils.collect_trajectories(envs, policy, tmax=tmax, verbose=2)
            
        total_rewards = np.sum(rewards, axis=0)


        if use_ppo:   
          for _ in range(SGD_PPO_epochs):
            L = -surrogate(policy, old_probs, states, actions, 
                           rewards, beta=beta, epsilon=epsilon,
                           use_ppo_clip=True)
            
            #L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards,
            #                                  epsilon=epsilon, beta=beta)
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L
        else:
          L = -surrogate(policy, old_probs, states, actions, 
                         rewards, beta=beta, )
          
          #L = -pong_utils.surrogate(policy, old_probs, states, actions, rewards, beta=beta)
          optimizer.zero_grad()
          L.backward()
          optimizer.step()
          del L
          
            
        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.995
        
        epsilon *=.999
        
        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))
        
        # display some progress every 20 iterations
        t1 = time.time()
        timings.append(t1-t0)
        estimated_time = np.mean(timings) * episode
        left_time = estimated_time - np.sum(timings)
        if (e+1)%5 ==0 :            
            print("\nEpisode: {:>3}, score: {:>6.1f},  time: {:>4.1f} min / {:4.1f} min".format(
                e+1,np.mean(total_rewards), 
                left_time/60, estimated_time/60), flush=True)
            print(total_rewards, flush=True)
            
        # update progress widget bar
        #timer.update(e+1)
        
        
        
    #timer.finish()
        
    # play game after training!
    pong_utils.play(env, policy, time=2000) 
    
    plt.plot(mean_rewards)
    plt.title(title)
    plt.savefig(title+".png")
    plt.xlabel("Paralel Episodes")
    plt.ylabel("Scores")
    plt.show()
    
    
    # save your policy!
    torch.save(policy, 'REINFORCE.policy')
    
    # load your policy if needed
    # policy = torch.load('REINFORCE.policy')
    
    # try and test out the solution!
    # policy = torch.load('PPO_solution.policy')