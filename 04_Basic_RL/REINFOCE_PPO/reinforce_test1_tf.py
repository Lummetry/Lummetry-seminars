# -*- coding: utf-8 -*-

import gym
# suppress warnings (please remove if gives error)
gym.logger.set_level(40) 
import numpy as np
from collections import deque, OrderedDict
from time import time
import itertools
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

from env_player import EnvPlayer

"""
torch:
  Simple rewards avg steps:    1353.0
  Norm disc rewards avg steps: 352.0
  
tensorflow:
  Training with normed_disc_rewards=True
  Episode 100     Average Score: 92.54    Average time/ep: 0.17s
  Episode 200     Average Score: 155.79   Average time/ep: 0.27s
  Episode 300     Average Score: 162.27   Average time/ep: 0.28s
  Episode 400     Average Score: 174.52   Average time/ep: 0.30s
  Environment solved in 422 episodes!     Average Score: 195.63
  Agent init
  Training with normed_disc_rewards=False
  Episode 100     Average Score: 21.02    Average time/ep: 0.05s
  Episode 200     Average Score: 46.70    Average time/ep: 0.08s
  Episode 300     Average Score: 83.94    Average time/ep: 0.15s
  Episode 400     Average Score: 181.34   Average time/ep: 0.32s
  Episode 500     Average Score: 114.80   Average time/ep: 0.20s
  Episode 600     Average Score: 44.91    Average time/ep: 0.08s
  Episode 700     Average Score: 53.18    Average time/ep: 0.10s
  Episode 800     Average Score: 80.72    Average time/ep: 0.15s
  Episode 900     Average Score: 77.35    Average time/ep: 0.14s
  Episode 1000    Average Score: 181.07   Average time/ep: 0.32s
  Environment solved in 1024 episodes!    Average Score: 195.06  

"""

class TFAgent:
  def __init__(self, s_size=4, h_size=16, a_size=2, name='test',
               k_train_func=True, set_learning_phase=True,
               bn=True):
    
    self.name = name
    if type(s_size) is not int:
      self.state_size = s_size
    else:
      self.state_size = (s_size,)
    self.h_size = h_size
    self.action_size = a_size
    self.k_train_func = k_train_func
    self.bn = bn
    self.set_learning_phase = set_learning_phase
    self.init_model()
    print("  Agent init: k_func={} set_learn_ph={}".format(
        self.k_train_func, 
        self.set_learning_phase if self.k_train_func else "IGNORED"))
    self.model.summary()
    return

  def init_model(self):
    tf_inp_state = tf.keras.layers.Input(self.state_size, name='state')
    tf_x = tf.keras.layers.Dense(self.h_size, 
                                 activation=None, 
                                 name='linear1')(tf_inp_state)
    if self.bn:
      tf_x = tf.keras.layers.BatchNormalization(name='bn1')(tf_x)
    tf_x = tf.keras.layers.Activation('selu', name='act1')(tf_x)
    tf_probas = tf.keras.layers.Dense(self.action_size, 
                                      activation='softmax', 
                                      name='out')(tf_x)
    self.model = tf.keras.models.Model(inputs=tf_inp_state, outputs=tf_probas)
    opt = tf.keras.optimizers.Adam(lr=1e-2)
    self.model.compile(loss='categorical_crossentropy', optimizer=opt)
    if self.k_train_func:
      self.train_func = self._get_train_func1()
    else:
      self.train_func = self._get_train_func2()
    return
  
  def _get_train_func1(self):
    tf_inp_state = tf.keras.layers.Input(self.state_size, name='state_trainer')
    tf_disc_rew = tf.keras.layers.Input((1,), name='rewards')
    tf_inp_action = tf.keras.layers.Input((1,), dtype=tf.int32, name='actions')
    tf_probas = self.model(tf_inp_state)
    tf_lp = K.learning_phase()
    tf_lp_check = tf.identity(tf_lp)
    tf_act_oh = tf.keras.layers.Lambda(
        function=lambda x: K.one_hot(x, 
                                     num_classes=self.action_size),
                                     name='onehotter')(tf_inp_action)
    tf_act_oh = tf.keras.layers.Lambda(function=lambda x: K.squeeze(x, axis=1),
                                       name='squeezer')(tf_act_oh)
    tf_probas_clip = K.clip(tf_probas, K.epsilon(), 1 - K.epsilon())
    tf_log_proba = K.log(tf_probas_clip)
    tf_act_probas = tf_act_oh * tf_log_proba 
    tf_loss_act = - (tf_act_probas * tf_disc_rew)
    tf_loss = K.sum(tf_loss_act)
    opt = tf.keras.optimizers.Adam(lr=1e-2)
    tf_updates = opt.get_updates(params=self.model.trainable_variables,
                                 loss=tf_loss)
    _inputs = [tf_inp_state, 
               tf_inp_action, 
               tf_disc_rew,]
    _outputs = [tf_loss, tf_lp_check]
    if self.set_learning_phase:
      _inputs += [K.learning_phase()]
    train_func = K.function(inputs=_inputs,
                            outputs=_outputs, #, K.learning_phase()],
                            updates=tf_updates)
    return train_func
  
  def _get_train_func2(self):
    return self._model_train_func
  
  def _model_train_func(self, inputs):
    """
    so, here we do a trick:
      we already compiled the Model with cross-entropy so basically
      now we are going to scale the actions by the discounted rewards
      for each observation and feed this as targets
    """
    np_states, np_actions, np_rewards = inputs
    np_act_oh = tf.keras.utils.to_categorical(
        np_actions, num_classes=self.action_size)
    np_act_adv = np_act_oh * np_rewards
    if np_act_adv.shape[-1] != self.action_size:
      raise ValueError("Problem!\n states: {}\n{}act_oh: {}\n act_adv: {}".format(
          np_states, np_act_oh,np_act_adv))
    loss = self.model.train_on_batch(np_states, np_act_adv)
    return loss
    
    
  def train(self,states, actions, disc_rewards):
    states = np.array(states)    
    actions = np.array(actions).reshape(-1,1)
    disc_rewards = np.array(disc_rewards).reshape(-1,1)
    if self.set_learning_phase:
      _learning_phase = [1]
    else:
      _learning_phase = []
    _inputs = [states, actions, disc_rewards] + _learning_phase
    _res = self.train_func(_inputs)
    if type(_res) in [tuple, list]:
      loss = _res[0]
    else:
      loss = _res
    return loss
  
  def act(self, state, return_proba=False, sampled=True):
    if len(state.shape) != 2:
      state = state.reshape(1,-1)
    probs = self.model.predict(state).squeeze()
    if not sampled:
      act = np.argmax(probs, axis=-1)
    else:
      act = np.random.choice(np.arange(self.action_size), p=probs)
    proba = probs[act]
    _ret = (act, proba) if return_proba else act 
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
  
def disc_rewards(rewards, discount):
  """
  this is the optimized version 
  """
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
  
  

def grid_dict_to_values(params_grid):
    """
    method to convert a grid search dict into a list of all combinations
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

def grid_dict_to_results(params_grid, value_keys):
  if type(value_keys) is str:
    value_keys = [value_keys]
  dict_results = OrderedDict()
  for vkey in value_keys:
    dict_results[vkey] = []
  for k in params_grid:
    dict_results[k] = []
  return  dict_results

def add_results(dict_results, 
                dict_grid_pos, 
                values, 
                value_keys):
  value_keys = [value_keys] if type(value_keys) == str else value_keys
  values = [values] if type(values) in [int, float] else values
  for k in dict_grid_pos:
    dict_results[k].append(dict_grid_pos[k])
  for i,k in enumerate(value_keys):
    dict_results[k].append(values[i])
  return dict_results

def grid_pos_to_params(grid_data, params):
  """
  converts a grid search combination to a dict for callbacks that expect kwargs
  """
  func_kwargs = {}
  for j,k in enumerate(params):
    func_kwargs[k] = grid_data[j]  
  return func_kwargs

def grid_dict_to_generator(dict_grid):
  _combs, _params = grid_dict_to_values(dict_grid)
  for _c in _combs:
    dict_pos = grid_pos_to_params(_c, _params)
    yield dict_pos
  

def reinforce(env, agent, n_episodes=2000, max_t=1000, 
              gamma=1.0, print_every=100, use_disc_rewards=True):
  solved = False
  scores_deque = deque(maxlen=100)
  scores = []
  timings = []
  ep_losses = []
  print("   Training with normed_disc_rewards={}".format(use_disc_rewards))
  for i_episode in range(1, n_episodes+1):
    t_0 = time()
    rewards = []
    states = []
    actions = []
    state = env.reset()
    for t in range(max_t):
      action= agent.act(state)
      next_state, reward, done, _ = env.step(action)
      rewards.append(reward)
      states.append(state)
      actions.append(action)
      if done:
        break 
      state = next_state

    scores_deque.append(sum(rewards))
    scores.append(sum(rewards))
    
    if not use_disc_rewards:
      discounts = [gamma**i for i in range(len(rewards)+1)]
      R = sum([a*b for a,b in zip(discounts, rewards)])
      disc_rewards = [R] * len(actions)
    else:
      disc_rewards = discounted_rewards(rewards, gamma)

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
    
    ep_loss = agent.train(states, actions, disc_rewards)
    ep_losses.append(ep_loss)
    
    t_1 = time()
    timings.append(t_1-t_0)
    
    if i_episode % print_every == 0:
      print('    Episode {}\tAverage Score: {:.2f}\tAverage time/ep: {:.2f}s'.format(
            i_episode, np.mean(scores_deque), np.mean(timings)))
      timings = []

    if np.mean(scores_deque)>=195.0:
      print('  Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
      solved = True
      break
      
  return solved, scores, i_episode


if __name__ == '__main__':
  
  e = gym.make('CartPole-v0')
  
  play_random = False
  
  if play_random:
    p1 = EnvPlayer(env=e)
    p1.play()
  
  e.seed(0)
  print('observation space:', e.observation_space)
  print('action space:', e.action_space)
  
  grid = {
      "Itr" : [1, 2, 3],
      "norm_d_rew" : [True, False],
      "k_func" : ['NO', "YES_LrnPhs1", "YES_NoLrnPhs"],
      'bn' : [True, False],
      }
  
  gen = grid_dict_to_generator(grid)
  
  results = grid_dict_to_results(grid, value_keys=['solved','n_ep'])
  
  best_agent = None
  best_steps = np.inf
  show_best_agent = False

  for ii,iter_params in enumerate(gen):
    print("\n\nIteration {:>2}".format(ii))
    NormDiscRewards = iter_params['norm_d_rew']
    _k_train_func = iter_params['k_func']
    bn = iter_params['bn']
    set_learning_phase = False
    if "YES" in _k_train_func:
      k_train_func = True
      set_learning_phase = True if "LP1" in _k_train_func else False
    else:
      k_train_func = False
    
    a = TFAgent(k_train_func=k_train_func,
                set_learning_phase=set_learning_phase,
                bn=bn)
    
    solved, scores, n_ep = reinforce(env=e, agent=a, use_disc_rewards=NormDiscRewards)
    
    if solved:
      if n_ep < best_steps:
        best_steps = n_ep
        best_agent = a
    results = add_results(results, 
                          iter_params, 
                          values=[solved,n_ep], 
                          value_keys=['solved','n_ep'])

  df_results = pd.DataFrame(results).sort_values('n_ep')
  print(df_results)      
  
  if show_best_agent:
    p2 = EnvPlayer(env=e, agent=best_agent)
    p2.play(cont=False, save_gif='cart_reinforce.gif')
