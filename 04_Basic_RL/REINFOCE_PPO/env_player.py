# -*- coding: utf-8 -*-

import cv2

from time import time, sleep
from datetime import datetime as dt

import numpy as np

import imageio

__VER__ = '0.9.1'


class EnvPlayer:
  def __init__(self, env=None, agent=None, save_gif='test.gif', frames_only=False):
    self.frames_only = frames_only    
    self.__version__ = __VER__
    self.env = env
    if not self.frames_only:
      if self.env is None:
        raise ValueError("Uknown environment!")
      else:
        if 'rgb_array' not in self.env.metadata['render.modes']:
          raise ValueError("Env {} does not support rgb rendering!")
    self.agent = agent
    self.id = dt.now().strftime("%Y%m%d_%H%M%S")
    self.save_gif = save_gif
    if self.save_gif:
      self.save_gif = self.id + "_" + self.save_gif
    self.done = True
    if self.agent:
      if hasattr(self.agent, "name"):
        self.agent_name = self.agent.name
      else:
        self.agent_name = 'X'
    else:
      self.agent_name = 'random'
    self.win_name = 'Gym_Env_Player_Agent_{}'.format(self.agent_name)
    self.video_started = False
    return
      
  def _get_next_frame(self, act=None):
    if self.done:
      self.state = self.env.reset()    
      self.done = False
    else:
      if act is None:
        if self.agent:
          act = self.agent.act(self.state)
        else:
          act = self.env.action_space.sample()
      if (type(act) is np.ndarray) and (act.shape[0] == 1) and len(act.shape)>1:
        act = act.squeeze()
      obs, r, done, info = self.env.step(act)
      self.done = done    
      self.state = obs
      self.reward =r
      self.last_action = act
      
    np_frm_rgb = self.env.render(mode='rgb_array')   
    np_frm = cv2.cvtColor(np_frm_rgb, cv2.COLOR_RGB2BGR)
    return np_frm
  
  def _start_video(self):
    self.buff_frames = []
    cv2.namedWindow(self.win_name,cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(self.win_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(self.win_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
    cv2.moveWindow(self.win_name, 1, 1) 
    self.video_started = True
    return
      
  def _end_video(self):
    cv2.destroyAllWindows()
    if self.env:
      self.env.close()
    self.video_started = False
    if self.save_gif:
      imageio.mimsave(self.save_gif, self.buff_frames)
      print("Animated gif saved in {}".format(self.save_gif))
    return
  
  def _show_message(self, np_img, _text):
    h , w = np_img.shape[:-1]
    font                   = cv2.FONT_HERSHEY_TRIPLEX
    bottomLeftCornerOfText = (1, 30)
    fontScale              = 0.6
    fontColor              = (255,0,0)
    thickness  = 2
    cv2.putText(
        img=np_img, 
        text=_text, 
        org=bottomLeftCornerOfText, 
        fontFace=font, 
        fontScale=fontScale,
        thickness=thickness,
        color=fontColor)
    return np_img
  
  def _quit_requested(self):
    key = cv2.waitKey(1) & 0xFF
    if (key == ord('q')) or (key == ord('Q')):
      return True
    else:
      return False
  
  def play(self, cont=True, sleep_time=0.05, save_gif=None):
    if self.save_gif is None:
      self.save_gif = save_gif
    self._start_video()
    while True:    
      out_frame = self._get_next_frame()
      self._play_frame(out_frame)
      if self._quit_requested(): break
      if sleep_time:
        sleep(sleep_time)
      if self._quit_requested(): break
      if self.done:
        out_frame = self._show_message(out_frame, "EPISODE DONE")
        for i in range(20):
          self._play_frame(out_frame, convert_to_bgr=True)
          if self._quit_requested(): break
          if sleep_time:
            sleep(sleep_time)
        if not cont:
          break
    self._end_video()
    return

  def play_action(self, act, sleep_time=0.05):
    if not self.video_started:
      self._start_video()
    out_frame = self._get_next_frame(act=act)
    self._play_frame(out_frame)
    if sleep_time:
      sleep(sleep_time)
    if self.done:
      out_frame = self._show_message(out_frame, "EPISODE DONE")
      for i in range(20):
        self._play_frame(out_frame, convert_to_bgr=True)
        if sleep_time:
          sleep(sleep_time)
    return
  
  def _play_frame(self, np_frame, convert_to_bgr=False):
    out_frame = np_frame.copy()
    if convert_to_bgr:
      out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(self.win_name,out_frame)
    self.buff_frames.append(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))
    return
    

  def play_frame(self, np_frame, done=False, sleep_time=0.01):
    if not self.video_started:
      self._start_video()
    self._play_frame(np_frame, convert_to_bgr=True)
    if self._quit_requested():
      pass
    if sleep_time:
      sleep(sleep_time)
    if done:
      np_frame = self._show_message(np_frame, "EPISODE DONE")
      for i in range(20):
        self._play_frame(np_frame, convert_to_bgr=True)
        if self._quit_requested():
          pass
        if sleep_time:
          sleep(sleep_time)
    return
    
  
  def close(self):
    self._end_video()
    return
    
