"""
Project:    rl-value-prediction-learning
File:       atati-dqn.py
Created by: louise
On:         01/05/18
At:         4:25 PM
"""
import argparse
import os
import random
import time
import gc

import gym
from gym import wrappers
import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F


class preprocessing():
    @staticmethod
    def to_grayscale(img):
        return np.mean(img, axis=2).astype(np.uint8)

    @staticmethod
    def downsample(img):
        return img[::2, ::2]

    @staticmethod
    def preprocess(img):
        return preprocessing.to_grayscale(preprocessing.downsample(img))

