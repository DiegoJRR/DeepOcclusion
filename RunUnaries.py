#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use("nbagg")
import skimage
import skimage.filters
import skimage.morphology
import math
import numpy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2
import scipy
import scipy.io as sio
import re, os, glob, pickle, shutil
from shutil import *
import sys
import random
sys.path.append('../roi_pooling/theano-roi-pooling/')
sys.path.append('./POM')


import hickle as hkl

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu3,floatX=float32"
from theano import *

import theano.tensor as T
theano.__version__
from theano.sandbox.cuda import dnn

import theano

import pandas as pd
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle, gzip
from theano.tensor.nnet.conv import conv2d

from random import randint
from theano.compile.nanguardmode import NanGuardMode
import time

from PIL import Image

#from pom_funcs import *
from pom_room import POM_room
from pom_evaluator import POM_evaluator
config.allow_gc =False

import Config
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



# In[2]:


import UnariesNet
uNet = UnariesNet.unariesNet()


# In[3]:


uNet.run_bulk_features(save_features = False)


# In[13]:


print  len(Config.img_index_list)


# In[13]:


print uNet.run_features(0,0).shape


# In[14]:


plt.imshow(uNet.run_features(0,0))
plt.show()


# In[2]:


from PIL import Image


# In[8]:


im = Image.open('/cvlabdata1/cvlab/datasets_people_tracking/ETH/day_2/annotation_final/cvlab_camera2/begin/00000100.png')


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.use("nbagg")
import matplotlib.pyplot as plt

plt.imshow(im)
plt.show()


# In[ ]:




