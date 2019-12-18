#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use("nbagg")
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re, os, glob, pickle, shutil,sys
from random import randint
import time
from shutil import *

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
import theano
import theano.tensor as T
from theano import *
theano.__version__
from theano.sandbox.cuda import dnn

import pandas as pd
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


from theano.compile.nanguardmode import NanGuardMode

#from pom_funcs import *
from pom_room import POM_room
from pom_evaluator import POM_evaluator
import POMLayers1
from EM_funcs import *
import ZtoY
config.allow_gc =False

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import Config


# In[2]:


from GaussianNet import gaussianNet
parts_predictor = gaussianNet()
parts_predictor.run_inference(bg_pretrained = True,regression_pretrained = True,verbose = True)


# In[ ]:




