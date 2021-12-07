from pylab import *
from numpy import *
import pandas as pd
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_iris
import pandas as pd
import os


class myFcm(object):
    def __init__(self, k=2):
        self.k = k
        self.MAX_ITER = 100
        self.n = 2
        self.m = 2.00