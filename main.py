# main.py
import time

from src import *

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


PATH = "config/parameter.json"


config = Config(PATH).load()

model = Model(config)
model.disp()
model.geo()
model.create()
model.plotmodel()