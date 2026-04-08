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

# propa1d = Wave1D(config,model)
# propa1d.ccerjan()
# propa1d.eq1d()
# propa1d.plot()
# propa1d.animation()

propa2d = Wave2D(config,model)
propa2d.ccerjan()
propa2d.eq2D()
propa2d.plot2D()
#propa2d.animation2D()

