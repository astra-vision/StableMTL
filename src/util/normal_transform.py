# Author: Bingxin Ke
# Last modified: 2024-04-18

import torch
import logging
import numpy as np
import os
import matplotlib.pyplot as plt

def draw_normal(normal, filename="test.png"):
    normal_np = normal.cpu().permute(1, 2, 0).numpy()
    vis = normal_np
    vis = (1 - vis) / 2  # transform the interval from [-1, 1] to [0, 1]
    plt.imsave(filename, vis)
    
