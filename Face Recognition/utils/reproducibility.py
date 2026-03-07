"""
Global seed setting utility for reproducibility.
"""
import numpy as np
import tensorflow as tf
import random

def set_global_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
