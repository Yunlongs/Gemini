from config import *
import pickle
import tensorflow as tf
import os
import glob
import csv
import networkx as nx
import numpy as np

with open(Gemini_dataset_dir+"train", "rb") as f:
    picklefile = pickle.load(f)
print(picklefile[0])
