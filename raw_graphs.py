#coding:utf-8
import itertools
import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/')
import networkx as nx
#import numpy as np
from subprocess import Popen, PIPE
import pdb
import os
import re,mmap
#from graph_edit_new import *

class raw_graph:
	def __init__(self, funcname, g, func_f):
		'''
		funcname:函数名
		g: Genius的ACFG
		func_f:DiscovRe的特征
		'''
		self.funcname = funcname
		self.old_g = g[0]
		self.g = nx.DiGraph()
		self.entry = g[1]
		self.fun_features = func_f
		self.attributing()


class raw_graphs:  # 二进制文件内的所有原生控制流图
	def __init__(self, binary_name):
		self.binary_name = binary_name
		self.raw_graph_list = []

	def append(self, raw_g):
		self.raw_graph_list.append(raw_g)

	def __len__(self):
		return len(self.raw_graph_list)
