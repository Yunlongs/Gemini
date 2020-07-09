import tensorflow as tf
from tensorflow.keras import layers
import glob
import pickle
import numpy as np
import networkx as nx
from config import *


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()



def read_cfg():
    all_function_dict = {}
    counts = []
    for a in arch:
        count = 0
        for v in version:
            for c in compiler:
                for o in optimizer:
                    filename = "_".join([v,a,c,o,"openssl"])
                    filepath = dir_name+filename+".cfg"
                    with open(filepath,"r") as f:
                        picklefile = pickle.load(StrToBytes(f))
                    for func in picklefile.raw_graph_list:
                        if len(func.g) < min_nodes_threshold:
                            continue
                        if all_function_dict.get(func.funcname) == None:
                            all_function_dict[func.funcname] = []
                        all_function_dict[func.funcname].append(func.g)
                        count +=1
        counts.append(count)
    print("for three arch:",counts)

    return all_function_dict

def dataset_split(all_function_dict):
    all_func_num = len(all_function_dict)
    train_func_num = int(all_func_num *0.8)
    test_func_num = int(all_func_num * 0.1)

    train_name = np.random.choice(list(all_function_dict.keys()),size =train_func_num,replace=False)
    train_func = {}
    for func in train_name:
        train_func[func] = all_function_dict[func]
        all_function_dict.pop(func)

    with open("dataset/train","wb") as f:
        pickle.dump(train_func,f)

    test_func  = {}
    test_name = np.random.choice(list(all_function_dict.keys()),size = test_func_num,replace=False)
    for func in test_name:
        test_func[func] = all_function_dict[func]
        all_function_dict.pop(func)
    with open("dataset/test","wb") as f:
        pickle.dump(test_func,f)

    valid_func = all_function_dict
    valid_num = len(all_function_dict)
    with open("dataset/valid","wb") as f:
        pickle.dump(valid_func,f)

    print("train dataset's num =%s ,valid dataset's num=%s , test dataset's num =%s"%(train_func_num,valid_num,test_func_num))

def adjmat(gr):
    return nx.adjacency_matrix(gr).toarray().astype('float32')

def zero_padded_adjmat(graph, size):
    unpadded = adjmat(graph)
    padded = np.zeros((size, size))
    if len(graph)>size:
        padded =  unpadded[0:size,0:size]
    else:
        padded[0:unpadded.shape[0], 0:unpadded.shape[1]] = unpadded
    return padded

def feature_vector(graph,size):
    feature_mat = np.zeros((size,9))
    for node in graph.nodes:
        if node==size:
            break
        feature = np.zeros((1,9))
        vector  = graph.nodes[node]['v']
        num_const = vector[0]
        if len(num_const)==1:
            feature[0,0] = num_const[0]
        elif len(num_const)>=2:
            feature[0,0:2] = np.sort(num_const)[::-1][:2]
        feature[0,2] =  len(vector[1])
        feature[0,3:] = vector[2:]
        feature_mat[node,:] = feature
    return feature_mat




def generate_pairs(type):
    assert type == b"train" or type == b"test" or type == b"valid","dataset type error!"
    filepath = "dataset/"+type.decode()
    with open(filepath,"rb") as f:
        func_dict = pickle.load(f)
    funcname_list = list(func_dict.keys())
    length = len(funcname_list)
    for funcname in func_dict.keys():
        func_list = func_dict[funcname]
        for i,g in enumerate(func_list):
            g_adjmat = zero_padded_adjmat(g,max_nodes)
            g_featmat = feature_vector(g,max_nodes)
            for j in range(2):
                if j==0:
                    g1_index = np.random.randint(low=0,high=len(func_list))
                    g1 = func_list[g1_index]
                    g1_adjmat = zero_padded_adjmat(g1,max_nodes)
                    g1_featmat = feature_vector(g1,max_nodes)
                    pair = (g_adjmat,g_featmat,g1_adjmat,g1_featmat,1)
                else:
                    index = np.random.randint(low=0,high = length)
                    while funcname_list[index] == funcname:
                        index = np.random.randint(low=0, high=length)
                    g2_index = np.random.randint(low=0,high = len(func_dict[funcname_list[index]]))
                    g2 = func_dict[funcname_list[index]][g2_index]
                    g2_adjmat = zero_padded_adjmat(g2,max_nodes)
                    g2_featmat = feature_vector(g2,max_nodes)
                    pair = (g_adjmat,g_featmat,g2_adjmat,g2_featmat,0)
                yield pair

def dataset_generation(type="train"):
    data = tf.data.Dataset.from_generator(generate_pairs,output_types=(tf.float32,tf.float32,tf.float32,tf.float32,tf.float32),args=[type])
    data = data.repeat()
    data = data.shuffle(buffer_size=Buffer_Size)
    data = data.batch(batch_size=mini_batch)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data


if __name__ == '__main__':
    all_func_dict = read_cfg()
    dataset_split(all_func_dict)





