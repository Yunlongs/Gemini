from config import *
import pickle
import tensorflow as tf
import os
import glob
import csv
import networkx as nx
import numpy as np

def generate_all_func_dict():
    all_func_dict = {}
    for a in arch:
        count = 0
        for v in version:
            for c in compiler:
                for o in optimizer:
                    binary_dir = vulseeker_rawdata_dir + "_".join([v,a,c,o,"openssl.idb"])
                    function_list_csv = open(binary_dir + os.sep + "functions_list_fea.csv","r")
                    for line in csv.reader(function_list_csv):
                        cfg = read_cfg(line[0],binary_dir)
                        dfg = read_dfg(line[0],binary_dir)
                        node_size = len(cfg)
                        if node_size < min_nodes_threshold:
                            continue
                        if all_func_dict.get(line[0]) == None:
                            all_func_dict[line[0]] = []
                        feature = read_feature(line[0],binary_dir,node_size)
                        count += 1
                        assert len(cfg.nodes) == len(dfg.nodes) == feature.shape[0] ,"binary:%s func:%s cfg:%d dfg:%d and feature:%d_matrix's shape not consistent!" %(binary_dir,line[0],len(cfg.nodes),len(dfg.nodes),feature.shape[0])
                        g = (cfg,dfg,feature)
                        all_func_dict[line[0]].append(g)
                    function_list_csv.close()
        print(a + " :" + str(count))
    return all_func_dict



def read_cfg(funcname,binary_dir):
    cfg_path = binary_dir + os.sep + funcname + "_cfg.txt"
    cfg = nx.read_adjlist(cfg_path)
    return cfg

def read_dfg(funcname,binary_dir):
    dfg_path = binary_dir + os.sep + funcname + "_dfg.txt"
    dfg = nx.read_adjlist(dfg_path)
    return dfg

def read_feature(funcname,binary_dir,nodes_num):
    feat_matrix = np.zeros(shape=(nodes_num,vulseeker_feature_size),dtype=np.int)
    feature_path = binary_dir + os.sep + funcname + "_fea.csv"
    f = open(feature_path,"r")
    for i,line in enumerate(csv.reader(f)):
        feat_matrix[i,:] = line[8:8+vulseeker_feature_size]
    f.close()
    return feat_matrix



def dataset_split(all_function_dict):
    all_func_num = len(all_function_dict)
    train_func_num = int(all_func_num * 0.8)
    test_func_num = int(all_func_num * 0.1)

    train_name = np.random.choice(list(all_function_dict.keys()),size =train_func_num,replace=False)
    train_func = {}
    for func in train_name:
        train_func[func] = all_function_dict[func]
        all_function_dict.pop(func)

    with open(vulseeker_dataset_dir+"train","wb") as f:
        pickle.dump(train_func,f)

    test_func  = {}
    test_name = np.random.choice(list(all_function_dict.keys()),size = test_func_num,replace=False)
    for func in test_name:
        test_func[func] = all_function_dict[func]
        all_function_dict.pop(func)
    with open(vulseeker_dataset_dir + "test","wb") as f:
        pickle.dump(test_func,f)

    valid_func = all_function_dict
    valid_num = len(all_function_dict)
    with open(vulseeker_dataset_dir + "valid","wb") as f:
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

def zero_padded_featmat(feat_matrix,size):
    padded = np.zeros(shape=(size,vulseeker_feature_size))
    nodes = feat_matrix.shape[0]
    if nodes > size:
        padded = feat_matrix[0:size,:]
    else:
        padded[0:nodes,:] = feat_matrix
    return padded



def generate_pairs(type=b"train"):
    assert type == b"train" or type == b"test" or type == b"valid", "dataset type error!"
    filepath = vulseeker_dataset_dir + type.decode()
    with open(filepath,"rb") as f:
        func_dict = pickle.load(f)
    funcname_list = list(func_dict.keys())
    length = len(funcname_list)
    for funcname in funcname_list:
        func_list = func_dict[funcname]
        if len(func_list) < 2:
            continue
        for i,func in enumerate(func_list):
            cfg,dfg,feat_matrix = func
            cfg,dfg,feat_matrix = zero_padded_adjmat(cfg,max_nodes),zero_padded_adjmat(dfg,max_nodes),zero_padded_featmat(feat_matrix,max_nodes)
            for j in range(2):
                if j == 0:
                    index = np.random.randint(low = 0, high= len(func_list))
                    while index == i:
                        index = np.random.randint(low=0, high=len(func_list))
                    func_1 = func_list[index]
                    cfg_1,dfg_1,feat_matrix_1 = func_1
                    cfg_1,dfg_1,feat_matrix_1 = zero_padded_adjmat(cfg_1,max_nodes), zero_padded_adjmat(dfg_1,max_nodes),zero_padded_featmat(feat_matrix_1,max_nodes)
                    pair = (cfg,dfg,feat_matrix,cfg_1,dfg_1,feat_matrix_1,1)
                else:
                    index = np.random.randint(low = 0, high = length)
                    while funcname_list[index] == funcname:
                        index = np.random.randint(low=0, high=length)
                    g2_index = np.random.randint(low=0, high=len(func_dict[funcname_list[index]]))
                    func_2 = func_dict[funcname_list[index]][g2_index]
                    cfg_2,dfg_2,feat_matrix_2 = func_2
                    cfg_2,dfg_2,feat_matrix_2 = zero_padded_adjmat(cfg_2,max_nodes),zero_padded_adjmat(dfg_2,max_nodes),zero_padded_featmat(feat_matrix_2,max_nodes)
                    pair = (cfg,dfg,feat_matrix,cfg_2,dfg_2,feat_matrix_2,-1)
                yield pair


def dataset_generation(type="train"):
    data = tf.data.Dataset.from_generator(generate_pairs,output_types=(tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32),args=[type])
    data = data.repeat()
    data = data.shuffle(buffer_size=Buffer_Size)
    data = data.batch(batch_size=mini_batch)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data


if __name__ == '__main__':
    all_func_dict = generate_all_func_dict()
    dataset_split(all_func_dict)
