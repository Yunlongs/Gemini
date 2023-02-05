import tensorflow as tf
import config
import pickle
import numpy as np
import random
from Gemini_data_1 import zero_padded_adjmat,feature_vector

model = tf.keras.models.load_model(config.Gemini_model_save_path)



class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def read_acfg(filepath, out_path = "dataset/data.acfg"):
    """
    读取IDA从binary中提取到的属性控制流图(ACFG)，并且转化成之后可操作的格式。
    filepath: ACFG的路径
    out_path: 对应输出文件的存储路径
    """
    all_function_dict = {}
    with open(filepath, "rb") as f:
        #picklefile = pickle.load(StrToBytes(f))
        picklefile = pickle.load(f)
    for func in picklefile.raw_graph_list:
        if len(func.g.node) < config.min_nodes_threshold:
            continue
        if all_function_dict.get(func.funcname) == None:
            all_function_dict[func.funcname] = []
        all_function_dict[func.funcname].append(func.g)
    with open(out_path,"wb") as f:
        pickle.dump(all_function_dict,f)



def generate_embeddings(data_path = "dataset/data.acfg"):
    """
    从转换后的ACFG数据文件中，读取每个函数的图属性信息，并使用DL将每个函数转化成一个embedding vector。
    Return:
        embeddings: 词典。 key 为函数名， value为对应的embedding vector。
    Note:
        embeddings的key可能需要根据情况自己设置，比如说，key应当设置为“当前软件名称”+ “exe/dll 文件名称" + ”函数名“。
        本函数实验中，仅使用函数名作为key，但这对于大规模项目来说是不合适，且不利于最后的检索的，请务必进行更改。
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    embeddings = {}
    for funcname, graphs in data.items():
        for i,graph in enumerate(graphs):
            func_id = funcname + "_" + str(i)  ## 请在这里改变key的设置。
            g_adjmat = zero_padded_adjmat(graph, config.max_nodes)
            g_featmat = feature_vector(graph, config.max_nodes)
            g_adjmat = np.expand_dims(g_adjmat,0)
            g_featmat = np.expand_dims(g_featmat,0)
            g_adjmat = tf.convert_to_tensor(g_adjmat)
            g_featmat = tf.convert_to_tensor(g_featmat)
            input = (g_adjmat, g_featmat, g_adjmat, g_featmat)
            sim, g1_embedding, g2_embedding = model(input)
            embeddings[func_id] = np.squeeze(g1_embedding.numpy())
    return embeddings

def save_embeddings(embeddings, save_file = "output/embeddings.txt"):
    """
    将生成得到的embedding存储到一个文件中去，方便之后进行比较。
    """
    with open(save_file, "w") as f:
        for key, value in embeddings.items():
            vector = value.tolist()
            vector_str = " ".join([str(v) for v in vector])
            f.write(key + "|" + vector_str + "\n")
#save_embeddings(embeddings, "output/embeddings.txt")

def load_embedding(file):
    """
    load embedding from a file.
    In the file, the embedding format is:
    func | embedding.

    :param file: filepath
    :return:
    """
    embedding_dict = {}
    with open(file, "r") as f:
        for line in f.readlines():
            if len(line) <= 1:
                continue
            sp = line.strip().split("|")
            funcname = sp[0]
            vector = sp[1:][0]
            vector = [float(v) for v in vector.split()]
            embedding_dict[funcname] = np.array(vector)
    return embedding_dict

def norm(vector):
    """
    l2 norm.
    :param vector:
    :return:
    """
    res = np.sqrt(np.sum(np.square(vector)))
    return vector / res

def infer_similarity(target_embedding, embedding_file):
    """
    根据target函数的embedding，在对应的embedding_file中的embedding寻找相似的函数。
    target_embedding: 你要进行search的函数对应的embedding，在供应链场景中，这里应当为CVE对应的那个函数。
    embedding_file: 存储当前binary所有函数的embedding的文件。
    Return:
        用print输出10个相似性最高的函数。
    """
    my_embeddings =  load_embedding(embedding_file)
    results = {}
    for funcname, embedding in my_embeddings.items():
        sim = np.dot(norm(target_embedding), norm(embedding))
        results[funcname] = sim
    results = sorted(results.items(), key=lambda d: d[1], reverse=True)
    for i in range(10): # 输出排序的前十个结果
        result = results[i]
        key, sim = result
        print(" ".join([key, str(sim)]))

def random_test():
    my_embeddings = load_embedding("output/embeddings.txt")
    target_funcs = np.random.choice(list(my_embeddings.keys()), 10)
    for target_func in target_funcs:
        target_embedding = my_embeddings[target_func]
        infer_similarity(target_embedding, "output/embeddings.txt")
        print("--------------------")


if __name__ == "__main__":
    #random_test()
    read_acfg("../dataset/dbghelp_w32.dll.cfg")
    generate_embeddings()

