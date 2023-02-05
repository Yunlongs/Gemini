# 基于DL的binary code相似性比较

## 使用方法
**NOTE:**
    使用此模型之前，应当确保已经使用IDA插件脚本(raw-feature-extractor)，把所需要的binary的ACFG提取完毕。

先说明一下，我们这个模型进行预测（训练完毕后）的流程， 以检测指定CVE为例子：

假设，目标软件集合为$P$，目标软件集合中的所有函数集合为$R(P)$， CVE对应的binary为$T$，CVE对应的函数为$F(T)$。

首先，使用`raw_feature_extractor`插件脚本提取出所有函数的ACFG，记为$ACFG(R(P))$和$ACFG(F(T))$.

使用模型，根据每个函数的ACFG，生成一个embedding vector，记为$E(R(P))$和$E(F(T))$。

比较CVE对应函数与所有函数之间的相似性: sim(R(P)) = cosine(E(R(P)), E(F(T)))。

根据相似性分数，进行排序，并设定阈值，即可识别那些软件含有目标CVE函数。

## 具体实现接口
以上描述的所有步骤，我均封装到了`similarity_inference.py`脚本中。

**1.读取ACFG，并转化为模型输入数据文件的接口。**
```buildoutcfg
def read_acfg(filepath, out_path = "dataset/data.acfg"):
    """
    读取IDA从binary中提取到的属性控制流图(ACFG)，并且转化成之后可操作的格式。
    filepath: ACFG的路径
    out_path: 对应输出文件的存储路径
    """
```

**2.根据数据文件，生成每个函数的embedding的接口。**
```buildoutcfg

def generate_embeddings(data_path = "dataset/data.acfg"):
    """
    从转换后的ACFG数据文件中，读取每个函数的图属性信息，并使用DL将每个函数转化成一个embedding vector。
    Return:
        embeddings: 词典。 key 为函数名， value为对应的embedding vector。
    Note:
        embeddings的key可能需要根据情况自己设置，比如说，key应当设置为“当前软件名称”+ “exe/dll 文件名称" + ”函数名“。
        本函数实验中，仅使用函数名作为key，但这对于大规模项目来说是不合适，且不利于最后的检索的，请务必进行更改。
    """
```

**3. 存储embedding的接口。**
```buildoutcfg
def save_embeddings(embeddings, save_file = "output/embeddings.txt"):
    """
    将生成得到的embedding存储到一个文件中去，方便之后进行比较。
    """
```
**4. 加载embedding的接口。**
```buildoutcfg
def load_embedding(file):
    """
    load embedding from a file.
    In the file, the embedding format is:
    func | embedding.

    :param file: filepath
    :return:
    """
```

**5. 推断目标函数与其他函数相似性的接口**
```buildoutcfg
def infer_similarity(target_embedding, embedding_file):
    """
    根据target函数的embedding，在对应的embedding_file中的embedding寻找相似的函数。
    target_embedding: 你要进行search的函数对应的embedding，在供应链场景中，这里应当为CVE对应的那个函数。
    embedding_file: 存储当前binary所有函数的embedding的文件。
    Return:
        用print输出10个相似性最高的函数。
    """
```