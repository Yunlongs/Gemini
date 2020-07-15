import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def generate_features(type = "train"):
    '''
    generate embedding features for each function
    :param type: 'train' or 'valid' or 'test'
    :return:
    '''
    model = tf.keras.models.load_model("output/Gemini/Experiment_4/model_weight")
    with open("dataset/"+type,"rb") as f:
        func_dict = pickle.load(f)
    feature_list = []
    feature_funcname_map = {}
    func_name_list = []
    func_label = {}
    for func_name in func_dict.keys():
        count = 0
        for g in func_dict[func_name]:
            g_adjmat = zero_padded_adjmat(g, max_nodes)
            g_featmat = feature_vector(g, max_nodes)
            g_adjmat = tf.expand_dims(g_adjmat,0)
            g_featmat = tf.expand_dims(g_featmat,0)
            input = (g_adjmat,g_featmat,g_adjmat,g_featmat)
            _,embedding,_ = model(input)
            feature_list.append(embedding)
            feature_funcname_map[embedding.ref()] = func_name
            func_name_list.append(func_name)
            count +=1
        func_label[func_name] = count

    with open("output/"+type+"_features","wb") as f:## embeddings of func squentially
        pickle.dump(feature_list,f)

    with open("output/"+type+"_feature_names","w") as f:
        for funcname in func_name_list:
            f.write(funcname+"\n")

    with open("output/func_label_count","wb") as f:
        pickle.dump(func_label,f)

    with open("output/feature_funcname_map","wb") as f: ## dict[features] = funcname
        pickle.dump(feature_funcname_map,f)


def random_select_evaluate_dataset(k=1000):
    '''
    random select k functions in test dataset
    :param k:
    :return: selected funcnames and corresponding embeddings
    '''
    model = tf.keras.models.load_model("output/Gemini/Experiment_4/model_weight")
    with open("dataset/test","rb") as f:
        test_dict = pickle.load(f)
    func_dict = {}
    for key in test_dict.keys():
        if len(test_dict[key])>1:
            func_dict[key] = test_dict[key]
    funcname_list = list(func_dict.keys())
    feature_list = []
    selected_funcname = np.random.choice(funcname_list,size=k//2,replace=False)
    for funcname in selected_funcname:
        nums = len(func_dict[funcname])
        rand_index = np.random.choice(range(nums),size=2)
        for index in rand_index:
            g = func_dict[funcname][index]
            g_adjmat = zero_padded_adjmat(g, max_nodes)
            g_featmat = feature_vector(g, max_nodes)
            g_adjmat = tf.expand_dims(g_adjmat, 0)
            g_featmat = tf.expand_dims(g_featmat, 0)
            input = (g_adjmat, g_featmat, g_adjmat, g_featmat)
            _, embedding, _ = model(input)
            feature_list.append(embedding)
    return selected_funcname,feature_list


def top_k_evaluate(selected_funcname,feature_list,topk_list):
    with open("output/test_features","rb") as f:
         test_features = pickle.load(f)
    with open("output/test_feature_names","r") as f:
        func_name_list = [line.strip() for line in f.readlines()]
    with open("output/func_label_count","rb") as f:
        func_label_count = pickle.load(f)
    test_features = tf.convert_to_tensor(test_features)
    test_features = tf.squeeze(test_features)
    test_features = tf.transpose(test_features)

    recalls = []
    for k in topk_list:
        total_recall = 0
        for i,func_name in enumerate(selected_funcname):
            count = min(k,func_label_count[func_name])
            for j in range(2):
                tp = 0
                feature = feature_list[i*2 +j]
                feature = tf.reshape(feature, shape=[-1, 1])
                score = np.array(cosine(test_features, feature))
                index = np.array(np.argsort(score))[-(k+1):-1].tolist()[::-1]
                for x in index:
                    if func_name == func_name_list[x]:
                        tp +=1
                recall = tp/count
                total_recall += recall
        mean_recall = total_recall/1000
        recalls.append(mean_recall)
        print("top"+str(k)+" recall:",mean_recall)
    plt.figure()
    plt.plot(topk_list,recalls)
    plt.xlabel("topk")
    plt.ylabel("recall")
    plt.show()


