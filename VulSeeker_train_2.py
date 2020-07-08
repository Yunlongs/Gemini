import tensorflow as tf
from tensorflow.keras import layers
from data_vulseeker_1 import generate_pairs,dataset_generation,zero_padded_adjmat
from config import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve,auc


class cfg_embedding_layer(layers.Layer):
    def __init__(self):
        super(cfg_embedding_layer,self).__init__()

    def build(self, input_shape):
        self.theta = self.add_weight(name="P0",shape=tf.TensorShape([embedding_size,embedding_size]))
        self.theta1 = self.add_weight(name="P1",shape=tf.TensorShape([embedding_size,embedding_size]))
        super(cfg_embedding_layer,self).build(input_shape)

    def call(self,input):
        '''
        :param input:shape = (batch,embedding_size,nodes)
        :return:
        '''
        curr_embedding = tf.einsum('ik,akj->aij',self.theta,input)
        curr_embedding = tf.nn.relu(curr_embedding)
        curr_embedding = tf.einsum('ik,akj->aij',self.theta1,curr_embedding)
        #curr_embedding = tf.nn.relu(curr_embedding)
        return curr_embedding

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape)
        return shape

class dfg_embedding_layer(layers.Layer):
    def __init__(self):
        super(dfg_embedding_layer,self).__init__()

    def build(self, input_shape):
        self.theta = self.add_weight(name="Q0",shape=tf.TensorShape([embedding_size,embedding_size]))
        self.theta1 = self.add_weight(name="Q1",shape=tf.TensorShape([embedding_size,embedding_size]))
        super(dfg_embedding_layer,self).build(input_shape)

    def call(self,input):
        '''
        :param input:shape = (batch,embedding_size,nodes)
        :return:
        '''
        curr_embedding = tf.einsum('ik,akj->aij',self.theta,input)
        curr_embedding = tf.nn.relu(curr_embedding)
        curr_embedding = tf.einsum('ik,akj->aij',self.theta1,curr_embedding)
        #curr_embedding = tf.nn.relu(curr_embedding)
        return curr_embedding

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape)
        return shape





def compute_graph_embedding(cfg_adjmat,dfg_adjmat,feature_mat,W1,W2,cfg_embed_layer,dfg_embed_layer):
    '''
    cfg_adjmat: shape = (batch,max_nodes,max_nodes)
    dfg_adjmat: shape = (batch,max_nodes,max_nodes)
    feature_mat: shape = (batch,max_nodes,feature_size)
    W1: shape = (embedding_size,feature_size)
    W2: shape = (embedding_size,embedding_size)
    '''
    feature_mat = tf.einsum('aij->aji',feature_mat) #shape = (batch,feature_size,max_nodes)

    init_embedding = tf.zeros(shape=(max_nodes,embedding_size))
    cfg_prev_embedding = tf.einsum('aik,kj->aij', cfg_adjmat, init_embedding)  # shape = (batch,nodes,embedding_size)
    cfg_prev_embedding = tf.einsum('aij->aji', cfg_prev_embedding)  # shape = (batch,embedding_size,nodes)
    dfg_prev_embedding = tf.einsum('aik,kj->aij',dfg_adjmat,init_embedding) # shape = (batch,nodes,embedding_size)
    dfg_prev_embedding = tf.einsum('aij->aji',dfg_prev_embedding) # shape = (batch,embedding_size,nodes)
    for iter in range(T):
        cfg_neighbor_embedding = cfg_embed_layer(cfg_prev_embedding)  #shape = (batch,embedding_size,nodes)
        dfg_neighbor_embedding = dfg_embed_layer(dfg_prev_embedding) #shape = (batch,embedding_size,nodes)
        term = tf.einsum('ik,akj->aij', W1, feature_mat)  # shape=(batch,embedding_size,nodes)
        curr_embedding = tf.nn.tanh(term + cfg_neighbor_embedding + dfg_neighbor_embedding)
        prev_embedding = curr_embedding                 # shape=(batch,embedding_size,nodes)
        prev_embedding = tf.einsum('aij->aji',prev_embedding) # shape = (batch,nodes,embedding_size)
        cfg_prev_embedding = tf.einsum('aik,akj->aij',cfg_adjmat,prev_embedding) #shape = (batch,nodes,embedding_size)
        cfg_prev_embedding = tf.einsum('aij->aji',cfg_prev_embedding) #shape =(batch,embedding_size,nodes)
        dfg_prev_embedding = tf.einsum('aik,akj->aij',dfg_adjmat,prev_embedding)
        dfg_prev_embedding = tf.einsum('aij->aji',dfg_prev_embedding)
    graph_embedding = tf.reduce_sum(curr_embedding,axis=2)  #shape = (batch,embedding_size)
    graph_embedding = tf.einsum('ij->ji',graph_embedding)
    graph_embedding = tf.matmul(W2,graph_embedding) #shape = (embedding_size,batch)
    return graph_embedding

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.cfg_embed_layer = cfg_embedding_layer()
        self.dfg_embed_layer = dfg_embedding_layer()
        self.W1 = tf.Variable(tf.random.uniform([embedding_size, vulseeker_feature_size], maxval=0.1, dtype=tf.float32))
        self.W2 = tf.Variable(tf.random.uniform([embedding_size, embedding_size], maxval=0.2, dtype=tf.float32))

    def call(self, inputs, training=None, mask=None):
        g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat = inputs
        g1_embedding = compute_graph_embedding(g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,self.W1,self.W2,self.cfg_embed_layer,self.dfg_embed_layer)
        g2_embedding = compute_graph_embedding(g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,self.W1,self.W2,self.cfg_embed_layer,self.dfg_embed_layer)
        sim_score = cosine(g1_embedding, g2_embedding)
        return sim_score,g1_embedding,g2_embedding

def cosine(q,a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(q),axis=0))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(a),axis=0))
    pooled_mul_12 = tf.reduce_sum(tf.multiply(q,a), axis=0)
    score = tf.divide(pooled_mul_12, pooled_len_1 * pooled_len_2 +0.0001, name="scores")
    return score

def loss(model,g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y):
    """
    Get the model's output(two graph's embeddings and their similarity),return the loss.
    :param model:
    :param g1_cfg_adjmat:
    :param g1_dfg_adjmat:
    :param g1_featmat:
    :param g2_cfg_adjmat:
    :param g2_dfg_adjmat:
    :param g2_featmat:
    :param y:
    :return:
    """
    input = (g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat)
    sim,g1_embedding,g2_embedding = model(input)
    if tf.reduce_max(sim)>1 or tf.reduce_min(sim)<-1:
        sim = sim * 0.999  # Here because the float num computation can overflow,such as 1.00000001.
    loss_value = tf.reduce_sum(tf.square(tf.subtract(sim,y)))
    return loss_value,sim,g1_embedding,g2_embedding

def grad(model,g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y):
    with tf.GradientTape() as tape:
        loss_value,sim,g1_embedding,g2_embedding = loss(model,g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y)
    return loss_value,tape.gradient(loss_value,model.trainable_variables),sim,g1_embedding,g2_embedding


def valid(model):
    valid_dataset = dataset_generation(type="valid")
    epoch_loss_avg_valid = tf.keras.metrics.Mean()
    epoch_accuracy_avg_valid = tf.keras.metrics.BinaryAccuracy()
    epoch_auc_avg = tf.keras.metrics.AUC()
    step = 0
    print("-------------------------")
    for g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y in valid_dataset:
        loss_value, grads, sim, _, _ = grad(model, g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y)
        epoch_loss_avg_valid(loss_value)
        epoch_accuracy_avg_valid.update_state(y, sim)
        sim = (sim + 1) / 2
        epoch_auc_avg.update_state(y,sim)

        if step % (valid_step_pre_epoch//100) == 0:
            print("valid step {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(step, epoch_loss_avg_valid.result(),
                                                                       epoch_accuracy_avg_valid.result(),epoch_auc_avg.result()))
            if step == valid_step_pre_epoch:
                break
        step += 1
    print("----------------------------")
    return epoch_loss_avg_valid.result(),epoch_accuracy_avg_valid.result(),epoch_auc_avg.result()

def test(model):
    epoch_loss_avg_test = tf.keras.metrics.Mean()
    epoch_accuracy_avg_test = tf.keras.metrics.BinaryAccuracy()
    epoch_auc_sim = []
    epoch_auc_ytrue = []
    test_dataset = dataset_generation(type="test")
    step = 0
    for g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y in test_dataset:
        loss_value, grads, sim, _, _ = grad(model, g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y)
        epoch_loss_avg_test(loss_value)
        epoch_accuracy_avg_test.update_state(y, sim)

        epoch_auc_sim = np.concatenate((epoch_auc_sim,sim))
        epoch_auc_ytrue = np.concatenate((epoch_auc_ytrue,y))
        fpr,tpr,thres = roc_curve(epoch_auc_ytrue,epoch_auc_sim,pos_label=1)
        auc_score = auc(fpr,tpr)
        if step % (test_step_pre_epoch//100) == 0:
            print("test step {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(step, epoch_loss_avg_test.result(),epoch_accuracy_avg_test.result(),auc_score))
            if step == test_step_pre_epoch:
                break
        step += 1
    plt.plot(fpr,tpr)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()
    print("----------------------------")


def train():
    optimizer = tf.optimizers.Adam(learning_rate)
    model = MyModel()
    model.build([(None,max_nodes,max_nodes),(None,max_nodes,max_nodes),(None,max_nodes,vulseeker_feature_size),(None,max_nodes,max_nodes),(None,max_nodes,max_nodes),(None,max_nodes,vulseeker_feature_size)])
    model.summary()
    max_auc = 0
    train_loss =[]
    valid_loss = []
    train_auc = []
    valid_auc = []
    train_accuracy = []
    valid_accuracy = []
    for epoch in range(epochs):
        train_dataset = dataset_generation()
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.BinaryAccuracy()
        epoch_auc_avg = tf.keras.metrics.AUC()
        step = 0
        for g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y in train_dataset:
            loss_value,grads,sim,_,_ = grad(model,g1_cfg_adjmat,g1_dfg_adjmat,g1_featmat,g2_cfg_adjmat,g2_dfg_adjmat,g2_featmat,y)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))

            epoch_loss_avg(loss_value)
            epoch_accuracy_avg.update_state(y,sim)
            sim = (sim+1)/2
            epoch_auc_avg.update_state(y,sim)
            if step%(step_per_epoch//100)==0:
                print("step {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(step,epoch_loss_avg.result(),epoch_accuracy_avg.result(),epoch_auc_avg.result()))
                if step==step_per_epoch:
                    break
            step +=1
        train_loss.append(epoch_loss_avg.result())
        train_accuracy.append(epoch_accuracy_avg.result())
        train_auc.append(epoch_auc_avg.result())
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(epoch,epoch_loss_avg.result(),epoch_accuracy_avg.result(),epoch_auc_avg.result()))
        v_loss,v_accuracy,v_auc = valid(model)
        valid_loss.append(v_loss)
        valid_accuracy.append(v_accuracy)
        valid_auc.append(v_auc)
        if v_auc>max_auc:
            model.save(vulseeker_model_save_path, save_format='tf')
            max_auc = v_auc
    test(model)
    plt.figure()
    x = range(epochs)
    plt.plot(x,train_loss,label="train_loss")
    plt.plot(x,valid_loss,label ="valid_loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x,train_accuracy,label="train_accuracy")
    plt.plot(x,valid_accuracy,label = "valid_accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x,train_auc,label="train_auc")
    plt.plot(x,valid_auc,label="valid_auc")
    plt.xlabel("epochs")
    plt.ylabel("AUC")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()
