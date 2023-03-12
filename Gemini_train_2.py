import tensorflow as tf
from tensorflow.keras import layers
from Gemini_data_1 import generate_pairs, dataset_generation, zero_padded_adjmat, feature_vector
import config
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# from Gemini.config import Gemini_model_save_path


class embedding_layer(layers.Layer):
    def __init__(self):
        super(embedding_layer, self).__init__()

    def build(self, input_shape):
        self.theta = self.add_weight(name="layer0", shape=tf.TensorShape(
            [config.embedding_size, config.embedding_size]))
        self.theta1 = self.add_weight(name="layer1", shape=tf.TensorShape(
            [config.embedding_size, config.embedding_size]))
        super(embedding_layer, self).build(input_shape)

    def call(self, input):
        '''
        :param input:shape = (batch,embedding_size,nodes)
        :return:
        '''
        curr_embedding = tf.einsum('ik,akj->aij', self.theta, input)
        curr_embedding = tf.nn.relu(curr_embedding)
        curr_embedding = tf.einsum('ik,akj->aij', self.theta1, curr_embedding)
        # curr_embedding = tf.nn.relu(curr_embedding)
        return curr_embedding

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape)
        return shape


def compute_graph_embedding(adjmat, feature_mat, W1, W2, embed_layer):
    '''
    adjmat: shape = (batch,max_nodes,max_nodes)
    feature_mat: shape = (batch,max_nodes,9)
    W1: shape = (embedding_size,9)
    W2: shape = (embedding_size,embedding_size)
    '''
    feature_mat = tf.einsum(
        'aij->aji', feature_mat)  # shape = (batch,9,max_nodes)

    init_embedding = tf.zeros(shape=(adjmat.shape[1], config.embedding_size))
    # shape = (batch,nodes,embedding_size)
    prev_embedding = tf.einsum('aik,kj->aij', adjmat, init_embedding)
    # shape = (batch,embedding_size,nodes)
    prev_embedding = tf.einsum('aij->aji', prev_embedding)
    for iter in range(config.T):
        # shape = (batch,embedding_size,nodes)
        neighbor_embedding = embed_layer(prev_embedding)
        # shape=(batch,embedding_size,nodes)
        term = tf.einsum('ik,akj->aij', W1, feature_mat)
        curr_embedding = tf.nn.tanh(term + neighbor_embedding)
        # shape=(batch,embedding_size,nodes)
        prev_embedding = curr_embedding
        # shape = (batch,nodes,embedding_size)
        prev_embedding = tf.einsum('aij->aji', prev_embedding)
        # shape = (batch,nodes,embedding_size)
        prev_embedding = tf.einsum('aik,akj->aij', adjmat, prev_embedding)
        # shape =(batch,embedding_size,nodes)
        prev_embedding = tf.einsum('aij->aji', prev_embedding)
    # shape = (batch,embedding_size)
    graph_embedding = tf.reduce_sum(curr_embedding, axis=2)
    graph_embedding = tf.einsum('ij->ji', graph_embedding)
    # shape = (embedding_size,batch)
    graph_embedding = tf.matmul(W2, graph_embedding)
    return graph_embedding


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embed_layer = embedding_layer()
        self.W1 = tf.Variable(tf.random.uniform(
            [config.embedding_size, config.Gemini_feature_size], maxval=0.1, dtype=tf.float32))
        self.W2 = tf.Variable(tf.random.uniform(
            [config.embedding_size, config.embedding_size], maxval=0.2, dtype=tf.float32))

    def call(self, inputs, training=None, mask=None):
        g1_adjmat, g1_feature_mat, g2_adjmat, g2_feature_mat = inputs
        g1_embedding = compute_graph_embedding(
            g1_adjmat, g1_feature_mat, self.W1, self.W2, self.embed_layer)
        g2_embedding = compute_graph_embedding(
            g2_adjmat, g2_feature_mat, self.W1, self.W2, self.embed_layer)
        sim_score = cosine(g1_embedding, g2_embedding)
        return sim_score, g1_embedding, g2_embedding


def cosine(q, a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(q), axis=0))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(a), axis=0))
    pooled_mul_12 = tf.reduce_sum(tf.multiply(q, a), axis=0)
    score = tf.divide(pooled_mul_12, pooled_len_1 *
                      pooled_len_2 + 0.0001, name="scores")
    return score


def loss(model, g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y):
    input = (g1_adjmat, g1_featmat, g2_adjmat, g2_featmat)
    sim, g1_embedding, g2_embedding = model(input)
    if tf.reduce_max(sim) > 1 or tf.reduce_min(sim) < -1:
        sim = sim * 0.999
    loss_value = tf.reduce_sum(tf.square(tf.subtract(sim, y)))
    return loss_value, sim, g1_embedding, g2_embedding


def grad(model, g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y):
    with tf.GradientTape() as tape:
        loss_value, sim, g1_embedding, g2_embedding = loss(
            model, g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), sim, g1_embedding, g2_embedding


def valid(model):
    valid_dataset = dataset_generation(type="valid")
    epoch_loss_avg_valid = tf.keras.metrics.Mean()
    epoch_accuracy_avg_valid = tf.keras.metrics.BinaryAccuracy()
    epoch_auc_avg = tf.keras.metrics.AUC()
    step = 0
    print("-------------------------")
    for g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y in valid_dataset:
        loss_value, grads, sim, _, _ = grad(
            model, g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y)
        epoch_loss_avg_valid(loss_value)
        epoch_accuracy_avg_valid.update_state(y, sim)
        sim = (sim + 1) / 2
        epoch_auc_avg.update_state(y, sim)

        if step % (config.valid_step_pre_epoch//100) == 0:
            print("valid step {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(step, epoch_loss_avg_valid.result(),
                                                                                          epoch_accuracy_avg_valid.result(), epoch_auc_avg.result()))
            if step == config.valid_step_pre_epoch:
                break
        step += 1
    print("----------------------------")
    return epoch_loss_avg_valid.result(), epoch_accuracy_avg_valid.result(), epoch_auc_avg.result()


def test(model):
    epoch_loss_avg_test = tf.keras.metrics.Mean()
    epoch_accuracy_avg_test = tf.keras.metrics.BinaryAccuracy()
    epoch_auc_sim = []
    epoch_auc_ytrue = []
    test_dataset = dataset_generation(type="test")
    step = 0
    for g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y in test_dataset:
        loss_value, grads, sim, _, _ = grad(
            model, g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y)
        epoch_loss_avg_test(loss_value)
        epoch_accuracy_avg_test.update_state(y, sim)

        epoch_auc_sim = np.concatenate((epoch_auc_sim, sim))
        epoch_auc_ytrue = np.concatenate((epoch_auc_ytrue, y))
        fpr, tpr, thres = roc_curve(
            epoch_auc_ytrue, epoch_auc_sim, pos_label=1)
        auc_score = auc(fpr, tpr)
        if step % (config.test_step_pre_epoch//100) == 0:
            print("test step {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(
                step, epoch_loss_avg_test.result(), epoch_accuracy_avg_test.result(), auc_score))
            if step == config.test_step_pre_epoch:
                break
        step += 1
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()
    print("----------------------------")


def train():
    optimizer = tf.optimizers.Adam(config.learning_rate)
    model = MyModel()
    model.build([(None, config.max_nodes, config.max_nodes), (None, config.max_nodes, config.Gemini_feature_size),
                (None, config.max_nodes, config.max_nodes), (None, config.max_nodes, config.Gemini_feature_size)])
    model.summary()
    max_auc = 0
    train_loss = []
    valid_loss = []
    train_auc = []
    valid_auc = []
    train_accuracy = []
    valid_accuracy = []
    for epoch in range(config.epochs):
        train_dataset = dataset_generation()
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.BinaryAccuracy(threshold=0)
        epoch_auc_avg = tf.keras.metrics.AUC()
        step = 0
        for g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y in train_dataset:
            loss_value, grads, sim, _, _ = grad(
                model, g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg(loss_value)
            y_pred = (sim+1)/2
            y = (y+1)/2
            epoch_accuracy_avg.update_state(y, y_pred)
            epoch_auc_avg.update_state(y, y_pred)
            if step % (config.step_per_epoch//100) == 0:
                print("step {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(
                    step, epoch_loss_avg.result(), epoch_accuracy_avg.result(), epoch_auc_avg.result()))
                if step == config.step_per_epoch:
                    break
            step += 1
        train_loss.append(epoch_loss_avg.result())
        train_accuracy.append(epoch_accuracy_avg.result())
        train_auc.append(epoch_auc_avg.result())
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, AUC: {:.3f}".format(
            epoch, epoch_loss_avg.result(), epoch_accuracy_avg.result(), epoch_auc_avg.result()))
        v_loss, v_accuracy, v_auc = valid(model)
        valid_loss.append(v_loss)
        valid_accuracy.append(v_accuracy)
        valid_auc.append(v_auc)
        if v_auc > max_auc:
            model.save(config.Gemini_model_save_path, save_format='tf')
            max_auc = v_auc
    test(model)
    plt.figure(figsize=(5, 4))
    plt.title("Loss curve")
    x = range(config.epochs)
    plt.plot(x, train_loss, label="train_loss")
    plt.plot(x, valid_loss, label="valid_loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(config.Gemini_fig_save_path+"loss.png")

    plt.figure(figsize=(5, 4))
    plt.title("Accuracy curve")
    plt.plot(x, train_accuracy, label="train_accuracy")
    plt.plot(x, valid_accuracy, label="valid_accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(config.Gemini_fig_save_path + "accuracy.png")

    plt.figure(figsize=(5, 4))
    plt.title("AUC curve")
    plt.plot(x, train_auc, label="train_auc")
    plt.plot(x, valid_auc, label="valid_auc")
    plt.xlabel("epochs")
    plt.ylabel("AUC")
    plt.legend()
    plt.savefig(config.Gemini_figure_save_path + "auc.png")


if __name__ == "__main__":
    train()
    # model = tf.keras.models.load_model(config.Gemini_model_save_path)
    # test(model)
