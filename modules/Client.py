import numpy as np
import random
from collections import deque
from itertools import starmap


class Client:
    def __init__(self, client_id, model, train, train_user_list, sampler_size):
        self.id = client_id
        self.model = model
        #self.train_set = train
        self.train_user_list = train_user_list
        self.sampler_size = sampler_size

    def predict(self, server_model, max_k):
        result = self.model.predict(server_model)
        result[list(self.train_user_list)] = -np.inf
        # the very fast top_k :-)
        unordered_top_k = np.argpartition(result, -max_k)[-max_k:]
        top_k = unordered_top_k[np.argsort(result[unordered_top_k])][::-1]
        top_k_score = result[top_k]
        prediction = {top_k[i]: top_k_score[i] for i in range(len(top_k))}

        return prediction

    def train(self, lr, positive_fraction, server_model):

        def sample_user_triples(training_list):
            training_set = set(training_list)
            for _ in range(len(training_set)):
                i = random.choice(training_list)
                j = random.randrange(self.item_size)
                while j in training_set:
                    j = random.randrange(self.item_size)
                yield i, j

        def compute_gradient(i, j):
            x_i = np.dot(server_model.item_vecs[i], self.model.user_vec) + server_model.item_bias[i]
            x_j = np.dot(server_model.item_vecs[j], self.model.user_vec) + server_model.item_bias[j]
            x_ij = x_i - x_j
            d_loss = 1 / (1 + np.exp(x_ij))

            wu = self.model.user_vec.copy()
            self.model.user_vec += lr * (d_loss * (server_model.item_vecs[i] - server_model.item_vecs[j]) - user_reg * wu)

            server_model.item_vecs[j] = np.add(server_model.item_vecs[j], lr * (d_loss * (-wu) - negative_item_reg * server_model.item_vecs[j]))
            server_model.item_bias[j] += - lr * (d_loss - bias_reg * server_model.item_bias[j])

            if positive_fraction:
                if random.random() >= 1 - positive_fraction:
                    server_model.item_vecs[i] = np.add(server_model.item_vecs[i],
                                                       lr * (d_loss * wu - positive_item_reg * server_model.item_vecs[i]))
                    server_model.item_bias[i] += lr * (d_loss - bias_reg * server_model.item_bias[j])

        bias_reg = 0
        user_reg = lr / 20
        positive_item_reg = lr / 20
        negative_item_reg = lr / 200

        sample = sample_user_triples(self.train_user_list)
        #sample = self.train_set.sample_user_triples()
        deque(starmap(lambda i, j: compute_gradient(i, j), sample), maxlen=0)


    def train_parallel(self, lr, positive_fraction, starting_model, target_model):

        def operation(i, j):
            x_i = self.model.predict_one(i, starting_model)
            x_j = self.model.predict_one(j, starting_model)
            x_ij = x_i - x_j
            d_loss = 1 / (1 + np.exp(x_ij))

            wu = self.model.user_vec.copy()
            self.model.user_vec += lr * (d_loss * (starting_model.item_vecs[i] - starting_model.item_vecs[j]) - user_reg * wu)

            target_model.item_vecs[j] = np.add(target_model.item_vecs[j], lr * (d_loss * (-wu) - negative_item_reg * starting_model.item_vecs[j]))
            target_model.item_bias[j] += - lr * (d_loss - bias_reg * starting_model.item_bias[j])

            if positive_fraction:
                if random.random() >= 1 - positive_fraction:
                    target_model.item_vecs[i] = np.add(target_model.item_vecs[i],
                                                       lr * (d_loss * wu - positive_item_reg * starting_model.item_vecs[i]))
                    target_model.item_bias[i] += lr * (d_loss - bias_reg * starting_model.item_bias[j])

        bias_reg = 0
        user_reg = lr / 20
        positive_item_reg = lr / 20
        negative_item_reg = lr / 200

        sample = self.train_set.sample_user_triples()
        deque(starmap(lambda i, j: operation(i, j), sample), maxlen=0)