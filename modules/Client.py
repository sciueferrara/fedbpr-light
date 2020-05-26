import numpy as np
import random
from collections import defaultdict, deque
from itertools import starmap

class Client:
    def __init__(self, client_id, model, train, train_user_list, sampler_size):
        self.id = client_id
        self.model = model
        self.train_set = train
        self.train_user_list = train_user_list
        self.sampler_size = sampler_size

    def predict(self, prediction, max_k):
        prediction[list(self.train_user_list)] = -np.inf
        top_k = prediction.argsort()[-max_k:][::-1]
        top_k_score = prediction[top_k]
        prediction = {top_k[i]: top_k_score[i] for i in range(len(top_k))}

        return prediction

    def train(self, lr, positive_fraction, server_model):

        def operation(i, j):
            x_i = self.model.predict_one(i, server_model)
            x_j = self.model.predict_one(j, server_model)
            x_ij = x_i - x_j
            d_loss = 1 / (1 + np.exp(x_ij))

            wu = self.model.user_vec.copy()
            self.model.user_vec += lr * (d_loss * (server_model.item_vecs[i] - server_model.item_vecs[j]) - user_reg * wu)

            server_model.item_vecs[j] = np.add(server_model.item_vecs[j], lr* d_loss * (-wu) - negative_item_reg * server_model.item_vecs[j])
            server_model.item_bias[j] += - lr * d_loss - bias_reg * server_model.item_bias[j]

            if positive_fraction:
                if random.random() >= 1 - positive_fraction:
                    server_model.item_vecs[i] = np.add(server_model.item_vecs[i],
                                                       lr * d_loss * (-wu) - positive_item_reg * server_model.item_vecs[i])
                    server_model.item_bias[i] += lr * d_loss - bias_reg * server_model.item_bias[j]

        bias_reg = 0
        user_reg = lr / 20
        positive_item_reg = lr / 20
        negative_item_reg = lr / 200

        sample = self.train_set.sample_user_triples()
        deque(starmap(lambda i, j: operation(i, j), sample), maxlen=0)
