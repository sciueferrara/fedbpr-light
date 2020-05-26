import numpy as np


class ClientModel:
    def __init__(self, n_factors):
        self.item_vecs = None
        self.item_bias = None
        self.user_vec = np.random.randn(n_factors) / 10

    def predict(self, server_model):
        return np.dot(server_model.item_vecs, self.user_vec) + server_model.item_bias

    def predict_one(self, i, server_model):
        return np.dot(server_model.item_vecs[i], self.user_vec) + server_model.item_bias[i]
