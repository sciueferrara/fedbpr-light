import numpy as np


class Client:
    def __init__(self, client_id, model, privacy_params, train_user_list, sampler_size, item_size):
        self.id = client_id
        self.model = model
        self.train_user_list = train_user_list
        self.sampler_size = sampler_size
        self.item_size = item_size

        self.pi = privacy_params[0]
        self.q = privacy_params[1]
        self.p = privacy_params[2]

        # The first perturbation is here!
        self.c = - np.ones(self.item_size)
        self.c[list(self.train_user_list)] = 1
        self.pert = np.random.random(self.item_size)
        self.c = np.ma.masked_where(self.pert < self.pi, self.c).filled(-1)
        self.c = np.ma.masked_where(self.pert < 0.5 * self.pi, self.c).filled(1)

    def predict(self, server_model, max_k):
        result = self.model.predict(server_model)
        result[list(self.train_user_list)] = -np.inf
        # the very fast top_k :-)
        unordered_top_k = np.argpartition(result, -max_k)[-max_k:]
        top_k = unordered_top_k[np.argsort(result[unordered_top_k])][::-1]
        top_k_score = result[top_k]
        prediction = {top_k[i]: top_k_score[i] for i in range(len(top_k))}

        return prediction

    def train(self, lr, server_model):
        bias_reg = 0
        user_reg = lr / 20
        positive_item_reg = lr / 20
        negative_item_reg = lr / 200

        new_perturbed = - np.ones(self.item_size)
        new_perturbation = np.random.random(self.item_size)
        new_perturbed[new_perturbation < 0.5 * ((self.c + 1) * self.q - (self.c - 1) * self.p)] = 1

        pert_pos = np.argwhere(new_perturbed == 1).flatten()
        pert_neg = np.argwhere(new_perturbed == -1).flatten()

        positive_sampled = np.random.choice(pert_pos, len(self.train_user_list))
        negative_sampled = np.random.choice(pert_neg, len(self.train_user_list))

        x_p = np.dot(server_model.item_vecs[positive_sampled], self.model.user_vec) +\
              server_model.item_bias[positive_sampled]

        x_n = np.dot(server_model.item_vecs[negative_sampled], self.model.user_vec) +\
              server_model.item_bias[negative_sampled]

        x_pn = x_p - x_n
        d_loss = (1 / (1 + np.exp(x_pn))).reshape((-1, 1))

        wu = self.model.user_vec.copy()
        self.model.user_vec += lr * np.sum(
                d_loss * (server_model.item_vecs[positive_sampled] - server_model.item_vecs[negative_sampled]) - user_reg * wu,
        axis=0)

        server_model.item_vecs[positive_sampled] = np.add(server_model.item_vecs[positive_sampled],
                                           lr * (d_loss * wu - positive_item_reg * server_model.item_vecs[
                                               positive_sampled]))
        server_model.item_bias[positive_sampled] += lr * (d_loss.reshape(-1) - bias_reg * server_model.item_bias[positive_sampled])

        server_model.item_vecs[negative_sampled] = np.add(server_model.item_vecs[negative_sampled],
                                           lr * (d_loss * (-wu) - negative_item_reg * server_model.item_vecs[negative_sampled]))
        server_model.item_bias[negative_sampled] += - lr * (d_loss.reshape(-1) - bias_reg * server_model.item_bias[negative_sampled])
