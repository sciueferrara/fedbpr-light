import random
import multiprocessing
from .Worker import Worker
random.seed(43)
import copy

class Server:
    def __init__(self, model, lr, fraction, positive_fraction, mp, send_strategy):
        self.mp = mp
        self._send_strategy = send_strategy
        self.model = model
        self.bak_model = None
        self.lr = lr
        self.fraction = fraction
        self.positive_fraction = positive_fraction

    def select_clients(self, clients, fraction=0.1):
        if fraction == 0:
            idx = random.sample(range(len(clients)), 1)
        else:
            idx = random.sample(range(len(clients)), int(fraction*len(clients)))
        return idx

    def train_on_client(self, clients, i):
        clients[i].train(self.lr, self.positive_fraction, self.model)
        # for k, v in resulting_dic.items():
        #     self.model.item_vecs[k] += self.lr * v
        # for k, v in resulting_bias.items():
        #     self.model.item_bias[k] += self.lr * v

    def train_model(self, clients):
        item_vecs_bak, item_bias_bak = self._send_strategy.backup_item_vectors(self.model) or (None, None)
        c_list = self.select_clients(clients, self.fraction)
        #for i in c_list:
        #    self._send_strategy.send_item_vectors(clients, i, self.model)
        if not self.mp:
            if len(c_list) > 1:
                self.bak_model = copy.deepcopy(self.model)
                for i in c_list:
                    clients[i].train_parallel(self.lr, self.positive_fraction, self.bak_model, self.model)
                self.bak_model = None
            else:
                for i in c_list:
                    self.train_on_client(clients, i)
        else:
            #TODO: Multiprocessing is not working properly
            tasks = multiprocessing.JoinableQueue()
            num_workers = multiprocessing.cpu_count() - 1
            workers = [Worker(tasks, self.train_on_client, clients) for _ in range(num_workers)]
            for w in workers:
                w.start()
            for i in c_list:
                tasks.put((clients, i))
            for i in range(num_workers):
                tasks.put(None)
            tasks.join()
        self._send_strategy.update_deltas(self.model, item_vecs_bak, item_bias_bak)

    def predict(self, clients, max_k):
        predictions = []
        for i, c in enumerate(clients):
            #self._send_strategy.send_item_vectors(clients, i, self.model)
            predictions.append(c.predict(self.model, max_k))
            #self._send_strategy.delete_item_vectors(clients, i)
        return predictions
