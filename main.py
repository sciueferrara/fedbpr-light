import numpy as np
import pandas as pd
import sys
import random
import os
import argparse
from modules import Server, ServerModel, Client, ClientModel, TripletSampler, SendStrategy
import utils.utils as utils
from progress.bar import IncrementalBar

np.random.seed(43)
random.seed(43)
np.set_printoptions(threshold=sys.maxsize)


def main(args):

    if not os.path.exists('results'):
        os.makedirs('results')

    #exp_type = utils.create_file_prefix(args.positive_fraction, args.with_delta, args.fraction, args.sampler_size)

    string = 'FedBPRPrivate-Frac' + str(args.fraction) + '-Samp' + args.sampler_size

    #processing_strategy = ProcessingStrategy.MultiProcessing() if args.mp else ProcessingStrategy.SingleProcessing()
    send_strategy = SendStrategy.SendDelta() if args.with_delta else SendStrategy.SendVector()

    for dataset in args.datasets:
        print("Working on", dataset, "dataset")

        if not os.path.exists('results/{}/recs'.format(dataset)):
            os.makedirs('results/{}/recs'.format(dataset))

        df = pd.read_csv('datasets/{}_trainingset.tsv'.format(dataset), sep='\t', names=['user_id', 'item_id', 'rating'])
        df, reverse_dict = utils.convert_unique_idx(df, 'item_id')
        user_size = len(df['user_id'].unique())
        item_size = len(df['item_id'].unique())
        print('Found {} users and {} items'.format(user_size, item_size))
        train_user_lists = utils.create_user_lists(df, user_size, 3)
        train_interactions_size = sum([len(user_list) for user_list in train_user_lists])
        print('{} interactions considered for training'.format(train_interactions_size))

        # Set parameters based on arguments
        if args.fraction == 0:
            round_modifier = int(train_interactions_size / user_size)
        else:
            round_modifier = int(train_interactions_size / (args.fraction * user_size))

        sampler_dict = {'single': 1,
                        'uniform': int(train_interactions_size/user_size)}
        sampler_size = sampler_dict.get(args.sampler_size)

        # Build final triplet samplers
        triplet_samplers = [TripletSampler(train_user_lists[u], item_size, sampler_size) for u in range(user_size)]

        for n_factors in args.n_factors:
            exp_setting_1 = "_F" + str(n_factors)
            for lr in args.lr:
                exp_setting_2 = exp_setting_1 + "_LR" + str(lr)

                # Create server and clients
                server_model = ServerModel(item_size, n_factors)
                server = Server(server_model, lr, args.fraction, args.mp, send_strategy)

                for pi in args.main_probability:
                    for q in args.q:
                        for p in args.p:
                            for h in args.imax:
                                exp_setting_3 = exp_setting_2 + "_pi" + str(pi) + "_q" + str(q) + "_p" + str(p) + "_h" + str(h)

                                clients = [Client(u, ClientModel(n_factors), (pi, q, p, h), train_user_lists[u],
                                                  sampler_size, item_size) for u in range(user_size)]

                                # Start training

                                for i in range(args.n_epochs * round_modifier):
                                    if i % round_modifier == 0:
                                        bar = IncrementalBar('Epoch ' + str(int(i / round_modifier + 1)), max=round_modifier)
                                    bar.next()
                                    server.train_model(clients)

                                    # Evaluation
                                    if ((i + 1) % (args.eval_every * round_modifier)) == 0:
                                        exp_setting_4 = exp_setting_3 + "_I" + str((i + 1) / round_modifier)
                                        results = server.predict(clients, max_k=100)
                                        with open('results/{}/recs/{}.tsv'.format(dataset, string+exp_setting_4), 'w') as out:
                                            for u in range(len(results)):
                                                for e, t in results[u].items():
                                                    out.write(str(u) + '\t' + str(reverse_dict[e]) + '\t' + str(t) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', help='Set the datasets you want to use', required=True)
    parser.add_argument('-F', '--n_factors', nargs='+', help='Set the latent factors you want', type=int, required=True)
    parser.add_argument('-pi', '--main_probability', nargs='+', type=float, help='Set the fraction of positive item to send (default 0)')
    parser.add_argument('-q', '--q', nargs='+', help='Set the fraction of positive item to send (default 0)', type=float)
    parser.add_argument('-p', '--p', nargs='+', help='Set the fraction of positive item to send (default 0)', type=float)
    parser.add_argument('-imax', '--imax', nargs='+', help='Set the fraction of positive item to send (default 0)', type=int)
    parser.add_argument('-U', '--fraction', help='Set the fraction of clients per round (0 for just one client)', type=float, default=0, required=True)
    parser.add_argument('-T', '--sampler_size', help='Set the sampler size: single for 1, uniform for R/U')
    parser.add_argument('-lr', '--lr', nargs='+', help='Set the learning rates', type=float, required=True)
    parser.add_argument('-E', '--n_epochs', help='Set the number of epochs', type=int, required=True)
    parser.add_argument('--with_delta', action='store_true', help='Use if you want server to send deltas instead of overwriting item information')
    parser.add_argument('--validation_size', help='Set a validation size, if needed', type=float, default=0)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--mp', action='store_true', help='Use if you want to use multiprocessing (if fraction > 0)')
    parsed_args = parser.parse_args()
    main(parsed_args)
