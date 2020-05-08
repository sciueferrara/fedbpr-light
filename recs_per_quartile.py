import numpy as np
import pandas as pd
import os
from utils import utils
import argparse


def get_user_quartiles(train_user_lists):
    dims = [len(l) for l in train_user_lists]
    q1 = np.quantile(dims, 0.25)
    q2 = np.quantile(dims, 0.5)
    q3 = np.quantile(dims, 0.75)

    diz = {}
    diz.update({u: 1 for u, l in enumerate(dims) if l < q1})
    diz.update({u: 2 for u, l in enumerate(dims) if q1 <= l < q2})
    diz.update({u: 3 for u, l in enumerate(dims) if q2 <= l < q3})
    diz.update({u: 4 for u, l in enumerate(dims) if l >= q3})

    return diz


def main(args):
    for dataset in args.datasets:
        print("Working on", dataset, "dataset")

        if not os.path.exists('results/Q-{}/recs/'.format(dataset)):
            os.makedirs('results/Q-{}/recs/'.format(dataset))

        names = ['user_id', 'item_id', 'rating', 'utc']
        df = pd.read_csv('datasets/' + dataset + '.tsv', sep='\t', dtype={'rating': 'float64', 'utc': 'int64'}, header=0, names=names)
        df = df.groupby('user_id').filter(lambda x: len(x) >= 20)
        df = utils.convert_unique_idx(df, 'user_id')
        df = utils.convert_unique_idx(df, 'item_id')
        user_size = len(df['user_id'].unique())
        total_user_lists = utils.create_user_lists(df, user_size)
        train_user_lists, _, _ = utils.split_train_test(total_user_lists,
                                                                                test_size=0.2,
                                                                                validation_size=args.validation_size)
        diz = get_user_quartiles(train_user_lists)

        for filename in os.listdir('results/{}/recs'.format(dataset)):
            if filename.startswith("P3Rec"):

                print(filename)

                f1 = open('results/Q-{}/recs/Q1-{}'.format(dataset, filename), 'w')
                f2 = open('results/Q-{}/recs/Q2-{}'.format(dataset, filename), 'w')
                f3 = open('results/Q-{}/recs/Q3-{}'.format(dataset, filename), 'w')
                f4 = open('results/Q-{}/recs/Q4-{}'.format(dataset, filename), 'w')

                file_dict = {
                    1: f1,
                    2: f2,
                    3: f3,
                    4: f4
                }

                with open('results/{}/recs/{}'.format(dataset, filename)) as f:
                    for l in f:
                        u = l.split('\t')[0]
                        print(diz[u])
                        file_dict.get(diz[u]).write(l)

                f1.close()
                f2.close()
                f3.close()
                f4.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', help='Set the datasets you want to use', required=True)
    parser.add_argument('--validation_size', help='Set a validation size, if needed', type=float, default=0)
    parsed_args = parser.parse_args()
    main(parsed_args)


