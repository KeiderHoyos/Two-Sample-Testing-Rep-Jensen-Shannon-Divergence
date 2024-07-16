import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import argparse

from utils import sampler_mixture
from utils import RJSD_estimator
from rjsd_fuse import rjsd_fuse
parser = argparse.ArgumentParser()

parser.add_argument('-e', '--experiment-name', required=True, help='Experiment name for saved files', type=str)
parser.add_argument('-nPerm', '--permTestSize', required= True, help='Number of permutations to compute power of test',type=int )
parser.add_argument('-sign', '--significance', required= True, help='significance level to reject the null hypothesis', type=float )
parser.add_argument('-rId', '--repId',required=True, help = 'repetition Id to store the results', type=int)
parser.add_argument('-datafolder', '--DATAFOLDER', required = True, type = str )
parser.add_argument('-epochs', '--epochs', required = False, default = 200, type = int)
parser.add_argument('-lr', '--lr', required = False, default = 0.01, type = float)
parser.add_argument('-deep', '--deep', required = False, default = False, type = bool)
parser.add_argument('-vary_size', '--vary_size', required = False, default = False, type = bool)
parser.add_argument('-fuse', '--fuse', required = False, default = False, type = bool)


args = parser.parse_args()

repetitions = 200 # each core will do one repetition
# permTestSize = 100
# numTestSets = 10
# significance = 0.05

# This code runs for one repetition to allow multiprocessing to run the code in parallel. A total of 200 repetitions are run in parallel.

def run():
    fname = args.DATAFOLDER + '/' + str(args.experiment_name) + str(args.repId) + '.npz'
    print(args.vary_size)
    if not args.vary_size:
        p_order_approx = np.array([1, 2, 4, 6, 10])
        repetitions = 200
        shifts = (0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2)
        sample_size = 500
        outputs = np.zeros((len(p_order_approx),len(shifts), repetitions))
        outputs = outputs.tolist()
        seed = 42
        # set torch and numpy seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Check if a GPU is available and if not, use a CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for s in (range(len(shifts))):
            shift = shifts[s]
            print('shift:', shift)
            seed += args.repId - 1
            X, Y = sampler_mixture(seed = seed, m=sample_size, n=sample_size, d=2, mu=20, std_1=1, std_2=shift)
            X = X.to(device)
            Y = Y.to(device)
            if args.fuse:
                for p in range(len(p_order_approx)):
                    outputs[p][s][args.repId - 1] = rjsd_fuse(X, Y, seed, order_approx = p_order_approx[p], number_permutations=args.permTestSize, alpha = args.significance)
            else:
                X_train, X_test = X[:len(X)//2], X[len(X)//2:]
                Y_train, Y_test = Y[:len(Y) // 2], Y[len(Y) // 2:]
                
                for p in range(len(p_order_approx)):
                    RJSD_est = RJSD_estimator(isotropic = False, order_approx=p_order_approx[p], deep = args.deep)
                    RJSD_est.fit(X_train, Y_train, epochs = args.epochs, lr = args.lr, verbose=False)
                    outputs[p][s][args.repId - 1] = RJSD_est.permutation_test(X_test, Y_test, significance= args.significance,permutations=args.permTestSize, seed = seed)
    else: 
        sample_sizes = (500, 1000, 1500, 2000, 2500, 3000)
        shift = 1.3
        p_order_approx = np.array([1, 2, 4, 6, 10])
        repetitions = 200
        outputs = np.zeros((len(p_order_approx),len(sample_sizes), repetitions))
        outputs = outputs.tolist()
        seed = 42
        # set torch and numpy seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Check if a GPU is available and if not, use a CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for s,sample_size in enumerate(sample_sizes):
            print('shift:', shift, ', sample size:', sample_size)
            seed += args.repId - 1
            X, Y = sampler_mixture(seed = seed, m=sample_size, n=sample_size, d=2, mu=20, std_1=1, std_2=shift)
            X = X.to(device)
            Y = Y.to(device)
            if args.fuse:
                for p in range(len(p_order_approx)):
                    outputs[p][s][args.repId - 1] = rjsd_fuse(X, Y, seed, order_approx = p_order_approx[p], number_permutations=args.permTestSize, alpha = args.significance)
            else:
                X_train, X_test = X[:len(X)//2], X[len(X)//2:]
                Y_train, Y_test = Y[:len(Y) // 2], Y[len(Y) // 2:]
            
                for p in range(len(p_order_approx)):
                    RJSD_est = RJSD_estimator(isotropic = False, order_approx=p_order_approx[p], deep = args.deep)
                    RJSD_est.fit(X_train, Y_train, epochs = 1000, lr = 0.0005, verbose=False)
                    outputs[p][s][args.repId - 1] = RJSD_est.permutation_test(X_test, Y_test, significance= args.significance,permutations=args.permTestSize, seed = seed)
    np.savez(fname, *outputs)

if __name__ == "__main__":
    run()