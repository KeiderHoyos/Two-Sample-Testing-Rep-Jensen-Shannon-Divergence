import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import argparse
import wandb
from sampler_galaxy import load_images_list, sampler_galaxy

from utils import RJSD_estimator
parser = argparse.ArgumentParser()

parser.add_argument('-e', '--experiment-name', required=True, help='Experiment name for saved files', type=str)
parser.add_argument('-repetitions', '--repetitions', required=False,default = 200, help='Number of repetitions', type=int)
parser.add_argument('-nPerm', '--permTestSize', required= True, help='Number of permutations to compute power of test',type=int )
parser.add_argument('-sign', '--significance', required= True, help='significance level to reject the null hypothesis', type=float )
parser.add_argument('-datafolder', '--DATAFOLDER', required = True, type = str )
parser.add_argument('-epochs', '--epochs', required = False, default = 1000, type = int)
parser.add_argument('-lr', '--lr', required = False, default = 0.0002, type = float)
parser.add_argument('-batch_size', '--batch_size', required = False, default = 250, type = int)
parser.add_argument('-alpha', '--alpha', required = False, default = 1.0, type = float)
parser.add_argument('-alpha0', '--alpha0', required = False, default = 1.0, type = float)
parser.add_argument('-sigma', '--sigma', required = False, default = 1.0, type = float)
parser.add_argument('-sigma0', '--sigma0', required = False, default = 1.0, type = float)
parser.add_argument('-epsilon', '--epsilon', required = False, default = 1e-10, type = float)
parser.add_argument('-parallel', '--parallel', required = False, default = False, type = bool)
parser.add_argument('-rId', '--repId',required=False, help = 'repetition Id to store the results', type=int)
parser.add_argument('-deep', '--deep', required = False, default = False, type = bool)
parser.add_argument('-vary_size', '--vary_size', required = False, default = False, type = bool)
parser.add_argument('-is_image', '--is_image', required = False, default = False, type = bool)
parser.add_argument('-validation', '--validation', required = False, default = False, type = bool) 

args = parser.parse_args()

# If paralle each core will do one repetition

def run():
    # start a new wandb run to track this script
    wdb_logger = wandb.init(
        # set the wandb project where this run will be logged
        project="two_sample_testing",
        name = f"tst_lr_{args.lr}_sigma_{args.sigma}_sigma0_{args.sigma0}",
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        }
    )
    if args.parallel:
        fname = args.DATAFOLDER + '/' + str(args.experiment_name) + str(args.repId) + '.npz'
    else:
        fname = args.DATAFOLDER + '/' + str(args.experiment_name) + f"_lr_{args.lr}_sigma_{args.sigma}_sigma0_{args.sigma0}" + '.npz'
    images_list = load_images_list(highres=False)
    
    if not args.vary_size:
        p_order_approx =  np.array([1, 2, 4, 6, 10])
        repetitions = args.repetitions
        corruptions = (0.25,)#(0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4)
        sample_size = 500
        outputs = np.zeros((len(p_order_approx),len(corruptions), repetitions))
        outputs = outputs.tolist()
        if args.parallel:
            seed = 42 + (args.repId - 1)
        else:
            seed = 42 
        # set torch and numpy seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.parallel:
            reps = 1
        else:
            reps = repetitions
        # Check if a GPU is available and if not, use a CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for s in (range(len(corruptions))):
            
            corruption = corruptions[s]
            for i in range(reps):
                print('Corruption:', corruption)
                X, Y = sampler_galaxy(key = seed, m=sample_size, n=sample_size, corruption = corruption,images_list=images_list)
                X = X.to(device)
                Y = Y.to(device)
                X_train, X_test = X[:len(X)//2], X[len(X)//2:]
                Y_train, Y_test = Y[:len(Y) // 2], Y[len(Y) // 2:]
                if args.validation:
                    X_val, Y_val = sampler_galaxy(key = seed + 201, m=sample_size//2, n=sample_size//2, corruption = corruption,images_list=images_list)
                    X_val = X_val.to(device)
                    Y_val = Y_val.to(device)
                else:
                    X_val, Y_val = None, None
                if args.parallel:
                    seed += repetitions
                    i = args.repId - 1
                else:
                    seed += 1
                for p in range(len(p_order_approx)):
                    RJSD_est = RJSD_estimator(isotropic = False, order_approx=p_order_approx[p], sigma= args.sigma, sigma0=args.sigma0, epsilon = args.epsilon, deep = args.deep, is_image=args.is_image)
                    RJSD_est.fit(X_train, Y_train, batch_size = args.batch_size, epochs = args.epochs, lr = args.lr, verbose=False, logger = wdb_logger, validation = args.validation, X_val = X_val, Y_val = Y_val)
                    outputs[p][s][i] = RJSD_est.permutation_test(X_test, Y_test, significance= args.significance,permutations=args.permTestSize, seed = seed)
                wdb_logger.log({"power_test": np.max(np.mean(np.array(outputs),-1))})
    else: 
        repetitions = args.repetitions
        corruption = 0.15
        sample_sizes = (1500,)#(500, 1000, 1500, 2000, 2500)
        p_order_approx =  np.array([1, 2, 4, 6, 10])
        outputs = np.zeros((len(p_order_approx),len(sample_sizes), repetitions))
        outputs = outputs.tolist()
        if args.parallel:
            seed = 42 + (args.repId - 1)
        else:
            seed = 42 
        # set torch and numpy seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.parallel:
            reps = 1
        else:
            reps = repetitions

        # Check if a GPU is available and if not, use a CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for s,sample_size in enumerate(sample_sizes):
            print('Corruption:', corruption, ', sample size:', sample_size)
            for i in range(reps):
                X, Y = sampler_galaxy(key = seed, m=sample_size, n=sample_size, corruption = corruption,images_list=images_list)
                X = X.to(device)
                Y = Y.to(device)
                X_train, X_test = X[:len(X)//2], X[len(X)//2:]
                Y_train, Y_test = Y[:len(Y) // 2], Y[len(Y) // 2:]
                if args.validation:
                    X_val, Y_val = sampler_galaxy(key = seed + 201, m=sample_size//2, n=sample_size//2, corruption = corruption,images_list=images_list)
                    X_val = X_val.to(device)
                    Y_val = Y_val.to(device)
                else:
                    X_val, Y_val = None, None
                if args.parallel:
                    seed += repetitions
                    i = args.repId - 1
                else:
                    seed += 1
                for p in range(len(p_order_approx)):
                    print("Order Approximation:", p_order_approx[p])
                    RJSD_est = RJSD_estimator(isotropic = False, order_approx=p_order_approx[p], sigma = args.sigma, sigma0=args.sigma0, epsilon = args.epsilon, deep = args.deep, is_image=args.is_image)
                    RJSD_est.fit(X_train, Y_train, batch_size = args.batch_size, epochs = args.epochs, lr = args.lr, verbose=False, logger = wdb_logger, validation = args.validation, X_val = X_val, Y_val = Y_val)
                    outputs[p][s][i] = RJSD_est.permutation_test(X_test, Y_test, significance= args.significance,permutations=args.permTestSize, seed = seed)
                wdb_logger.log({"power_test": np.max(np.mean(np.array(outputs),-1))})
    np.savez(fname, *outputs)

if __name__ == "__main__":
    run()
