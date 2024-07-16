import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import argparse
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import os
import wandb


from utils import RJSD_estimator
parser = argparse.ArgumentParser()

parser.add_argument('-e', '--experiment-name', required=True, help='Experiment name for saved files', type=str)
parser.add_argument('-nPerm', '--permTestSize', required= True, help='Number of permutations to compute power of test',type=int )
parser.add_argument('-sign', '--significance', required= True, help='significance level to reject the null hypothesis', type=float )
parser.add_argument('-datafolder', '--DATAFOLDER', required = True, type = str )
parser.add_argument('-epochs', '--epochs', required = False, default = 1000, type = int)
parser.add_argument('-lr', '--lr', required = False, default = 0.0002, type = float)
parser.add_argument('-sigma', '--sigma', required = False, default = 0.1, type = float)
parser.add_argument('-sigma0', '--sigma0', required = False, default = 0.1, type = float)
parser.add_argument('-parallel', '--parallel', required = False, default = False, type = bool)
parser.add_argument('-rId', '--repId',required=False, help = 'repetition Id to store the results', type=int)
args = parser.parse_args()

# parameters
N1 = 1000
img_size = 64
batch_size = 100
K = 10
N = 100 # 100 originally in the paper (NUmber of different test sets)

def run():
    cwd = os.getcwd()
    print('Current working directory:', cwd)
    if args.parallel:
        repetitions = 1
    else:
        repetitions = K
    print('Repetitions:', repetitions)
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
    # Load the CIFAR 10 data and CIFAR 10.1
    if args.parallel:
        fname = args.DATAFOLDER + '/' + str(args.experiment_name) + str(args.repId) + '.npz'
    else:
        fname = args.DATAFOLDER + '/' + str(args.experiment_name) + f"_lr_{args.lr}_sigma_{args.sigma}_sigma0_{args.sigma0}" + '.npz'
    
    # seed = (args.repId - 1)*N
    p_order_approx = np.array([1, 2, 4, 6, 10])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    # Configure data loader
    print('Loading CIFAR10 data')
    dataset_test = datasets.CIFAR10(root='./cifar_data/cifar10', download=True,train=False,
                            transform=transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    print('creating dataloader')
    print('Length of dataset:', len(dataset_test))
    print(dataset_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000, shuffle=False, num_workers=2) # 10000
    # Obtain CIFAR10 images
    for i, (imgs, Labels) in enumerate(dataloader_test):
        print('i:', i)
        data_all = imgs
        label_all = Labels
    Ind_all = np.arange(len(data_all))
    print('CIFAR10 data loaded')
    # Obtain CIFAR10.1 images
    data_new = np.load('./cifar_data/cifar10.1_v4_data.npy')
    data_T = np.transpose(data_new, [0,3,1,2])
    ind_M = np.random.choice(len(data_T), len(data_T), replace=False)
    data_T = data_T[ind_M]
    TT = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans = transforms.ToPILImage()
    data_trans = torch.zeros([len(data_T),3,img_size,img_size])
    data_T_tensor = torch.from_numpy(data_T)
    for i in range(len(data_T)):
        d0 = trans(data_T_tensor[i])
        data_trans[i] = TT(d0)
    Ind_v4_all = np.arange(len(data_T))
    print('CIFAR10.1 data loaded')
    outputs = [[] for _ in range(len(p_order_approx))]
    seed = 0
    for kk in range(repetitions):
        print('Repetition:', kk)
        if args.parallel:
            kk = args.repId - 1
        
        torch.manual_seed(kk * 19 + N1)
        torch.cuda.manual_seed(kk * 19 + N1)
        np.random.seed(seed=1102 * (kk + 10) + N1)

        # Collect CIFAR10 images
        Ind_tr = np.random.choice(len(data_all), N1, replace=False)
        Ind_te = np.delete(Ind_all, Ind_tr)
        train_data = []
        for i in Ind_tr:
            train_data.append([data_all[i], label_all[i]])

        dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
        )

        # Collect CIFAR10.1 images
        np.random.seed(seed=819 * (kk + 9) + N1)
        Ind_tr_v4 = np.random.choice(len(data_T), N1, replace=False)
        Ind_te_v4 = np.delete(Ind_v4_all, Ind_tr_v4)
        New_CIFAR_tr = data_trans[Ind_tr_v4]
        New_CIFAR_te = data_trans[Ind_te_v4]
        
        # Run two-sample test on the training set
        # Fetch training data
        s1_tr = data_all[Ind_tr]
        s2_tr = data_trans[Ind_tr_v4]

        X_train = s1_tr
        Y_train = s2_tr
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)

        
        for p in range(len(p_order_approx)):
            print('p: ',p)
            RJSD_est = RJSD_estimator(isotropic = False, order_approx=p_order_approx[p], sigma= args.sigma, sigma0=args.sigma0, deep = True, is_image=True)
            RJSD_est.fit(X_train, Y_train, epochs = args.epochs, lr = args.lr, verbose=False)
            
            for k in range(N):
                print('p: ',p,'Repetition:', k)
                # Fetch test data
                np.random.seed(seed=1102 * (k + 1) + N1)
                data_all_te = data_all[Ind_te]
                N_te = len(data_trans) - N1
                Ind_N_te = np.random.choice(len(Ind_te), N_te, replace=False)
                s1_te = data_all_te[Ind_N_te]
                s2_te = data_trans[Ind_te_v4]
                X_test = s1_te
                Y_test = s2_te
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)
                # concatenate the split data 
                outputs[p].append(RJSD_est.permutation_test(X_test, Y_test, significance= args.significance,permutations=args.permTestSize, seed = 1102 * (k + 1) + N1))
            wdb_logger.log({"power_test_p_" + str(p): np.mean(np.array(outputs[p]))})
        wdb_logger.log({"power_test": np.max(np.mean(np.array(outputs),-1))})
    np.savez(fname, *outputs)

if __name__ == "__main__":
    run()