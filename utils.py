
import torch
import numpy as np

# imports from representation-itl library
import repitl.kernel_utils as ku
import repitl.matrix_itl as itl
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.distributions as dist


class RJSD_estimator():
    def __init__(self, approx = True, order_approx = 4, isotropic = True, sigma = 1.0, sigma0= 1.0, alpha = 1.0, alpha0 = 1.0, epsilon = 1e-10, deep = False, is_image = False):
        self.sigma = sigma
        self.sigma0 = sigma0 # sigma at the output layer of the deep network
        # alpha and alpha0 are the coefficient to multiply the mean squared distance for the Gaussian kernel
        self.alpha = alpha
        self.alpha0 = alpha0 
        self.epsilon = epsilon
        self.approx = approx
        self.order_approx = order_approx
        self.isotropic = isotropic
        self.sigma_inverse = None
        self.deep = deep
        self.is_image = is_image
    def fit(self, X, Y, epochs = 100, lr = 0.1, batch_size = 500, verbose = False, logger = None, validation = False, X_val = None, Y_val = None):
        if self.deep:
            self.fit_deep(X,Y,epochs = epochs, lr = lr, batch_size = batch_size, verbose = verbose, logger = logger, validation = validation, X_val = X_val, Y_val = Y_val)
        else:
            N = X.shape[0]
            M = Y.shape[0]
            # Creating the mixture of the two distributions
            Z = torch.cat((X,Y))

            if self.isotropic:
                sigma = torch.tensor(self.sigma, device = X.device).clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([sigma], lr=lr)
            else:
                sigma_ = self.sigma*torch.eye(X.shape[1], device=X.device)
                sigma_inverse = sigma_.inverse().clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([sigma_inverse], lr=lr)

            for i in range(epochs):
                optimizer.zero_grad()
                if self.isotropic:
                    Kx = ku.gaussianKernel(X,X, sigma)
                    Ky = ku.gaussianKernel(Y,Y, sigma)
                    Kz = ku.gaussianKernel(Z,Z, sigma)
                else:
                    X_ = X @ sigma_inverse
                    Y_ = Y @ sigma_inverse
                    Kx = ku.gaussianKernel(X_,X_,1)
                    Ky = ku.gaussianKernel(Y_,Y_,1)
                    Kz = ku.gaussianKernel(torch.cat((X_,Y_),dim=0),torch.cat((X_,Y_),dim=0),1)
                divergence = -1*RJSD(Kx,Ky,Kz, approx = self.approx, order_approx = self.order_approx)  
                divergence.backward()
                optimizer.step()
                if verbose and i % 50 == 0:
                    print('Iteration: {}, Divergence: {}'.format(i,-1*divergence.item()))
            if self.isotropic:
                self.sigma = sigma.detach().item()
            else:
                self.sigma_inverse = sigma_inverse.detach()
    def forward(self, X, Y):
        
        with torch.no_grad():
            if self.deep:
                return self.forward_deep(X,Y)
            if self.isotropic:
                Kx = ku.gaussianKernel(X,X, self.sigma)
                Ky = ku.gaussianKernel(Y,Y, self.sigma)
                Kz = ku.gaussianKernel(torch.cat((X,Y),dim=0),torch.cat((X,Y),dim=0), self.sigma)
            else:
                X_ = X @ self.sigma_inverse
                Y_ = Y @ self.sigma_inverse
                Kx = ku.gaussianKernel(X_,X_,1)
                Ky = ku.gaussianKernel(Y_,Y_,1)
                Kz = ku.gaussianKernel(torch.cat((X_,Y_),dim=0),torch.cat((X_,Y_),dim=0),1)
            JSD = RJSD(Kx,Ky,Kz, approx=self.approx, order_approx = self.order_approx)
        return JSD
    def fit_deep(self, X, Y, epochs = 1000, lr = 0.05, verbose = False, batch_size = 500, logger = None, validation = False, X_val = None, Y_val = None):
        # batch size only for images
        # Adapting the code from the deep MMD paper
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)

        if not self.is_image:
            batch_size = X.shape[0]
            x_in = X.shape[1]
            H = 50 # number of neurons in the hidden layer
            x_out = 50 # number of neurons in the output layer
            model_u = ModelLatentF(x_in, H, x_out)
            model_u.to(X.device)
            epsilonOPT = torch.tensor(self.epsilon, device = X.device, dtype = X.dtype)
            epsilonOPT.requires_grad = True
            sigmaOPT = torch.tensor(self.sigma, device = X.device, dtype = X.dtype)  # default: np.sqrt(np.random.rand(1) * 0.3
            sigmaOPT.requires_grad = True
            sigma0OPT = torch.tensor(self.sigma0, device = X.device, dtype = X.dtype) # default: np.sqrt(np.random.rand(1) * 0.002)
            sigma0OPT.requires_grad = True
        else:
            model_u = Featurizer()
            model_u.to(X.device)
            epsilonOPT = torch.log(torch.tensor(self.epsilon, device = X.device, dtype = X.dtype)) #  torch.log(epsilon) they use this initialization
            epsilonOPT.requires_grad = True
            # sigmaOPT = torch.tensor(self.sigma, device = X.device, dtype = X.dtype) # 1000 np.sqrt(2 * 32 * 32)
            # sigmaOPT.requires_grad = True
            # sigma0OPT = torch.tensor(self.sigma0, device = X.device, dtype = X.dtype)  #np.sqrt(0.005) # the values originally used in deep MMD are too small
            # sigma0OPT.requires_grad = True
            kernel_input = GaussianKernel(sigma = self.sigma, learn_sigma=True, track_running_stats = False)
            kernel_input.to(X.device)
            kernel_deep = GaussianKernel(sigma = self.sigma0, learn_sigma=True, track_running_stats = False) # No need to optimze sigma0
            kernel_deep.to(X.device)


        # Dataloader
        
        dataloader_X = torch.utils.data.DataLoader(
                X,
                batch_size= batch_size,
                shuffle=True,
        )
        dataloader_Y = torch.utils.data.DataLoader(
                Y,
                batch_size= batch_size,
                shuffle=True,
        )
        # Setup optimizer for training deep kernel
        # optimizer_u = torch.optim.Adam(list(model_u.parameters())+[sigmaOPT]+[sigma0OPT]+[epsilonOPT], lr=lr) # 
        optimizer_u = torch.optim.Adam(list(model_u.parameters())+list(kernel_input.parameters())+ list(kernel_deep.parameters())+[epsilonOPT], lr=lr)
        # Train deep kernel to maximize test power
        JSD_max = 0
        count = 0
        for epoch in range(epochs):
            for i, (X_batch, Y_batch) in enumerate(zip(dataloader_X, dataloader_Y)):
                # Compute epsilon, sigma and sigma_0
                optimizer_u.zero_grad()
                ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
                # sigma = sigmaOPT 
                # sigma0_u = sigma0OPT 
                # Compute output of the deep network
                X_deep = model_u(X_batch)
                Y_deep = model_u(Y_batch)
                # just for initialization purposes, let's check the median distance
                # XY = torch.cat((X_deep, Y_deep), dim=0)
                # print('Median distance sigma0: ', torch.median(Pdist2(XY, XY)).item())

                # Compute kernels
                #  Kz_deep = gaussianKernel(torch.cat((X_deep, Y_deep), dim=0), torch.cat((X_deep, Y_deep), dim=0), sigma0_u)
                Kz_deep = kernel_deep(torch.cat((X_deep, Y_deep), dim=0))
                # reshape X_batch and Y_batch
                N1 = len(X_batch)
                # Reshaping to compute kernel (This is done in case of images, for other data types, this is not necessary)
                X_batch_flat = X_batch.contiguous().view(X_batch.shape[0], -1)
                Y_batch_flat = Y_batch.contiguous().view(Y_batch.shape[0], -1)
                # Kz = gaussianKernel(torch.cat((X_batch_flat, Y_batch_flat), dim=0), torch.cat((X_batch_flat, Y_batch_flat), dim=0), sigma)
                Kz = kernel_input(torch.cat((X_batch_flat, Y_batch_flat), dim=0))
                # Combined kernels
                Kz_mix = (1-ep)*Kz_deep*Kz + ep*Kz
                
                Kx_mix = Kz_mix[:N1, :N1]
                Ky_mix = Kz_mix[N1:, N1:]
                # Compute JSD
                JSD = -1*RJSD(Kx_mix, Ky_mix, Kz_mix, approx = True, order_approx = self.order_approx)
                # Update deep kernel
                JSD.backward()
                optimizer_u.step()
                if verbose and epoch % 50 == 0:
                    print('Iteration: {}, Divergence: {}'.format(epoch,-1*JSD.item()))
            if validation and X_val is not None and Y_val is not None:
                self.model = model_u
                self.epsilon = epsilonOPT
                self.kernel_input = kernel_input
                self.kernel_deep = kernel_deep
                JSD_train = self.forward_deep(X, Y)
                JSD_val = self.forward_deep(X_val, Y_val)
                Z_val = torch.cat([X_val, Y_val], dim=0)
                Z_val = Z_val[torch.randperm(Z_val.shape[0])]
                JSD_perm = self.forward_deep(Z_val[:N1].detach(), Z_val[N1:].detach())
                # JSD_diff = JSD_val - JSD_perm
                # Check validation for early stopping
                if JSD_val.item() > JSD_max:
                    JSD_max = JSD_val.item()
                    best_model = model_u
                    best_epsilon = epsilonOPT
                    best_kernel_input = kernel_input
                    best_kernel_deep = kernel_deep
                    count = 0
                else:
                    count += 1
                if count == 20:
                    break

                if logger is not None:
                    # log according to the order approx
                    log_dict = {
                        "epoch": epoch,
                        str(self.order_approx) + "/Dtrain": JSD_train.item(),
                        str(self.order_approx) + "/Dval": JSD_val.item(),
                        str(self.order_approx) + "/Dperm": JSD_perm.item()
                    }
                    logger.log(log_dict)
        if validation and JSD_max > 0:
            self.model = best_model
            self.epsilon = best_epsilon
            self.kernel_input = best_kernel_input
            self.kernel_deep = best_kernel_deep
        else:
            self.model = model_u
            self.epsilon = epsilonOPT
            self.kernel_input = kernel_input
            self.kernel_deep = kernel_deep
    def forward_deep(self, X, Y):
        self.model.eval()
        with torch.no_grad():
            ep = torch.exp(self.epsilon)/(1+torch.exp(self.epsilon))
            N1 = len(X)
            X_deep = self.model(X)
            Y_deep = self.model(Y)
            # Kz_deep = gaussianKernel(torch.cat((X_deep, Y_deep), dim=0), torch.cat((X_deep, Y_deep), dim=0), self.sigma0)
            Kz_deep = self.kernel_deep(torch.cat((X_deep, Y_deep), dim=0))
            # Reshaping to compute kernel
            X_flat = X.contiguous().view(X.shape[0], -1)
            Y_flat = Y.contiguous().view(Y.shape[0], -1)
            # Kz = gaussianKernel(torch.cat((X_flat, Y_flat), dim=0), torch.cat((X_flat, Y_flat), dim=0), self.sigma)
            Kz = self.kernel_input(torch.cat((X_flat, Y_flat), dim=0))
            Kz_mix = (1-ep)*Kz_deep*Kz + ep*Kz
            Kx_mix = Kz_mix[:N1, :N1]
            Ky_mix = Kz_mix[N1:, N1:]
            JSD = RJSD(Kx_mix, Ky_mix, Kz_mix, approx = self.approx, order_approx = self.order_approx)
        return JSD

    def permutation_test(self,X,Y,significance = 0.05, permutations = 100, seed = 0):
        torch.manual_seed(seed)
        N_X = X.shape[0]
        N_Y = Y.shape[0]
        Z = torch.cat([X, Y], dim=0)
        jsd_null = []

        for i in range(permutations):
            Z = Z[torch.randperm(Z.shape[0])]
            if self.deep:
                div = self.forward_deep(Z[:N_X].detach(), Z[N_X:].detach())
            else:
                div = self.forward(Z[:N_X].detach(), Z[N_X:].detach())
            jsd_null.append(div)    
        jsd_null = torch.tensor(jsd_null)
        thr_jsd = torch.quantile(jsd_null, (1 - significance))
        if self.deep:
            JSD = self.forward_deep(X,Y)
        else:
            JSD = self.forward(X,Y)
        return 1 if JSD >= thr_jsd else 0
    
# Model used in deep MMD paper
class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant
# Define the deep network for MMD-D
class Featurizer(nn.Module):
    def __init__(self, channels = 3 , img_size = 64 ):
        super(Featurizer, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)] #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False), # 3: channels
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 300))
                # Initialize weights
        for m in self.adv_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, img):
        out = self.model(img)
        out = out.contiguous().view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature


def gaussianKernel(X, Y, sigma):
    Dxy = Pdist2(X, Y)
    K = torch.exp(-Dxy / (2 * sigma ** 2))
    # ensure the diagonal is 1 by replacing the diagonal elements of K with 1 (avoiding numerical issues)
    K = K - torch.diag(K.diag()) + torch.eye(K.shape[0], device = X.device)
    return K

from typing import Optional
class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix
    @author: Junguang Jiang
    @contact: JiangJunguang1123@outlook.com
    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1., learn_alpha: Optional[bool] = False, learn_sigma: Optional[bool] = False):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.nn.Parameter(torch.tensor(sigma * sigma),requires_grad=learn_sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=learn_alpha)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)
        l2_distance = torch.cdist(X, X, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        l2_distance_square = l2_distance ** 2

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))



def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def sampler_mixture(
    seed,
    m,
    n,
    d,
    mu=20,
    std_1=1,
    std_2=2,
):
    """
    Sampler Mixture of Gaussians.

    Returns samples X of shape (m, d) and Y of shape (n, d).
    
    Mixture of four Gaussians with means 
        (mu, mu), (-mu, mu), (mu, -mu), (-mu, -mu)
    for the first two dimensions and zeros for the other dimensions,
    and standard deviations
        std_1, std_1, std_1, std_X_Y 
    where std_X_Y is std_1 for X and std_2 for Y.

    Parameters
    ----------
    seed : int
        Seed for torch random number generator.
    m : int
        Number of samples for X.
    n : int
        Number of samples for Y.
    d: int
        Dimension for both X and Y (greater of equal to 2).
    mu: scalar
        mean parameter of the Gaussians as explained above.
    std_1: scalar
        standard deviation parameter for the Gaussians as explained above.
    std_2: scalar
        standard deviation parameter for the Gaussians as explained above.

    Returns
    -------
    X : torch.Tensor
        Tensor of shape (m, d).
    Y : torch.Tensor
        Tensor of shape (n, d).
    """
    assert d >= 2
    
    def sampler_normal(seed, mu_1, mu_2, std, N, d):
        mean = torch.zeros(d)
        mean[0] = mu_1
        mean[1] = mu_2
        cov = torch.eye(d) * std ** 2
        normal_dist = dist.MultivariateNormal(mean, cov)
        return normal_dist.sample((N, ))
    
    torch.manual_seed(seed)
    
    # X
    choice = torch.randint(4, (m, ))
    m_0 = torch.sum(choice == 0)  # m = m_0 + m_1 + m_2 + m_3
    m_1 = torch.sum(choice == 1)
    m_2 = torch.sum(choice == 2)
    m_3 = torch.sum(choice == 3)
    mu_pairs = ((mu, mu), (-mu, mu), (mu, -mu), (-mu, -mu))
    std = std_1
    samples_0 = sampler_normal(seed, *mu_pairs[0], std, m_0, d)
    samples_1 = sampler_normal(seed, *mu_pairs[1], std, m_1, d)
    samples_2 = sampler_normal(seed, *mu_pairs[2], std, m_2, d)
    samples_3 = sampler_normal(seed, *mu_pairs[3], std, m_3, d)
    X = torch.cat((samples_0, samples_1, samples_2, samples_3))
    X = X[torch.randperm(X.size(0))]
    
    # Y
    choice = torch.randint(4, (n, ))
    n_0 = torch.sum(choice == 0)  # n = n_0 + n_1 + n_2 + n_3
    n_1 = torch.sum(choice == 1)
    n_2 = torch.sum(choice == 2)
    n_3 = torch.sum(choice == 3)
    mu_pairs = ((mu, mu), (-mu, mu), (mu, -mu), (-mu, -mu))
    std = std_1
    std_different = std_2
    samples_0 = sampler_normal(seed, *mu_pairs[0], std, n_0, d)
    samples_1 = sampler_normal(seed, *mu_pairs[1], std, n_1, d)
    samples_2 = sampler_normal(seed, *mu_pairs[2], std, n_2, d)
    samples_3 = sampler_normal(seed, *mu_pairs[3], std_different, n_3, d)
    Y = torch.cat((samples_0, samples_1, samples_2, samples_3))
    Y = Y[torch.randperm(Y.size(0))]
    
    return X, Y

def vonNeumannEntropy(K, normalize = True, rank = None, retrieve_rank = False):
    ek, _ = torch.linalg.eigh(K)
    if rank is None:
        N = len(ek)
        lambda1 = ek[-1] # Largest eigenvalue
        rtol = lambda1*N*torch.finfo(ek.dtype).eps
        mk = torch.gt(ek, 0)
        mek = ek[mk]
    elif rank < K.shape[0]:
        ek_lr = torch.zeros_like(ek)
        ek_lr[-rank:] = ek[-rank:]
        ek_lr = ek_lr/ek_lr.sum() 
        mk = torch.gt(ek_lr, 0)
        mek = ek_lr[mk]

    if normalize:
        mek = mek/mek.sum()   
    H = -1*torch.sum(mek*torch.log(mek))
    if retrieve_rank:
        rank = compute_rank(ek)
        return H, rank
    return H
def compute_rank(eigenvalues):
    # Similar to pytorch implementation
    N = len(eigenvalues)
    eigenvalues = torch.abs(eigenvalues)
    lambda1 = eigenvalues[-1] # Largest eigenvalue
    rtol = N*torch.finfo(eigenvalues.dtype).eps
    rank = torch.sum(eigenvalues > rtol*lambda1)
    return rank
# Taylor expansion
def matrix_log(Q, order=4):
    n = Q.shape[0]
    Q = Q - torch.eye(n).detach().to(Q.device)
    cur = Q
    res = torch.zeros_like(Q).detach().to(Q.device)
    for k in range(1, order + 1):
        if k % 2 == 1:
            res = res + cur * (1. / float(k))
        else:
            res = res - cur * (1. / float(k))
        cur = cur @ Q
    return res

def vonNeumannEntropy_approx(K, order=4):
    K = K/K.trace()
    return torch.trace(- K @ matrix_log(K, order))

def permuteGram(K, seed = None):
    """
    Randomly permutes the rows and columns of a square matrix
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    assert K.shape[0] == K.shape[1], f"matrix dimensions must be the same"
    idx = torch.randperm(K.shape[0])
    K = K[idx, :]
    K = K[:, idx]
    return K




def RJSD(Kx,Ky,Kz,bias_correction = None, rank_approx = 'default', tikhonov_reg = 1e-2, seed = None, approx = False, order_approx = 4):
    N = Kx.shape[0]
    M = Ky.shape[0]
    if not approx: 
        if not bias_correction:
            Hx = vonNeumannEntropy(Kx)
            Hy = vonNeumannEntropy(Ky)
            Hz = vonNeumannEntropy(Kz)
        elif bias_correction == 'subsampling':
            Hx = vonNeumannEntropy(Kx)
            Hy = vonNeumannEntropy(Ky)
            # subsampling Kz
            Kz  = permuteGram(Kz, seed = seed)
            maxNM = max(N,M)
            Kz = Kz[:maxNM,:maxNM]
            Hz = vonNeumannEntropy(Kz)
        elif bias_correction == 'low_rank':
            Hx,rankKx = vonNeumannEntropy(Kx,retrieve_rank = True)
            Hy, rankKy = vonNeumannEntropy(Ky,retrieve_rank = True)
            rank_approx = max(rankKx,rankKy) + 10
            Hz = vonNeumannEntropy(Kz, rank = rank_approx) # the threshold is 2*eps because Kz has twice as many eigenvalues
        elif bias_correction == 'low_rank_all':
            rank = rank_approx
            _, evx, _ = torch.svd_lowrank(Kx, q=rank, niter=3)
            _, evy, _ = torch.svd_lowrank(Ky, q=rank, niter=3)
            _, evz, _ = torch.svd_lowrank(Kz, q=rank, niter=3)
            evx = evx[evx > 0]
            evy = evy[evy > 0]
            evz = evz[evz > 0]
            evx = evx/evx.sum()
            evy = evy/evy.sum()
            evz = evz/evz.sum()
            Hx = -torch.sum(evx*torch.log(evx))
            Hy = -torch.sum(evy*torch.log(evy))
            Hz = -torch.sum(evz*torch.log(evz))
        elif bias_correction == 'tikhonov':
            # get the average value of Kz without the diagonal
            avg = (Kz.nansum() - Kz.trace())/((N+M)**2 - (N+M))
            # rho =  torch.exp(-(torch.tan(np.pi/2*avg)**2)/tikhonov_reg)
            rho =  torch.exp(-avg/tikhonov_reg)
            Kz_reg = (1-rho)/(N+M)*Kz + rho/(N+M)*torch.eye(N+M, device=Kz.device)
            Kx_reg = (1-rho)/N*Kx + rho/(N+M)*torch.eye(N, device=Kx.device)
            Ky_reg = (1-rho)/M*Ky + rho/(N+M)*torch.eye(M, device=Ky.device)
            remainder_X = M*rho/(N+M)*torch.log(rho/(N+M))
            remainder_Y = N*rho/(N+M)*torch.log(rho/(N+M))
            Hx = vonNeumannEntropy(Kx_reg, normalize = False) - remainder_X
            Hy = vonNeumannEntropy(Ky_reg, normalize = False) - remainder_Y
            Hz = vonNeumannEntropy(Kz_reg)
    else:
        Hx = vonNeumannEntropy_approx(Kx, order = order_approx)
        Hy = vonNeumannEntropy_approx(Ky, order = order_approx)
        Hz = vonNeumannEntropy_approx(Kz, order = order_approx)
    JSD =  (Hz - (N/(N+M)*Hx + M/(N+M)*Hy))
    return JSD



def deep_JSD(X,Y,model):
    phiX = model(X)
    phiY = model(Y)
    # Creating the mixture of both distributions
    # phiZ =  torch.cat((phiX,phiY))
    covX = torch.matmul(torch.t(phiX),phiX)
    covY = torch.matmul(torch.t(phiY),phiY)
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hz = vonNeumannEntropy((covX+covY)/2)
    JSD =  (Hz - 0.5*(Hx + Hy))
    return JSD

def RJSD_mapping(phiX,phiY):
    covX = torch.matmul(torch.t(phiX),phiX)
    covY = torch.matmul(torch.t(phiY),phiY)
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hz = vonNeumannEntropy((covX+covY)/2)
    JSD =  (Hz - 0.5*(Hx + Hy))
    return JSD

def JSD_cov(covX,covY):
    Hx = vonNeumannEntropy(covX)
    Hy = vonNeumannEntropy(covY)
    Hz = vonNeumannEntropy((covX+covY)/2)
    JSD =  (Hz - 0.5*(Hx + Hy))
    return JSD


def mmd(x, y, sigma):
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value, pass none to not get p-value
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    k = torch.exp((-1 / (2 * sigma ** 2)) * dists ** 2)
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    # mmd unbiased does not sum the diagonal terms.
    mmd = (k_x.sum() - n) / (n * (n - 1)) + (k_y.sum() - m) / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd


