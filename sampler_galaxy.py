from galaxy_mnist import GalaxyMNIST, GalaxyMNISTHighrez
import torch
import torch
import numpy as np

def load_images_list(highres):
    if highres:
        dataset = GalaxyMNISTHighrez(  # [3, 224, 224]
            root='./galaxy_data',
            download=True,
            train=False,
        )
    else:
        dataset = GalaxyMNIST(  # [3, 64, 64]
            root='./galaxy_data',
            download=True,
            train=False,
        )

    (custom_train_images, custom_train_labels), (custom_test_images, custom_test_labels) = dataset.load_custom_data(test_size=0.5, stratify=True) 
    images = torch.cat((custom_train_images, custom_test_images))
    labels = torch.cat((custom_train_labels, custom_test_labels))

    images_list = (
        images[torch.where(labels == 3, True, False)].numpy(),
        images[torch.where(labels == 2, True, False)].numpy(),
        images[torch.where(labels == 1, True, False)].numpy(),
        images[torch.where(labels == 0, True, False)].numpy(),    
    )
    
    return images_list



def sampler_galaxy(key, m, n, corruption, images_list):
    """
    For X: we sample uniformly from images with labels 3, 2, 1.
    For Y: with probability 'corruption' we sample uniformly from images with labels 3, 2, 1.
           with probability '1 - corruption' we sample uniformly from images with labels 0.
    """
    images_0, images_1, images_2, images_3 = images_list
    
    # X
    # set torch and numpy seeds
    torch.manual_seed(key)
    np.random.seed(key)
    choice = np.random.choice(3, size=m)
    m_0 = np.sum(choice == 0)  # m = m_0 + m_1 + m_2
    m_1 = np.sum(choice == 1)
    m_2 = np.sum(choice == 2)
    indices_0 = np.random.permutation(images_0.shape[0])[:m_0]
    indices_1 = np.random.permutation(images_1.shape[0])[:m_1]
    indices_2 = np.random.permutation(images_2.shape[0])[:m_2]
    X = np.concatenate((images_0[indices_0], images_1[indices_1], images_2[indices_2]))
    X = np.random.permutation(X)
        
    # Y
    np.random.seed(key)
    choice = np.random.choice(4, size=n, p=[(1-corruption) / 3, (1-corruption) / 3, (1-corruption) / 3, corruption])
    n_0 = np.sum(choice == 0)  # n = n_0 + n_1 + n_2 + n_3
    n_1 = np.sum(choice == 1)
    n_2 = np.sum(choice == 2)
    n_3 = np.sum(choice == 3)
    indices_0 = np.random.permutation(images_0.shape[0])[:n_0]
    indices_1 = np.random.permutation(images_1.shape[0])[:n_1]
    indices_2 = np.random.permutation(images_2.shape[0])[:n_2]
    indices_3 = np.random.permutation(images_3.shape[0])[:n_3]
    Y = np.concatenate((images_0[indices_0], images_1[indices_1], images_2[indices_2], images_3[indices_3]))
    Y = np.random.permutation(Y)
    # Convert X and Y to float
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    # # Normalize X and Y between -1 and 1
    # X = X / 255 * 2 - 1
    # Y = Y / 255 * 2 - 1
    return torch.from_numpy(X), torch.from_numpy(Y)
