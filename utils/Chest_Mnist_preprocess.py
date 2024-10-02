from medmnist import INFO, Evaluator
from medmnist import ChestMNIST
import numpy as np

def data_split(proportions,indexes,idx_batch,N,n_nets):
    np.random.shuffle(indexes)
    print(proportions)
    ## Balance
    proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
    print(proportions)
    proportions = proportions/proportions.sum()
    
    proportions = (np.cumsum(proportions)*len(indexes)).astype(int)[:-1]
    print(proportions)
    idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(indexes,proportions))]
    return idx_batch
        
if __name__  == '__main__':
    train_dataset = ChestMNIST(split='train', transform=None, download=False,as_rgb= True, root ='././data/ChestMnist')
    X_train_total = train_dataset.imgs
    Y_train_total = np.argmax(train_dataset.labels, axis=1)
    Y_train_total[Y_train_total>0] = 1
    test_dataset = ChestMNIST(split='test', transform=None, download=False,as_rgb= True, root ='././data/ChestMnist')
    X_test_total = test_dataset.imgs
    Y_test_total = np.argmax(test_dataset.labels , axis=1)
    Y_test_total[Y_test_total>0] = 1

    print(X_train_total.shape,Y_train_total.shape)
    labels = np.unique(Y_train_total)
    print(labels)
    indexes = [[] for i in range(len(labels))]
    indexes_test = [[] for i in range(len(labels))]
    sanity = 0
    for i in labels:
        indexes[i] = np.where(Y_train_total==i)[0]
        indexes_test[i] = np.where(Y_test_total==i)[0]
        sanity += len(indexes[i])
        print(i,len(indexes[i]),sanity)
        
    min_size = 0
    N = Y_train_total.shape[0]
    N_test = Y_test_total.shape[0]
    net_dataidx_map = {}
    n_nets = 5
    alpha = 1
    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        idx_batch_test = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in labels:
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            idx_batch = data_split(proportions,indexes[k],idx_batch,N,n_nets)
            idx_batch_test = data_split(proportions,indexes_test[k],idx_batch_test,N_test,n_nets)
            for idx_j in idx_batch:
                print(len(idx_j))
            min_size = min([len(idx_j) for idx_j in idx_batch])
    print(np.unique(Y_train_total))
    outfile = './data/ChestMnist/train/'
    for i,i_s in enumerate(idx_batch):
        print(i,np.unique(Y_train_total[i_s]))
        label_client = Y_train_total[i_s]
        data_client = X_train_total[i_s]
        np.savez(outfile+'chunk_'+str(i), x=data_client, y=label_client)
    outfile = './data/ChestMnist/test/'
    for i,i_s in enumerate(idx_batch_test):
        print(i,np.unique(Y_test_total[i_s]))
        label_client = Y_test_total[i_s]
        data_client = X_test_total[i_s]
        np.savez(outfile+'chunk_'+str(i), x=data_client, y=label_client)
    # for i in np.sort(idx_batch[0]):
    #     print(Y_train_total[i])
    # print(Y_train_total[np.sort(idx_batch[0])[0]])
    # p = np.take(Y_train_total, np.sort(idx_batch[9]))
    # print(np.unique(p))
    # for idx in idx_batch:
    #     print(Y_train_total[idx[100]])
    #     break
    #     p = np.take(Y_train_total, idx, axis=0)
    #     print(p)
    #     for i in labels:
    #         print(len(np.where(p==i)[0]))
    #         print(Y_train_total[idx])
    # for j in range(n_nets):
    #     np.random.shuffle(idx_batch[j])
    #     net_dataidx_map[j] = idx_batch[j]

