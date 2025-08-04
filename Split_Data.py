from torch.utils.data import Dataset
import numpy as np
import torch
import random

class Non_iid(Dataset):
    def __init__(self, x, y):
        # mnist数据集
        # self.x_data = x.unsqueeze(1).to(torch.float32)
        # self.y_data = y.to(torch.int64)


        # cifar10数据集
        self.x_data = x.transpose((0, 3, 1, 2))
        self.x_data = torch.from_numpy(self.x_data).to(torch.float32)
        y_tensor = torch.tensor(y)
        self.y_data = y_tensor.to(torch.int64)

        self.cuda_available = torch.cuda.is_available()
    
    #Return the number of data
    def __len__(self):
        return len(self.x_data)
    
    #Sampling
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        if self.cuda_available:
            return x.cuda(), y.cuda()
        else:
            return x, y


def data_stats(non_iid_datasets, num_classes, num_clients):

    client_data_counts = {client:{} for client in range(num_clients)}
    client_total_samples = []
    for client, data in enumerate(non_iid_datasets):
        total_sample = 0
        for label in range(num_classes):
            idx_label = len(np.where(data.y_data == label)[0])
            # client_data_counts[client].append(idx_label/data.__len__() * 100)
            label_sum = np.sum(idx_label)
            client_data_counts[client][label] = label_sum
            total_sample += label_sum
        client_total_samples.append(total_sample)

    return client_data_counts, client_total_samples

def Non_iid_split(num_classes, num_clients, tr_datasets, te_datasets, alpha):
    """
    Input: num_classes, num_clients, datasets, alpha
    Output: Dataset classes of the number of num_clients 
    """
    tr_idx_batch = [[] for _ in range(num_clients)]
    # 为每个客户生成一个空列表[]
    tr_data_index_map = {}
    te_idx_batch = [[] for _ in range(num_clients)]
    te_data_index_map = {}

    #for each calss in the training/test dataset
    for label in range(num_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients)) #It generates dirichichlet random variable with alpha over num_clients
        # 这行代码生成了一个Dirichlet分布的随机向量，proportions的和为1
        # 假设num_clients=3且alpha=0.1，np.random.dirichlet(np.repeat(0.1, 3))
        # 可能产生一个输出向量，例如[0.35, 0.55, 0.10]。这意味着第一个客户端将获得大约35%的数据，第二个客户端获得55%，第三个客户端获得10%。


        tr_idx_label = np.where(torch.tensor(tr_datasets.targets) == label)[0]  # 用于cifar10数据集
        # 获得这个标签下样本的索引 mnist数据集
        # tr_idx_label = np.where(tr_datasets.targets == label)[0]
        # np.where()函数返回一个索引
        # 这里使用了torch.tensor(tr_datasets.targets) == label来找到所有目标标签等于label的样本位置
        np.random.shuffle(tr_idx_label)
        # 打乱tr_idx_label，使得数据在分配前是随机排序的
        tr_proportions = (np.cumsum(proportions) * len(tr_idx_label)).astype(int)[:-1]
        # 这行代码将标签的数量分配给不同的client
        # np.cumsum(proportions)在计算累积和，如果proportions = [0.1, 0.2, 0.3, 0.4]，那么累积和将是[0.1, 0.3, 0.6, 1.0]。
        tr_idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(tr_idx_batch, np.split(tr_idx_label, tr_proportions))]
        te_idx_label = np.where(torch.tensor(te_datasets.targets) == label)[0] # 用于cifar10数据集
        # 根据tr_proportions中的分割点，使用np.split()将tr_idx_label分割成多个子列表，每个子列表代表一个客户端应该获得的样本索引。
        # 然后，这些索引被添加到tr_idx_batch的相应客户端列表中。
        # zip()函数将tr_idx_batch和np.split()返回的子列表进行迭代，并将它们组合成一个新的列表。
        # te_idx_label = np.where(te_datasets.targets == label)[0]
        np.random.shuffle(te_idx_label)
        te_proportions = (np.cumsum(proportions) * len(te_idx_label)).astype(int)[:-1]

        te_idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(te_idx_batch, np.split(te_idx_label, te_proportions))]
        
    for client in range(num_clients):
        np.random.shuffle(tr_idx_batch[client])
        # 先将数据打乱
        tr_data_index_map[client] = tr_idx_batch[client]
        # 获得每个client下的每个标签下的 索引
        te_data_index_map[client] = te_idx_batch[client]

    Non_iid_tr_datasets = []
    Non_iid_te_datasets = []
    for client in range(num_clients):
        tr_x_data = tr_datasets.data[tr_data_index_map[client]]
        # 获得这个索引下的clien数据
        tr_indices = tr_data_index_map[client]
        tr_y_data = [tr_datasets.targets[idx] for idx in tr_indices] # 用于cifar10数据集

        # mnist数据集
        # tr_y_data = tr_datasets.targets[tr_data_index_map[client]]
        Non_iid_tr_datasets.append(Non_iid(tr_x_data, tr_y_data))

        te_x_data = te_datasets.data[te_data_index_map[client]]
        te_indices = te_data_index_map[client]
        te_y_data = [te_datasets.targets[idx] for idx in te_indices]  # 用于cifar10数据集
        # te_y_data = te_datasets.targets[te_data_index_map[client]]
        Non_iid_te_datasets.append(Non_iid(te_x_data, te_y_data))

    return Non_iid_tr_datasets, Non_iid_te_datasets

    # for client in range(num_clients):
    #     tr_x_data = tr_datasets.data[tr_data_index_map[client]]
    #     tr_indices = tr_data_index_map[client]
    #     tr_y_data = [tr_datasets.targets[idx] for idx in tr_indices]
    #     # tr_y_data = tr_datasets.targets[tr_data_index_map[client]]
    #     Non_iid_tr_datasets.append(Non_iid(tr_x_data, tr_y_data))
    #
    #     te_x_data = te_datasets.data[te_data_index_map[client]]
    #     te_indices = te_data_index_map[client]
    #     te_y_data = [te_datasets.targets[idx] for idx in te_indices]
    #     Non_iid_te_datasets.append(Non_iid(te_x_data, te_y_data))
    #
    # return Non_iid_tr_datasets, Non_iid_te_datasets

    # for client in range(num_clients):
    #     tr_x_data = tr_datasets.data[tr_data_index_map[client]]
    #     tr_indices = tr_data_index_map[client]
    #     client_tr_targets = [tr_datasets.targets[idx] for idx in tr_indices]
    #     tr_y_data = tr_datasets.targets[tr_data_index_map[client]]
    #     te_indices = te_data_index_map[client]
    #     client_te_targets = [te_datasets.targets[idx] for idx in te_indices]
    #     Non_iid_tr_datasets.append(Non_iid(tr_x_data, tr_y_data))
    #     Non_iid_te_datasets.append(Non_iid(te_x_data, te_y_data))
    # return Non_iid_tr_datasets, Non_iid_te_datasets

