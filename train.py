# --model DeepJSCC --dataset cifar10 --local_epoch 5 --comm_rounds 500 --n_bit 8 --m_bit 16 --batch_size 64 --learning_rate 5e-4 --num_clients 20 --schedulingsize 5
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from train_argument import parser, print_args
from fractions import Fraction
import random
import copy
from time import time
from model.CNN_model import CNN_simple, DeepJSCC, DeepJSCC1
from utils import *
from Simulator import Simulator
from Split_Data import Non_iid_split, data_stats
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 一个客户时，调度大小也要改为1
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torchcong
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    save_folder = args.affix
    log_folder = os.path.join(args.log_root, save_folder)   # return a new path
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)
    
    setattr(args, 'log_folder', log_folder)     # setattr(obj, var, val) assign object attribute to its value, just like args.'log_folder' = log_folder
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, 'train', 'info')
    print_args(args, logger)                    # It prints arguments

    local_compute_energy = []
    epsilon_ks = np.random.uniform(1.5e3, 2.5e3, args.num_clients)  # cycles/sample, 随机生成
    f_ks = np.random.uniform(0.8e9, 1.2e9, args.num_clients)  # cycles/s, 随机生成
    zeta_ks = np.random.uniform(0.5e-28, 1.5e-28, args.num_clients)  # 有效电容系数, 随机生成

    for i in range(args.num_clients):
        # 计算本地计算能耗
        D_m = len(Non_iid_tr_datasets[i])
        epsilon_k = epsilon_ks[i]
        f_k = f_ks[i]
        zeta_k = zeta_ks[i]
        # tau_m_k = args.local_epoch * epsilon_k * D_m / f_k 用于计算本地计算时间
        # 所有轮次 每个设备本地计算的总能耗
        E_cmp_k_m = args.comm_rounds * args.local_epoch * zeta_k * epsilon_k * D_m * (f_k ** 2)
        local_compute_energy.append(E_cmp_k_m)

    # total_samples = sum(len(dataset) for dataset in Non_iid_tr_datasets)
    # print(total_samples)
    # total_samples=50000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.model == "simpleCNN":
        model = CNN_simple().to(device)
    elif args.model == 'DeepJSCC':
        model = DeepJSCC(c= 8, channel_type='AWGN', snr=snr, n_bit=args.n_bit, device=device).to(device)
    #   c=40时，压缩率为 (40*8*8)/ (3*32*32) = 0.8333
    elif args.model == 'DeepJSCC1':
        model = DeepJSCC1(c=8, channel_type='AWGN', snr=snr, n_bit=args.n_bit, device=device).to(device)
    else:
        print("未包含这个网络模型")
    trainer = Simulator(args, logger, local_tr_data_loaders, local_te_data_loaders, device)
    trainer.initialization(copy.deepcopy(model))
    # 每个客户的通信能耗初始化为0
    trainer.FedAvg()

    for i in range(args.num_clients):
        print(
            f'设备 {i + 1}: 本地计算能耗: {local_compute_energy[i]:.2e} J')

if __name__ == '__main__':
    args = parser()
    print_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_classes = 10
    if args.dataset =='mnist':
        # 默认mnist
        tr_dataset = torchvision.datasets.MNIST(args.data_root,
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)

        # evaluation during training
        # MNIST数据集的输入通道数为1，黑色图片
        te_dataset = torchvision.datasets.MNIST(args.data_root,
                                        train=False,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
    elif args.dataset =='cifar10':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
        tr_dataset = torchvision.datasets.CIFAR10(args.data_root,
                                                  train=True,
                                                  transform=transform,
                                                  download=True)

        # evaluation during training
        # MNIST数据集的输入通道数为1，黑色图片
        te_dataset = torchvision.datasets.CIFAR10(args.data_root,
                                                  train=False,
                                                  transform=transform,
                                                  download=True)
    else:
        pass
    # 通过从参数为 0.1 的狄利克雷分布中分配标签，以非独立同分布的方式在设备上分布训练数据集
    # 非独立同分布指的是不同客户端拥有的每个类别的样本数量可能不同，即标签服从非独立同分布
    Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
            num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)

    local_tr_data_loaders = [DataLoader(dataset, num_workers = 0,
                                        batch_size = args.batch_size,
                                        shuffle = False)
                    for dataset in Non_iid_tr_datasets]
    local_te_data_loaders = [DataLoader(dataset, num_workers = 0,
                                        batch_size = args.batch_size,
                                        shuffle = False)
                    for dataset in Non_iid_te_datasets]

    client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
    client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)

    while 1 in np.remainder(client_total_samples, args.batch_size) or 1 in np.remainder(client_total_te_samples, args.batch_size): #There should be more than one sample in a batch
            Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
            num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)
            client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
            client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)

    same_seeds(2048)
    snr = 19
    main(args)

