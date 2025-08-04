import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor
from time import time
import numpy as np
import copy
from model import CNN_model
from server import Server_Class
from Split_Data import Non_iid_split
from client import Client_Class
from utils import*
import matplotlib.pyplot as plt

class Simulator():
    def __init__(self, args, logger, local_tr_data_loaders, local_te_data_loaders, device):
        self.args = args
        self.logger = logger
        self.Clients_list = None
        self.Clients_list = None
        self.Server = None
        self.local_tr_data_loaders = local_tr_data_loaders
        self.local_te_data_loaders = local_te_data_loaders
        self.device = device

    def initialization(self, model):
        n_mse = nn.MSELoss(reduction='none').to(self.device)
        # loss = nn.CrossEntropyLoss()

        self.Server = Server_Class.Server(self.args, model)

        self.Clients_list = [Client_Class.Client(self.args, copy.deepcopy(self.Server.global_model), n_mse,
                                        client_id, tr_loader, te_loader, self.device, scheduler=None)
                                        for (client_id, (tr_loader, te_loader)) in enumerate(zip(self.local_tr_data_loaders, self.local_te_data_loaders))]
    def FedAvg(self):
        best_loss = 0
        best_psnr = 0
        loss_history = []   # 获得每轮的loss和 psnr，save the current average accuracy to the history
        psnr_history = []

        total_communication_energy = 0
        # 用于计算全部轮次客户通信能耗的总和
        for rounds in np.arange(self.args.comm_rounds):
            perround_communication_energy = 0
            # 用于计算每轮通信 所有客户通信能耗的总和
            sampled_clients = self.Server.sample_clients()
            # 联邦的训练过程：
            # 1.随机采样，返回采样到客户的索引 2.在采样得到的客户上训练 每个客户多次本地epoch训练完成后
            # 将模型参数上传到服务器，服务器聚合下发模型再次训练
            begin_time = time()
            avg_loss =[]        # 每轮测试的loss
            avg_psnr =[]        # 每轮测试的psnr
            self.logger.info("-"*30 + "Epoch start" + "-"*30)

            # 将模型广播给采样到的客户Set the current global model to sampled clients
            self.Server.broadcast(self.Clients_list, sampled_clients)
            # 每轮通信在采样的客户上先测试再训练
            # for client_idx in sampled_clients:
            #     loss, psnr = self.Clients_list[client_idx].local_test(client_idx , rounds)
            #
            #     # 获得每个客户的loss和psnr local_test得到的是每个客户，每个测试batch的平均值
            #     avg_loss.append(loss), avg_psnr.append(psnr)

            # 训练过程中 每轮通信每设备上传参数的通信能量消耗
            local_communication_energy_perclient = []
            for client_idx in sampled_clients:
                loss, psnr = self.Clients_list[client_idx].local_training(rounds)
                # 记录每个客户训练过程中的psnr和loss变化
                avg_loss.append(loss), avg_psnr.append(psnr)
                print("round{}-client{}-train | student loss:{} | student psnr:{}".format(rounds, client_idx, loss, psnr))

                local_communication_energy_perclient.append(self.Clients_list[client_idx].comm_energery_perround)

            # 每轮通信 所有客户上传参数的通信能耗总和
            perround_communication_energy += sum(local_communication_energy_perclient)

            total_communication_energy += perround_communication_energy

            # 聚合
            self.Server.aggregation(self.Clients_list, sampled_clients)

            # 在采样的客户上取平均，获得每轮的loss和psnr，作为每轮测试的最终值
            avg_loss_round = np.mean(avg_loss)
            avg_psnr_round = np.mean(avg_psnr)

            loss_history.append(avg_loss_round) # save the current average accuracy to the history
            psnr_history.append(avg_psnr_round)
            # 打印本轮：测试和训练时间、avg_loss、avg_psnr
            self.logger.info('round: %d, avg_loss: %.3f, avg_psnr: %.3f, spent: %.2f' %(rounds, avg_loss_round,
                                                                                               avg_psnr_round, time()-begin_time,))
            cur_loss = avg_loss_round
            cur_psnr = avg_psnr_round

            if cur_loss < best_loss:
                best_loss = cur_loss
            if cur_psnr > best_psnr:
                best_psnr = cur_psnr

        self.logger.info("联邦学习训练过程中的总通信能耗为：{:.2f}J".format(total_communication_energy))
        # 语义通信方法对比使用相同数据下的一个用户，若体现出联邦学习的作用用聚合后的数据和聚合前的数据进行对比
        # 200轮次训练结束后，进行最终的测试。Check final accuracy
        self.Server.broadcast(self.Clients_list, range(0, self.args.num_clients))
        final_loss =[]
        final_psnr =[]
        for client_idx, client in enumerate(self.Clients_list):
            # 为最后一轮进行测试
            loss , psnr = client.local_test(client_idx, rounds = self.args.comm_rounds)
            final_loss.append(loss)
            final_psnr.append(psnr)
            self.logger.info('client_id: %d , final loss: %.3f, final psnr: %.3f' %(
                             client_idx, loss, psnr))
        final_avg_loss = np.mean(final_loss)
        final_avg_psnr= np.mean(final_psnr)

        self.logger.info(">>>>> Training process finish")
        self.logger.info("Best test loss {:.4f}".format(best_loss))
        self.logger.info("Best test psnr {:.4f}".format(best_psnr))
        self.logger.info("Final test loss {:.4f}".format(final_avg_loss))
        self.logger.info("Final test psnr {:.4f}".format(final_avg_psnr))

        self.logger.info(">>>>> loss history during training")
        self.logger.info(loss_history)

        self.logger.info(">>>>> psnr history during training")
        self.logger.info(psnr_history)

        # 信噪比范围
        SNR_list = [1, 4, 7, 10, 13, 16, 19, 22, 25]
        avg_psnr_list = []
        avg_loss_list = []

        for snr in SNR_list:
            total_psnr = 0
            total_loss = 0

            print(f'Testing at SNR = {snr} dB')
            for client_idx, client in enumerate(self.Clients_list):
                # 得到每个客户的psnr和loss
                client_avg_loss, client_avg_psnr = client.snr_test(snr)
                total_psnr += client_avg_psnr
                total_loss += client_avg_loss

                print(
                    f'Client {client_idx} - Average PSNR: {client_avg_psnr:.2f} dB, Average Loss: {client_avg_loss:.4f}')

            avg_psnr = total_psnr / self.args.num_clients
            avg_loss = total_loss / self.args.num_clients
            avg_psnr_list.append(avg_psnr)
            avg_loss_list.append(avg_loss)

            print(f'Total Average PSNR at SNR {snr}: {avg_psnr:.2f} dB')
            print(f'Total Average Loss at SNR {snr}: {avg_loss:.4f}')

        # 创建第一个图表：SNR vs PSNR
        plt.figure(figsize=(10, 5))
        plt.plot(SNR_list, avg_psnr_list, marker='o', color='b', label='PSNR')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Average PSNR (dB)')
        plt.title('SNR vs PSNR')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 创建第二个图表：SNR vs Loss
        plt.figure(figsize=(10, 5))
        plt.plot(SNR_list, avg_loss_list, marker='o', color='r', label='Loss')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Average Loss')
        plt.title('SNR vs Loss')
        plt.legend()
        plt.grid(True)
        plt.show()